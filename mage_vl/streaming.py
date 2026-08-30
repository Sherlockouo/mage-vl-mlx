"""StreamMind gate for Mage-VL — proactive event-gated streaming (MLX port).

Ports streammind_gate.py: per-frame "silent vs speak" scoring for streaming
video. Pipeline per time step (frame):
    vision patches [B,T,P,D] --mean over P--> [B,T,D]
      -> PreNet (Linear + leaky_relu)
      -> VideoMamba (LayerNorm -> Mamba-1 mixer -> residual -> LayerNorm)
      -> PostNet (leaky_relu + Linear)
      -> ClsNet (4-layer Qwen3) -> [B,T,2] silent/speak logits

The classifier runs a length-2 (perception, target) sequence in the reference and
reads position 0; under causal attention position 0 sees only itself, so at
inference we run it on the perception token alone (length 1) — identical result,
and RoPE at position 0 is identity.

Mamba mixer adapted from mlx_lm.models.mamba (Mamba-1).
"""
import mlx.core as mx
import mlx.nn as nn

try:
    from ..qwen3.config import ModelConfig as Qwen3Config
    from ..qwen3.language import Qwen3Model
except ImportError:  # 顶级导入
    from mlx_vlm.models.qwen3.config import ModelConfig as Qwen3Config
    from mlx_vlm.models.qwen3.language import Qwen3Model

D_MODEL = 2560
D_INNER = 5120
D_STATE = 16
DT_RANK = 160
CONV_K = 4


def _leaky_relu(x, slope=0.01):
    return mx.where(x > 0, x, slope * x)


class MambaMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(D_MODEL, D_INNER * 2, bias=False)
        self.conv1d = nn.Conv1d(D_INNER, D_INNER, kernel_size=CONV_K, groups=D_INNER, bias=True, padding=0)
        self.x_proj = nn.Linear(D_INNER, DT_RANK + 2 * D_STATE, bias=False)
        self.dt_proj = nn.Linear(DT_RANK, D_INNER, bias=True)
        self.A_log = mx.zeros((D_INNER, D_STATE))
        self.D = mx.ones((D_INNER,))
        self.out_proj = nn.Linear(D_INNER, D_MODEL, bias=False)

    def _ssm_step(self, x, A, state):
        dbc = self.x_proj(x)
        delta, B, C = mx.split(dbc, [DT_RANK, DT_RANK + D_STATE], axis=-1)
        delta = nn.softplus(self.dt_proj(delta))
        new_state = mx.expand_dims(delta * x, -1) * mx.expand_dims(B, 1)
        if state is not None:
            new_state = new_state + state * mx.exp(mx.expand_dims(delta, -1) * A)
        y = (new_state @ mx.expand_dims(C, -1)).squeeze(2)
        return y + self.D * x, new_state

    def __call__(self, x):
        # x: [B, T, D_MODEL]
        xz = self.in_proj(x)
        xin, z = mx.split(xz, 2, axis=-1)
        xpad = mx.pad(xin, [(0, 0), (CONV_K - 1, 0), (0, 0)])
        xin = nn.silu(self.conv1d(xpad))
        A = -mx.exp(self.A_log)
        state = None
        ys = []
        for t in range(xin.shape[1]):
            y_t, state = self._ssm_step(xin[:, t], A, state)
            ys.append(y_t)
        y = mx.stack(ys, axis=1)
        return self.out_proj(nn.silu(z) * y)


class MambaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(D_MODEL)
        self.mixer = MambaMixer()

    def __call__(self, x):
        # mamba_ssm Block: residual=x; h=norm(x); h=mixer(h); return (h, x)
        return self.mixer(self.norm(x)), x


class VideoMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssms = [MambaBlock()]
        self.norm_fn = nn.LayerNorm(D_MODEL)

    def __call__(self, x):
        h, residual = self.ssms[0](x)
        return self.norm_fn(h + residual)


class PreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(D_MODEL, D_MODEL)

    def __call__(self, x):
        return _leaky_relu(self.fc3(x))


class PostNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(D_MODEL, D_MODEL)

    def __call__(self, x):
        return self.fc3(_leaky_relu(x))


def _cls_config():
    return Qwen3Config(
        model_type="qwen3", hidden_size=D_MODEL, num_hidden_layers=4,
        intermediate_size=12288, num_attention_heads=32, num_key_value_heads=8,
        rms_norm_eps=1e-6, vocab_size=2, max_position_embeddings=8192,
        rope_theta=1000000.0, head_dim=128, tie_word_embeddings=False,
    )


class ClsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Qwen3Model(_cls_config())
        self.lm_head = nn.Linear(D_MODEL, 2, bias=False)

    def __call__(self, embeds):
        # embeds: [N, 1, D] -> logits [N, 1, 2]
        return self.lm_head(self.model(None, input_embeddings=embeds))


class ClsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_model = ClsModel()


class StreamMindGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_net = PreNet()
        self.mamba_model = VideoMamba()
        self.post_net = PostNet()
        self.cls_net = ClsNet()

    def perception_tokens(self, vision_tokens):
        # vision_tokens: [B, T, P, D] -> [B, T, D]
        x = vision_tokens.mean(axis=2)
        B, T, D = x.shape
        x = self.pre_net(x.reshape(B * T, D)).reshape(B, T, D)
        x = self.mamba_model(x)
        x = self.post_net(x.reshape(B * T, D)).reshape(B, T, D)
        return x

    def __call__(self, vision_tokens):
        """Return [B, T, 2] silent/speak logits per time step."""
        tokens = self.perception_tokens(vision_tokens)
        B, T, D = tokens.shape
        logits = self.cls_net.cls_model(tokens.reshape(B * T, 1, D))  # [B*T,1,2]
        return logits.reshape(B, T, 2)

    def speak_probs(self, vision_tokens):
        """Softmax P(speak) per time step -> [B, T]."""
        return mx.softmax(self(vision_tokens), axis=-1)[..., 1]

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            # mamba conv1d [C,1,K] -> mlx Conv1d [C,K,1]
            if k.endswith("mixer.conv1d.weight") and v.shape[-1] != 1:
                v = v.moveaxis(2, 1)
            out[k] = v
        return out
