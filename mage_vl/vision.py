import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


class VisionRotaryEmbedding:
    """3D (T,H,W) rotary frequency constructor with 4:6:6 split.

    Mirrors ``VisionRotaryEmbedding`` in modeling_mage_vl.py.
    head_dim = hidden_size // num_heads; half = head_dim // 2 is split 4:6:6
    across the temporal / height / width axes (in units of half//16).
    """

    def __init__(self, config: VisionConfig):
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta
        half = head_dim // 2
        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit
        self.inv_freq_t = 1.0 / (base ** (mx.arange(self.t_size, dtype=mx.float32) / self.t_size))
        self.inv_freq_h = 1.0 / (base ** (mx.arange(self.h_size, dtype=mx.float32) / self.h_size))
        self.inv_freq_w = 1.0 / (base ** (mx.arange(self.w_size, dtype=mx.float32) / self.w_size))

    def from_positions(self, patch_positions: mx.array) -> mx.array:
        # patch_positions: [seq, 3] with (t, h, w) indices
        t_pos = patch_positions[:, 0].astype(mx.float32)
        h_pos = patch_positions[:, 1].astype(mx.float32)
        w_pos = patch_positions[:, 2].astype(mx.float32)
        ft = mx.outer(t_pos, self.inv_freq_t)
        fh = mx.outer(h_pos, self.inv_freq_h)
        fw = mx.outer(w_pos, self.inv_freq_w)
        return mx.concatenate([ft, fh, fw], axis=-1)  # [seq, half]


def rotate_half(x: mx.array) -> mx.array:
    """Interleaved rotation: (x1, x2, x3, x4) -> (-x2, x1, -x4, x3)."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return mx.stack([-x_odd, x_even], axis=-1).reshape(x.shape)


def apply_rotary_pos_emb(q, k, freqs):
    # q, k: (B, H, L, D); freqs: (B, L, D)
    orig_dtype = q.dtype
    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    cos = mx.expand_dims(mx.cos(freqs), 1).astype(mx.float32)
    sin = mx.expand_dims(mx.sin(freqs), 1).astype(mx.float32)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


class Attention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=True)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(self, x, rotary_pos_emb, attention_mask=None):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rotary_pos_emb is not None:
            q, k = apply_rotary_pos_emb(q, k, rotary_pos_emb)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=attention_mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.embed_dim)
        return self.proj(out)


class MLP(nn.Module):
    """SigLIP-style MLP: fc1 -> gelu(tanh) -> fc2."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        # config.hidden_act == "gelu" -> transformers ACT2FN["gelu"] is exact (erf) GELU.
        # NB: in MLX only approx="none" is exact erf; "precise"/"tanh" are the tanh approx.
        self.act = nn.GELU(approx="none")

    def __call__(self, x):
        return self.fc2(self.act(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def __call__(self, x, rotary_pos_emb, attention_mask=None):
        x = x + self.self_attn(self.layer_norm1(x), rotary_pos_emb, attention_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class PatchMerger(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # nn.Sequential in HF -> mlp.0 (Linear), mlp.1 (GELU), mlp.2 (Linear)
        self.mlp = [
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(approx="none"),  # HF merger uses exact nn.GELU()
            nn.Linear(self.hidden_size, config.out_hidden_size),
        ]

    def __call__(self, x):
        x = self.ln_q(x).reshape(-1, self.hidden_size)
        for layer in self.mlp:
            x = layer(x)
        return x


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.spatial_merge_size = config.spatial_merge_size
        self.frame_windows_size = config.frame_windows_size
        # Patch embed as a linear over flattened (C, ph, pw) patches (conv with
        # kernel==stride==patch collapses to this). Weight is reshaped in sanitize.
        patch_dim = config.num_channels * config.patch_size * config.patch_size
        self.patch_embedding = nn.Linear(patch_dim, config.hidden_size, bias=False)
        self.layernorm_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.merger = PatchMerger(config)
        self.rope = VisionRotaryEmbedding(config)

    def _window_mask(self, grid_thw):
        """Block-diagonal additive mask matching _build_cu_seqlens with fixed_t.

        Frames are grouped into windows of ``frame_windows_size`` along t; each
        window attends only within itself. Returns None if a single full block.
        """
        fixed_t = self.frame_windows_size
        lengths = []
        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            if fixed_t and fixed_t > 0 and t > fixed_t:
                for _ in range(t // fixed_t):
                    lengths.append(fixed_t * h * w)
                if t % fixed_t:
                    lengths.append((t % fixed_t) * h * w)
            else:
                lengths.append(t * h * w)
        if len(lengths) == 1:
            return None
        # Assign each patch a segment id; patches attend only within the same segment.
        total = sum(lengths)
        seg = mx.zeros((total,), dtype=mx.int32)
        idx = mx.arange(total)
        start = 0
        for i, size in enumerate(lengths):
            seg = mx.where((idx >= start) & (idx < start + size), mx.array(i, mx.int32), seg)
            start += size
        allow = seg[:, None] == seg[None, :]
        return mx.where(allow, mx.array(0.0, mx.float32), mx.array(-mx.inf, mx.float32))

    def __call__(self, pixel_values, grid_thw, patch_positions,
                 output_hidden_states=False, skip_merger=False):
        freqs = self.rope.from_positions(patch_positions)  # [seq, half]
        freqs = mx.concatenate([freqs, freqs], axis=-1)  # [seq, head_dim]
        freqs = mx.expand_dims(freqs, 0)  # [1, seq, head_dim]

        mask = self._window_mask(grid_thw)
        if mask is not None:
            mask = mask.astype(pixel_values.dtype)  # bf16 for SDPA

        return self.forward_with_inputs(
            pixel_values, freqs, mask,
            output_hidden_states=output_hidden_states, skip_merger=skip_merger)

    def prepare_static_inputs(self, grid_thw, patch_positions):
        """Precompute rope freqs + window mask for a fixed (grid, positions).

        The returned arrays are reusable constants — e.g. for live video where
        every frame shares the same canvas/grid — and let callers run the
        tower under ``mx.compile`` without re-running the python-side mask
        builder on each call.
        """
        freqs = self.rope.from_positions(patch_positions)
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        freqs = mx.expand_dims(freqs, 0)
        mask = self._window_mask(grid_thw)
        return freqs, mask

    def forward_with_inputs(self, pixel_values, freqs, mask,
                            output_hidden_states=False, skip_merger=False):
        """Layer-stack forward given precomputed rope freqs / window mask."""
        # pixel_values: [total_patches, C*patch*patch]
        h = self.patch_embedding(pixel_values)
        h = mx.expand_dims(h, 0)  # [1, total_patches, hidden]

        h = self.layernorm_pre(h)
        if mask is not None:
            if mask.dtype != h.dtype:
                mask = mask.astype(h.dtype)

        hidden_states = []
        for layer in self.layers:
            h = layer(h, freqs, mask)
            if output_hidden_states:
                hidden_states.append(h)

        if skip_merger:
            return (h[0], tuple(hidden_states)) if output_hidden_states else h[0]
        out = self.merger(h)
        return (out, tuple(hidden_states)) if output_hidden_states else out

    def sanitize(self, weights):
        out = {}
        for k, v in weights.items():
            if k.endswith("embeddings.patch_embedding.weight"):
                # HF conv weight [out, in, ph, pw] -> linear [out, in*ph*pw]
                out[k.replace("embeddings.patch_embedding", "patch_embedding")] = v.reshape(
                    v.shape[0], -1
                )
            elif ".encoder.layers." in k:
                out[k.replace(".encoder.layers.", ".layers.")] = v
            else:
                out[k] = v
        return out
