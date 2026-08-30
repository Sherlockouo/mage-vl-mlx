"""Memory-SAFE structural test: a miniature Mage-VL (tiny dims, 2 layers) that
exercises every ported code path without large allocations. Validates shapes and
that config parsing / vision forward / embed-merge / language forward all run.

Does NOT load the real 5B model (that must be 4-bit quantized on a 16GB machine).
"""
import mlx.core as mx
from mlx_vlm.models.mage_vl import Model, ModelConfig

# Tiny config. head_dim must satisfy the vision RoPE 4:6:6 split (half % 16 == 0),
# so vision head_dim = 32 (half = 16 -> unit 1 -> t/h/w = 4/6/6). out_hidden_size
# must equal text hidden_size so merged features drop into the token embeddings.
HID = 48
cfg = {
    "model_type": "mage_vl",
    "image_token_id": 90,
    "video_token_id": 91,
    "vision_config": {
        "hidden_size": 64,
        "num_attention_heads": 2,   # head_dim = 32
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "patch_size": 16,
        "num_channels": 3,
        "out_hidden_size": HID,
        "spatial_merge_size": 2,
        "frame_windows_size": 4,
        "rope_theta": 10000.0,
    },
    "text_config": {
        "model_type": "qwen3",
        "hidden_size": HID,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 2,
        "intermediate_size": 96,
        "head_dim": 24,
        "vocab_size": 200,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 5000.0, "rope_type": "default"},
        "tie_word_embeddings": False,
    },
}
model_config = ModelConfig.from_dict(cfg)
assert model_config.text_config.rope_theta == 5000.0, "rope_parameters not parsed"
model = Model(model_config)
mx.eval(model.parameters())

vc = model_config.vision_config
patch_dim = vc.num_channels * vc.patch_size * vc.patch_size


def patch_positions_2x2(t, h, w):
    pos = []
    for tt in range(t):
        for hb in range(0, h, 2):
            for wb in range(0, w, 2):
                for dh in range(2):
                    for dw in range(2):
                        pos.append([tt, hb + dh, wb + dw])
    return mx.array(pos)


# --- Image path: single 4x4 patch grid (t=1) ---
t, h, w = 1, 4, 4
n = t * h * w
pv = mx.random.normal((n, patch_dim))
pp = patch_positions_2x2(t, h, w)
grid = mx.array([[t, h, w]])
feats = model.vision_tower(pv, grid, pp)
mx.eval(feats)
assert feats.shape == (n // 4, HID), feats.shape
print(f"[image] vision merged {feats.shape} OK")

# Full forward with image tokens
n_tok = n // 4
ids = [1, 2] + [cfg["image_token_id"]] * n_tok + [3, 4]
input_ids = mx.array([ids])
out = model(input_ids, pixel_values=pv, image_grid_thw=grid, patch_positions=pp)
mx.eval(out.logits)
assert out.logits.shape == (1, len(ids), 200), out.logits.shape
print(f"[image] logits {out.logits.shape} OK")

# --- Video path: t=6 > frame_windows_size(4) exercises the block window mask ---
tv, hv, wv = 6, 4, 4
nv = tv * hv * wv
pvv = mx.random.normal((nv, patch_dim))
ppv = patch_positions_2x2(tv, hv, wv)
gridv = mx.array([[tv, hv, wv]])
fv = model.vision_tower(pvv, gridv, ppv)
mx.eval(fv)
assert fv.shape == (nv // 4, HID), fv.shape
print(f"[video] vision merged {fv.shape} (window mask path) OK")

# --- sanitize key remapping (no real weights needed) ---
fake = {
    "model.visual.embeddings.patch_embedding.weight": mx.zeros((64, 3, 16, 16)),
    "model.visual.encoder.layers.0.layer_norm1.weight": mx.zeros((64,)),
    "model.visual.merger.mlp.0.weight": mx.zeros((256, 256)),
    "model.language_model.layers.0.self_attn.q_proj.weight": mx.zeros((HID, HID)),
    "model.language_model.embed_tokens.weight": mx.zeros((200, HID)),
    "lm_head.weight": mx.zeros((200, HID)),
}
s = model.sanitize(fake)
expected = {
    "vision_tower.patch_embedding.weight": (64, 768),
    "vision_tower.layers.0.layer_norm1.weight": (64,),
    "vision_tower.merger.mlp.0.weight": (256, 256),
    "language_model.model.layers.0.self_attn.q_proj.weight": (HID, HID),
    "language_model.model.embed_tokens.weight": (200, HID),
    "language_model.lm_head.weight": (200, HID),
}
for k, shp in expected.items():
    assert k in s, f"missing remapped key {k}"
    assert s[k].shape == shp, f"{k}: {s[k].shape} != {shp}"
print("[sanitize] key remapping + conv->linear reshape OK")

print("\nALL STRUCTURAL CHECKS PASSED")
