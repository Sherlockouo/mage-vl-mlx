"""Convert the HF microsoft/Mage-VL checkpoint to MLX format (optionally quantized).

Self-contained: reads the safetensors shards directly (mmap), applies our
mage_vl sanitize (HF->MLX remap + conv->linear reshape), quantizes with MLX, and
writes a clean MLX model dir (weights + config with quantization + tokenizer/
processor files). No transformers, no trust_remote_code prompt, no processor
round-trip — so it avoids the pitfalls of the generic mlx_vlm.convert path.

MEMORY: safetensors are mmap'd, so resident memory stays well under the 9.5GB
total; a 4-bit convert peaks ~5GB. Fine on 16GB.

Usage:
    python scripts/convert.py --hf-path reference --out mage-vl-mlx-4bit --bits 4
    python scripts/convert.py --hf-path reference --out mage-vl-mlx-8bit --bits 8
    python scripts/convert.py --hf-path reference --out mage-vl-mlx-bf16 --no-quantize
"""
import argparse
import json
import os
import shutil

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_vlm.models.mage_vl import Model, ModelConfig

AUX_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
    "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-path", default="reference")
    ap.add_argument("--out", default="mage-vl-mlx-4bit")
    ap.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 6, 8])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()

    import sys
    log = lambda m: (print(m, flush=True), sys.stdout.flush())
    cfg = json.load(open(f"{args.hf_path}/config.json"))
    cfg.setdefault("text_config", {})
    cfg.setdefault("vision_config", {})
    model = Model(ModelConfig.from_dict(cfg))
    log("[1/4] model built")

    # Load weights from all shards (mmap), then remap HF -> MLX.
    idx = json.load(open(f"{args.hf_path}/model.safetensors.index.json"))["weight_map"]
    weights = {}
    for shard in sorted(set(idx.values())):
        weights.update(mx.load(f"{args.hf_path}/{shard}"))
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))
    log(f"[2/4] weights loaded ({len(weights)} tensors)")

    if not args.no_quantize:
        gs = args.group_size
        # Quantize Linear/Embedding whose input dim divides the group size.
        nn.quantize(
            model, group_size=gs, bits=args.bits,
            class_predicate=lambda _p, m: (
                isinstance(m, (nn.Linear, nn.Embedding))
                and m.weight.shape[-1] % gs == 0
            ),
        )
        cfg["quantization"] = {"group_size": gs, "bits": args.bits}
        cfg["quantization_config"] = {"group_size": gs, "bits": args.bits}
    log("[3/4] quantized" if not args.no_quantize else "[3/4] (no quantize)")

    os.makedirs(args.out, exist_ok=True)
    flat = dict(tree_flatten(model.parameters()))
    # Materialize in small batches; a single eval over all 5B params overruns the
    # Metal command buffer ("excessive GPU errors").
    items = list(flat.items())
    for i in range(0, len(items), 8):
        mx.eval([v for _, v in items[i:i + 8]])
    log(f"[4/4] materialized {len(items)} tensors; saving")
    mx.save_safetensors(f"{args.out}/model.safetensors", flat, metadata={"format": "mlx"})
    json.dump(cfg, open(f"{args.out}/config.json", "w"), indent=2)
    for f in AUX_FILES:
        src = f"{args.hf_path}/{f}"
        if os.path.exists(src):
            shutil.copy(src, f"{args.out}/{f}")

    total = sum(v.size * v.dtype.size for v in flat.values()) / 1e9
    print(f"converted -> {args.out}  ({total:.2f} GB, "
          f"{'bf16' if args.no_quantize else str(args.bits)+'-bit'})")


if __name__ == "__main__":
    main()
