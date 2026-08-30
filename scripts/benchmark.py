"""Performance benchmark for the MLX Mage-VL model on Apple Silicon.

Measures prefill throughput, decode throughput, and peak memory for a text-only
prompt and an image prompt. Run on a converted (quantized) model dir.

Usage:
    python scripts/benchmark.py --mlx mage-vl-mlx-4bit --image reference/examples/dog.jpg
"""
import argparse
import time
from pathlib import Path

import mlx.core as mx
from PIL import Image

from mlx_lm.models.cache import KVCache
from mlx_vlm.models.mage_vl.processing import MageVLProcessor
from mlx_vlm.utils import load_model


def run(model, input_ids, kwargs, max_tokens):
    cache = [KVCache() for _ in model.layers]
    n_prompt = input_ids.shape[1]
    t0 = time.perf_counter()
    out = model(input_ids, cache=cache, **kwargs)
    y = mx.argmax(out.logits[:, -1, :], axis=-1)
    mx.eval(y)
    prefill_t = time.perf_counter() - t0

    eos = {151645, 151643}
    n = 0
    t1 = time.perf_counter()
    for _ in range(max_tokens):
        if int(y.item()) in eos:
            break
        n += 1
        out = model(y[None], cache=cache)
        y = mx.argmax(out.logits[:, -1, :], axis=-1)
        mx.eval(y)
    decode_t = time.perf_counter() - t1
    return n_prompt, prefill_t, n, decode_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx", default="mage-vl-mlx-4bit")
    ap.add_argument("--tokenizer-src", default="reference")
    ap.add_argument("--image", default="reference/examples/dog.jpg")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = load_model(Path(args.mlx))
    mx.eval(model.parameters())
    load_t = time.perf_counter() - t0
    weight_mem = mx.get_active_memory() / 1e9
    proc = MageVLProcessor(args.tokenizer_src)

    print(f"model: {args.mlx}")
    print(f"load: {load_t:.1f}s | weights resident: {weight_mem:.2f} GB\n")

    cases = {}
    # text-only
    txt = proc.apply_chat_template(
        [{"role": "user", "content": "Explain what a transformer neural network is."}],
        add_generation_prompt=True,
    )
    ti = proc(txt)
    cases["text-only"] = (mx.array(ti["input_ids"]), {})
    # image
    it = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}],
        add_generation_prompt=True,
    )
    ii = proc(it, images=Image.open(args.image).convert("RGB"))
    cases["image"] = (
        mx.array(ii["input_ids"]),
        dict(pixel_values=mx.array(ii["pixel_values"]),
             image_grid_thw=mx.array(ii["image_grid_thw"]),
             patch_positions=mx.array(ii["patch_positions"])),
    )

    print(f"{'case':<12}{'prompt tok':>11}{'prefill tok/s':>15}{'decode tok/s':>14}{'peak GB':>10}")
    print("-" * 62)
    for name, (ids, kw) in cases.items():
        # warmup
        run(model, ids, kw, 4)
        best = None
        mx.reset_peak_memory()
        for _ in range(args.runs):
            np_, pt, nd, dt = run(model, ids, kw, args.max_tokens)
            pfs = np_ / pt
            dts = nd / dt if dt > 0 else 0
            if best is None or dts > best[3]:
                best = (np_, pt, dts, pfs)
        peak = mx.get_peak_memory() / 1e9
        np_, pt, dts, pfs = best
        print(f"{name:<12}{np_:>11}{pfs:>15.1f}{dts:>14.1f}{peak:>10.2f}")


if __name__ == "__main__":
    main()
