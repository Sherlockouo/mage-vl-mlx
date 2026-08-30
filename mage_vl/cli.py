"""Command-line recognition: images / videos / plain chat.

    python -m mage_vl.cli --model ./mage-vl-mlx --image photo.jpg --prompt "描述画面"
    python -m mage_vl.cli --model ./mage-vl-mlx --video clip.mp4 --frames 8
    python -m mage_vl.cli --model ./mage-vl-mlx --text "你好"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ._link import ensure_plugin_installed


def main() -> int:
    ap = argparse.ArgumentParser(description="Mage-VL MLX command-line recognition")
    ap.add_argument("--model", default=os.environ.get(
        "MAGE_MODEL_DIR", str(Path.cwd() / "mage-vl-mlx")), help="converted MLX model dir")
    ap.add_argument("--image", help="image file to describe")
    ap.add_argument("--video", help="video file to describe (samples frames)")
    ap.add_argument("--frames", type=int, default=8, help="frames sampled from video")
    ap.add_argument("--max-pixels", type=int, default=1_000_000)
    ap.add_argument("--text", help="text-only chat")
    ap.add_argument("--prompt", default="用一到两句话描述当前画面。")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    if not Path(args.model).exists():
        print(f"model dir not found: {args.model}\n"
              "convert weights first:  python scripts/convert.py --help", file=sys.stderr)
        return 1

    ensure_plugin_installed()

    from .engine import MageVLEngine  # noqa: E402  (after plugin link)

    engine = MageVLEngine(args.model)
    if args.text:
        res = engine.generate_text(args.text, max_tokens=args.max_tokens)
    elif args.video:
        prep = engine.preprocess_video(args.video, num_frames=args.frames,
                                       max_pixels=args.max_pixels)
        res = engine.summarize(prep, args.prompt, max_tokens=args.max_tokens)
    elif args.image:
        res = engine.generate_image(args.image, args.prompt, max_tokens=args.max_tokens,
                                    max_pixels=args.max_pixels)
    else:
        ap.error("one of --image / --video / --text is required")
        return 1

    print(res["text"])
    print(f"\n[prefill {res['prefill_s']}s | decode {res['decode_s']}s | "
          f"{res['tokens']} tok @ {res['tokens_per_s']} tok/s]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
