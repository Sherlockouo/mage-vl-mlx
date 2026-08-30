"""Native Mage-VL processor (image path) — transformers-free.

Uses `tokenizers` (tokenizer.json), `jinja2` (chat_template.jinja) and the numpy
image preprocessor. Produces model-ready inputs:
    input_ids, attention_mask, pixel_values, image_grid_thw, patch_positions

Video/codec path is not yet ported; use images=... only.
"""
import json
import os

import numpy as np

from .image_processing import preprocess_images

IMAGE_PAD = "<|image_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
VIDEO_PAD = "<|video_pad|>"


class MageVLProcessor:
    def __init__(self, model_path):
        from jinja2 import Template
        from tokenizers import Tokenizer

        self.model_path = model_path
        self.tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
        with open(os.path.join(model_path, "chat_template.jinja")) as f:
            self._template = Template(f.read())
        cfg = json.load(open(os.path.join(model_path, "preprocessor_config.json")))
        self.patch_size = int(cfg["patch_size"])
        self.merge_size = int(cfg["merge_size"])
        self.min_pixels = int(cfg.get("min_pixels", cfg["size"]["shortest_edge"]))
        self.max_pixels = int(cfg.get("max_pixels", cfg["size"]["longest_edge"]))
        self.image_token = IMAGE_PAD

    def apply_chat_template(self, messages, add_generation_prompt=True, **kwargs):
        return self._template.render(
            messages=messages, add_generation_prompt=add_generation_prompt, **kwargs
        )

    def _expand_image_pads(self, text, token_counts):
        idx = 0
        while IMAGE_PAD in text and idx < len(token_counts):
            n = int(token_counts[idx])
            text = text.replace(IMAGE_PAD, "\0" * n, 1)  # placeholder that can't collide
            idx += 1
        return text.replace("\0", IMAGE_PAD)

    def _rewrite_video_pad_frames(self, text, n_per_frame, frame_seconds):
        """Replace one <|vision_start|><|video_pad|><|vision_end|> span with
        per-frame <X.X seconds><|vision_start|><|image_pad|>*n<|vision_end|> blocks."""
        block = "".join(
            f"<{s:.1f} seconds>{VISION_START}{IMAGE_PAD * n_per_frame}{VISION_END}"
            for s in frame_seconds
        )
        span = VISION_START + VIDEO_PAD + VISION_END
        return text.replace(span, block, 1)

    def __call__(self, text, images=None, videos=None, add_generation_prompt=True,
                 video_backend="frames", codec_dir=None, fixed_num_frames=None,
                 max_frames=32, target_fps=None):
        """text: a chat string (already templated) or list of messages.
        images: PIL/np image or list thereof.
        videos: video file path (frames backend) — one video.
        video_backend: "frames" (default, portable) or "codec" (needs codec_dir).
        codec_dir: pre-generated codec asset dir (canvas_*.jpg + src_patch_position.npy).
        """
        if not isinstance(text, str):
            text = self.apply_chat_template(text, add_generation_prompt=add_generation_prompt)

        out = {}
        if videos is not None:
            merge_factor = self.merge_size * self.merge_size
            if str(video_backend).lower() == "codec":
                from .codec_processing import (
                    process_codec_asset_dir, rewrite_text_with_codec_positions,
                )
                if codec_dir is None:
                    raise ValueError("video_backend='codec' requires codec_dir=<asset dir>")
                vd = process_codec_asset_dir(
                    codec_dir, max_pixels=self.max_pixels,
                    patch_size=self.patch_size, merge_size=self.merge_size,
                )
                text = rewrite_text_with_codec_positions(
                    text, vd["patch_positions"], fps=vd["fps"], decimals=1,
                )
                out["pixel_values"] = vd["pixel_values"]
                out["image_grid_thw"] = vd["image_grid_thw"]
                out["patch_positions"] = vd["patch_positions"]
            else:
                from .video_processing import decode_video_frames, preprocess_video_frames
                path = videos[0] if isinstance(videos, (list, tuple)) else videos
                frames, indices, fps = decode_video_frames(
                    path, max_frames=max_frames,
                    fixed_num_frames=fixed_num_frames, target_fps=target_fps,
                )
                vd = preprocess_video_frames(
                    frames, indices, fps, patch_size=self.patch_size,
                    merge_size=self.merge_size, min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                grid = vd["image_grid_thw"]
                n_per_frame = int(grid[0, 1] * grid[0, 2]) // merge_factor
                text = self._rewrite_video_pad_frames(text, n_per_frame, vd["frame_seconds"])
                out["pixel_values"] = vd["pixel_values"]
                out["image_grid_thw"] = grid
                out["patch_positions"] = vd["patch_positions"]

        if images is not None:
            img_data = preprocess_images(
                images,
                patch_size=self.patch_size,
                merge_size=self.merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            grid = img_data["image_grid_thw"]
            merge_factor = self.merge_size * self.merge_size
            token_counts = (grid[:, 0] * grid[:, 1] * grid[:, 2]) // merge_factor
            text = self._expand_image_pads(text, token_counts)
            out["pixel_values"] = img_data["pixel_values"]
            out["image_grid_thw"] = grid
            out["patch_positions"] = img_data["patch_positions"]

        enc = self.tokenizer.encode(text, add_special_tokens=False)
        ids = np.array([enc.ids], dtype=np.int64)
        out["input_ids"] = ids
        out["attention_mask"] = np.ones_like(ids)
        return out

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
