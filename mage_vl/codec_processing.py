"""Codec video glue for Mage-VL — the portable (numpy/pure-python) parts of
codec_video_processing_mage_vl.py.

IMPORTANT — scope: the actual codec *canvas generation* (HEVC bit-cost readiness
via the external ``cv-preinfer`` binary, or the DCVC-RT neural codec that shells
out to the bundled ``neural_codec/`` package with checkpoints + CUDA) is NOT
reimplementable in numpy/MLX — it is a separate neural video codec. This module
ports everything DOWNSTREAM of the engine:

  - load a codec asset dir (canvas_*.jpg + src_patch_position.npy + meta.json)
  - drop fully-padding canvases
  - reorder codec positions into 2x2 block layout, aligned to image_grid_thw
  - build pixel_values from canvases (via the native image processor, with the
    codec min/max_pixels clamp so smart_resize never desyncs the grid)
  - rewrite the chat-template vision span into per-timestamp token runs

So: run the external engine (or a pre-generated asset dir) to get canvases, then
this module + the MLX model consume them. On Apple Silicon the practical video
path is ``video_backend="frames"`` (see video_processing.py); codec is supported
here for when a codec asset dir is available.
"""
import json
import os

import numpy as np

from .image_processing import _block_layout_indices, preprocess_images

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"


def load_codec_asset_dir(out_dir):
    """Load a pre-generated codec asset dir -> dict(images, src_positions, fps)."""
    from PIL import Image

    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    canvas_files = meta.get("canvas_files")
    if not canvas_files:
        for ext in ("npy", "jpg", "png"):
            hits = sorted(p for p in os.listdir(out_dir)
                          if p.startswith("canvas_") and p.endswith("." + ext))
            if hits:
                canvas_files = hits
                break
        canvas_files = canvas_files or []
    images = []
    for name in canvas_files:
        fp = os.path.join(out_dir, name)
        if name.endswith(".npy"):
            images.append(Image.fromarray(np.load(fp)))
        else:
            images.append(Image.open(fp).convert("RGB"))
    src_positions = np.load(os.path.join(out_dir, "src_patch_position.npy"))
    fps = float(meta.get("fps") or 30.0)
    return {"images": images, "src_positions": src_positions, "fps": fps, "meta": meta}


def drop_padding_canvases(images, src_positions):
    """Drop fully-padding canvases (all-negative timestamps) and their patches."""
    n = len(images)
    if n == 0:
        return images, src_positions, 0
    total = src_positions.shape[0]
    if total % n != 0:
        raise ValueError(f"src_positions {total} not divisible by canvas count {n}")
    ppc = total // n
    positions = src_positions.reshape(n, ppc, 3)
    canvas_t = positions[..., 0]
    keep = (canvas_t >= 0).any(axis=1)
    if bool((keep & ~((canvas_t >= 0).all(axis=1))).any()):
        raise ValueError("half-padding canvas encountered; padding must be canvas-granular")
    dropped = int(n - int(keep.sum()))
    if dropped == 0:
        return images, src_positions, 0
    kept_images = [im for im, k in zip(images, keep.tolist()) if k]
    kept_positions = positions[keep].reshape(-1, 3)
    return kept_images, kept_positions, dropped


def codec_positions_for_processor(src_positions, image_grid_thw):
    """Reorder codec (t*h*w,3) row-major positions into 2x2 block layout,
    chunked to match image_grid_thw rows."""
    positions = np.asarray(src_positions, dtype=np.int64)
    expected = int((image_grid_thw.prod(axis=1)).sum())
    if expected != positions.shape[0]:
        raise ValueError(f"codec position length mismatch: grid={expected} pos={positions.shape[0]}")
    chunks, off = [], 0
    for row in image_grid_thw:
        t, h, w = int(row[0]), int(row[1]), int(row[2])
        n = t * h * w
        chunk = positions[off:off + n]
        chunks.append(chunk[_block_layout_indices(t, h, w, 2)])
        off += n
    return np.concatenate(chunks, axis=0)


def _timestamp_runs(patch_positions, fps, decimals=1, spatial_merge_size=2):
    """Consecutive runs of the t-coordinate -> [(timestamp_str, token_count)]."""
    t_values = patch_positions[:, 0]
    # unique_consecutive
    changes = np.concatenate([[True], t_values[1:] != t_values[:-1]])
    idx = np.where(changes)[0]
    counts = np.diff(np.concatenate([idx, [len(t_values)]]))
    merge_factor = spatial_merge_size ** 2
    runs = []
    for i, c in zip(idx, counts):
        t_val = int(t_values[i])
        if t_val < 0:
            continue
        tok = int(c) // merge_factor
        if tok <= 0:
            continue
        runs.append((f"<{t_val / float(fps):.{decimals}f} seconds>", tok))
    return runs


def rewrite_text_with_codec_positions(text, patch_positions, fps, decimals=1):
    """Replace the vision span in the chat string with codec-aware token runs."""
    parts = []
    for ts, tok in _timestamp_runs(patch_positions, fps, decimals):
        parts.extend([ts, VISION_START, IMAGE_PAD * tok, VISION_END, "\n"])
    vision_text = "".join(parts)
    first_vs, last_ve = text.find(VISION_START), text.rfind(VISION_END)
    if first_vs == -1 or last_ve == -1:
        return text
    tail = last_ve + len(VISION_END)
    if tail < len(text) and text[tail] == "\n":
        tail += 1
    return text[:first_vs] + vision_text + text[tail:]


def codec_canvases_to_pixels(images, max_pixels, patch_size=16, merge_size=2):
    """Patchify codec canvases. Clamp min/max_pixels to the canvas sizes so
    smart_resize is a no-op and image_grid_thw stays aligned to src positions."""
    canvas_pixels = [im.width * im.height for im in images]
    proc_max = max(int(max_pixels), max(canvas_pixels, default=int(max_pixels)))
    proc_min = min(canvas_pixels) if canvas_pixels else 1
    return preprocess_images(
        images, patch_size=patch_size, merge_size=merge_size,
        min_pixels=proc_min, max_pixels=proc_max,
    )


def process_codec_asset_dir(out_dir, max_pixels=150000, patch_size=16, merge_size=2):
    """Full downstream pipeline from a pre-generated codec asset dir.

    Returns dict(pixel_values, image_grid_thw, patch_positions, fps) plus the
    rewrite is applied by the caller via rewrite_text_with_codec_positions.
    """
    payload = load_codec_asset_dir(out_dir)
    images, src_positions, _ = drop_padding_canvases(payload["images"], payload["src_positions"])
    if not images:
        raise RuntimeError(f"codec asset dir {out_dir} produced no usable canvases")
    img_data = codec_canvases_to_pixels(images, max_pixels, patch_size, merge_size)
    grid = img_data["image_grid_thw"]
    patch_positions = codec_positions_for_processor(src_positions, grid)
    return {
        "pixel_values": img_data["pixel_values"],
        "image_grid_thw": grid,
        "patch_positions": patch_positions,
        "fps": payload["fps"],
    }
