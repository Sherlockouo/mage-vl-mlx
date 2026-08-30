"""Native (numpy + PIL) image preprocessing for Mage-VL — a port of the slow
``Qwen2VLImageProcessor`` path used by ``processing_mage_vl.py``, plus the
``build_patch_positions`` block-layout position builder from
``video_processing_mage_vl.py``.

Produces exactly the three tensors the MLX model consumes:
  - pixel_values    : [total_patches, C*patch*patch]  (feature order C,ph,pw)
  - image_grid_thw  : [num_images, 3]                  (t=1, h_p, w_p)
  - patch_positions : [total_patches, 3]               (t,h,w) in 2x2 block order

pixel_values row i corresponds to patch_positions row i (both in 2x2 block order).

Resize uses PIL BICUBIC on the uint8 image to match the SLOW Qwen2VLImageProcessor
(the Fast variant has small normalization rounding differences).
"""
import math

import numpy as np
from PIL import Image

# CLIP normalization constants from preprocessor_config.json
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
MAX_RATIO = 200


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Qwen2-VL smart_resize: round H/W to multiples of ``factor`` within the
    [min_pixels, max_pixels] budget while preserving aspect ratio."""
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be < {MAX_RATIO}, got {max(height, width) / min(height, width):.2f}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def _block_layout_indices(t, h, w, sms):
    """Row-major (t,h,w) -> 2x2 block order indices. Mirrors
    _convert_positions_to_block_layout in video_processing_mage_vl.py."""
    if sms == 1:
        return np.arange(t * h * w)
    idx = np.arange(t * h * w).reshape(t, h, w)
    h_m, w_m = h // sms, w // sms
    idx = idx.reshape(t, h_m, sms, w_m, sms).transpose(0, 1, 3, 2, 4).reshape(-1)
    return idx


def build_patch_positions(grid_thw, spatial_merge_size=2, frame_indices=None):
    """[num_samples,3] grid -> [sum(t*h*w),3] (t,h,w) positions in block layout.

    frame_indices: optional list (one entry per grid row) of real source-frame
    indices to use as the t-coordinate (training convention for 3D RoPE). Pass
    None for dense arange(t)."""
    out = []
    for sample_idx, row in enumerate(grid_thw):
        t_v, h_v, w_v = int(row[0]), int(row[1]), int(row[2])
        h_coords = np.tile(np.repeat(np.arange(h_v), w_v), t_v)
        w_coords = np.tile(np.tile(np.arange(w_v), h_v), t_v)
        fi = None
        if frame_indices is not None and sample_idx < len(frame_indices):
            fi = frame_indices[sample_idx]
        if fi is not None:
            fi = np.asarray(fi, dtype=np.int64)
            if fi.size != t_v:
                raise ValueError(f"frame_indices[{sample_idx}] len {fi.size} != t {t_v}")
            t_coords = np.repeat(fi, h_v * w_v)
        else:
            t_coords = np.repeat(np.arange(t_v), h_v * w_v)
        pp = np.stack([t_coords, h_coords, w_coords], axis=1)
        pp = pp[_block_layout_indices(t_v, h_v, w_v, spatial_merge_size)]
        out.append(pp)
    return np.concatenate(out, axis=0).astype(np.int64)


def _patchify(arr, patch, sms):
    """arr: [C,H,W] float -> [gh*gw, C*patch*patch] in 2x2 block row order,
    feature order (C, ph, pw)."""
    c, H, W = arr.shape
    gh, gw = H // patch, W // patch
    a = arr.reshape(c, gh // sms, sms, patch, gw // sms, sms, patch)
    #      axes: 0=C 1=ghb 2=hin 3=ph 4=gwb 5=win 6=pw
    a = a.transpose(1, 4, 2, 5, 0, 3, 6)  # (ghb, gwb, hin, win, C, ph, pw)
    return a.reshape(gh * gw, c * patch * patch)


def preprocess_images(
    images,
    patch_size=16,
    merge_size=2,
    min_pixels=3136,
    max_pixels=4000000,
):
    """Preprocess a list of PIL/np images.

    Returns dict with numpy arrays: pixel_values [N,768], image_grid_thw [n,3],
    patch_positions [N,3].
    """
    if not isinstance(images, (list, tuple)):
        images = [images]

    factor = patch_size * merge_size
    all_pixels, grids = [], []
    for img in images:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB")
        w0, h0 = img.size  # PIL: (width, height)
        rh, rw = smart_resize(h0, w0, factor, min_pixels, max_pixels)
        if (rw, rh) != (w0, h0):
            img = img.resize((rw, rh), resample=Image.BICUBIC)

        arr = np.asarray(img).astype(np.float32)  # [H,W,C]
        arr = arr / 255.0
        arr = (arr - IMAGE_MEAN) / IMAGE_STD
        arr = arr.transpose(2, 0, 1)  # [C,H,W]

        gh, gw = rh // patch_size, rw // patch_size
        all_pixels.append(_patchify(arr, patch_size, merge_size))
        grids.append([1, gh, gw])

    pixel_values = np.concatenate(all_pixels, axis=0).astype(np.float32)
    image_grid_thw = np.array(grids, dtype=np.int64)
    patch_positions = build_patch_positions(image_grid_thw, merge_size)
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "patch_positions": patch_positions,
    }
