"""Native (numpy + PyAV) frame-sampling video preprocessing for Mage-VL.

Ports the DEFAULT ``video_backend="frames"`` path of video_processing_mage_vl.py:
decode -> sample frames -> Qwen2VL image patchify -> per-frame timestamp tags.
Mage-ViT is purely spatial, so each sampled frame is treated as an image and the
model's image path consumes the result unchanged.

Produces:
    pixel_values    : [T*H_p*W_p, C*patch*patch]
    image_grid_thw  : [T, 3]  rows [1, H_p, W_p]   (one window per frame)
    patch_positions : [T*H_p*W_p, 3]  block layout, t-axis = REAL frame indices
    frame_seconds   : list[float]  timestamp (s) of each sampled frame

NOTE on fidelity: the reference pre-resizes frames with torchvision BICUBIC+
antialias; here resize happens once inside the numpy image processor (PIL BICUBIC).
Token counts / positions / timestamps match exactly; pixel values may differ
slightly from the torchvision path (documented, torch-free by design).
"""
import numpy as np

from .image_processing import _patchify, build_patch_positions, smart_resize, IMAGE_MEAN, IMAGE_STD


def choose_target_frames(duration_seconds, max_frames, fixed_num_frames=None, target_fps=None):
    if target_fps is not None and target_fps > 0:
        return min(max(1, int(duration_seconds * target_fps)), max_frames)
    if fixed_num_frames is not None:
        return int(fixed_num_frames)
    if duration_seconds < 10:
        return 8
    if duration_seconds < 30:
        return 16
    return max_frames


def select_frame_indices(frame_count, target_count):
    if frame_count <= target_count:
        return list(range(frame_count))
    return [int(round(x)) for x in np.linspace(0, frame_count - 1, target_count)]


def decode_video_frames(path, max_frames=32, fixed_num_frames=None, target_fps=None):
    """Decode + sample frames with PyAV. Returns (frames_pil, indices, fps)."""
    import av
    from PIL import Image

    container = av.open(path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    total = stream.frames or 0
    if total <= 0:
        # Some containers don't report frame count; estimate from duration.
        if stream.duration and stream.time_base:
            total = int(float(stream.duration * stream.time_base) * fps)
        else:
            total = 0
    if total <= 0:
        # Fall back to a full decode count (short clips only).
        total = sum(1 for _ in container.decode(stream))
        container.close()
        container = av.open(path)
        stream = container.streams.video[0]

    duration = total / fps if fps else 0.0
    target = choose_target_frames(duration, max_frames, fixed_num_frames, target_fps)
    indices = select_frame_indices(total, target)
    wanted = set(indices)
    last = max(indices) if indices else 0

    grabbed = {}
    for i, frame in enumerate(container.decode(stream)):
        if i in wanted:
            grabbed[i] = frame.to_image().convert("RGB")
        if i >= last:
            break
    container.close()

    frames_pil = [grabbed[i] for i in indices if i in grabbed]
    kept_indices = [i for i in indices if i in grabbed]
    return frames_pil, kept_indices, fps


def preprocess_video_frames(
    frames_pil,
    frame_indices,
    fps,
    patch_size=16,
    merge_size=2,
    min_pixels=3136,
    max_pixels=4000000,
):
    """Patchify sampled frames into the model's image-path tensors."""
    from PIL import Image

    if not frames_pil:
        raise ValueError("no frames to preprocess")

    factor = patch_size * merge_size
    # smart_resize the first frame to fix a single (H_p, W_p) for the whole video.
    w0, h0 = frames_pil[0].size
    rh, rw = smart_resize(h0, w0, factor, min_pixels, max_pixels)
    gh, gw = rh // patch_size, rw // patch_size

    all_pixels = []
    for img in frames_pil:
        img = img.convert("RGB")
        if img.size != (rw, rh):
            img = img.resize((rw, rh), resample=Image.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = (arr - IMAGE_MEAN) / IMAGE_STD
        arr = arr.transpose(2, 0, 1)
        all_pixels.append(_patchify(arr, patch_size, merge_size))

    T = len(frames_pil)
    pixel_values = np.concatenate(all_pixels, axis=0).astype(np.float32)
    # image_grid_thw: one [1, H_p, W_p] row per frame (each frame = one attn window)
    image_grid_thw = np.array([[1, gh, gw]] * T, dtype=np.int64)
    # patch_positions over the merged [T, H_p, W_p] grid, real frame idx on t-axis
    video_grid = np.array([[T, gh, gw]], dtype=np.int64)
    patch_positions = build_patch_positions(
        video_grid, spatial_merge_size=merge_size,
        frame_indices=[np.asarray(frame_indices, dtype=np.int64)],
    )
    frame_seconds = [float(i) / float(fps) if fps else 0.0 for i in frame_indices]
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "patch_positions": patch_positions,
        "frame_seconds": frame_seconds,
    }
