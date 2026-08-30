"""Mage-VL optimized inference engine (Apple Silicon / MLX).

MLX on Apple Silicon requires lazy arrays to be created and evaluated on the
same thread (and its thread-local GPU stream). So this engine runs a single
dedicated **inference worker thread** that loads the model itself and executes
every model call. CPU-side work (PyAV decode / numpy preprocess / tokenizer) is
free to run on any thread; everything touching ``mlx.core`` is submitted to the
worker and serialized.

Exposed building blocks:

  * ``preprocess_video``      – decode + sample frames (any thread; pure CPU/numpy)
  * ``submit``                – enqueue a callable onto the inference worker
  * ``vision_features``       – per-frame vision prefill (worker side)
  * ``gate_timeline``         – StreamMind event-gate scoring (worker side)
  * ``summarize``             – token-streamed work-summary generation (worker side)
  * ``describe_frame``        – single-frame live analysis (blocks until done)
  * ``describe_frame_stream`` – same, streaming text deltas + per-stage timings

Performance notes (measured on M4 Max, 4-bit weights):

  * Generation streams through :class:`StreamingDetokenizer` — incremental text
    instead of re-decoding the whole history on every token.
  * Single-frame analysis freezes RoPE/mask constants per (h, w) grid and runs
    the vision stack under ``mx.compile`` when shapes repeat (~15% faster,
    bit-identical output). Toggle with ``MAGE_COMPILE_VISION=0|1`` (default on).
  * ``summarize`` accepts the cached ``vision_tokens`` from ``vision_features``
    so long video passes do not run the vision tower twice.
"""
import io
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_lm.models.cache import KVCache
from . import video_processing as vp
from .detokenize import StreamingDetokenizer
from .processing import MageVLProcessor
from .streaming import StreamMindGate
from mlx_vlm.utils import load_model

from ._link import ensure_plugin_installed

EOS_IDS = {151645, 151643}

log = logging.getLogger("api.engine")

FrameCB = Callable[[int, float], None]          # (frame_idx, second)
ProbCB = Callable[[int, float], None]           # (frame_idx, P(speak))
SegCB = Callable[[str], None]                   # incremental decoded segment


class _Future:
    """Minimal future returned by blocking worker calls (e.g. describe_frame)."""

    def __init__(self):
        self._event = threading.Event()
        self._value = None
        self._error: Optional[BaseException] = None

    def set_result(self, value):
        self._value = value
        self._event.set()

    def set_exception(self, exc: BaseException):
        self._error = exc
        self._event.set()

    def result(self, timeout: Optional[float] = None):
        self._event.wait(timeout)
        if self._error is not None:
            raise self._error
        return self._value


class MageVLEngine:
    def __init__(self, model_dir: str, gate_path: Optional[str] = None,
                 video_max_pixels: int = 2_000_000, max_input_tokens: int = 22_000):
        self.model_dir = Path(model_dir)
        self._gate_path = gate_path
        self.video_max_pixels = video_max_pixels
        self.max_input_tokens = max_input_tokens
        self.compile_vision = os.environ.get("MAGE_COMPILE_VISION", "1") != "0"
        # grid key (gh, gw) -> {"step": compiled/plain vision forward, ...}
        self._vision_static: dict[tuple[int, int], dict] = {}
        self._queue: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._ready = threading.Event()
        self._load_error: Optional[BaseException] = None

        threading.Thread(target=self._worker_main, daemon=True, name="mage-inference").start()
        if not self._ready.wait(600):
            raise RuntimeError("inference worker did not finish loading the model")
        if self._load_error is not None:
            raise self._load_error

    # ------------------------------------------------------------- worker

    def _worker_main(self):
        """Owns the model + processor + gate. Runs queued tasks forever."""
        try:
            mx.eval(mx.array([1.0]))  # ensure this thread's GPU stream exists
            ensure_plugin_installed()
            t0 = time.time()
            self.model = load_model(self.model_dir)
            self.proc = MageVLProcessor(str(self.model_dir))
            self._merge = self.proc.merge_size

            self.gate: Optional[StreamMindGate] = None
            self.gate_path: Optional[str] = None
            if self._gate_path and Path(self._gate_path).exists():
                g = StreamMindGate()
                w = mx.load(self._gate_path)
                g.load_weights(list(g.sanitize(w).items()))
                g.eval()
                self.gate = g
                self.gate_path = str(self._gate_path)
            log.info(
                "model ready in %.1fs | gate=%s | compile_vision=%s",
                time.time() - t0,
                f"ON ({self.gate_path})" if self.gate else "OFF (no streammind_gate weights)",
                self.compile_vision,
            )
            self._ready.set()
        except Exception as exc:  # noqa: BLE001
            self._load_error = exc
            self._ready.set()
            return

        while True:
            task = self._queue.get()
            try:
                task()
            except Exception:  # noqa: BLE001 — a bad task must not kill the worker
                log.exception("inference worker task failed")

    def submit(self, fn: Callable[[], None]):
        """Enqueue ``fn`` onto the inference worker thread."""
        self._queue.put(fn)

    # ---------------------------------------------------------------- video prep
    # Pure CPU / numpy — safe on any thread.

    def preprocess_video(self, video_path, num_frames=None, max_frames=32, target_fps=None,
                         max_pixels: Optional[int] = None):
        """Decode + sample frames, with an input-token budget guardrail.

        High-res screen recordings blow up model input tokens (each merged patch
        becomes a text token, and the Qwen3 KV cache is ~151MB per 1k tokens).
        If ``frames x patches/frame`` exceeds ``max_input_tokens``, frames are
        automatically halved (down to 4) until the budget fits. Falls back to a
        clear error when even 4 frames are over budget.

        ``max_pixels`` overrides the per-frame pixel budget (default: the engine's
        ``video_max_pixels``) — video-analysis callers use a smaller budget for
        the whole-video narrative pass (structure/style/story doesn't need to
        read fine text, and fewer pixels = much cheaper prefill).
        """
        max_pixels = max_pixels or self.video_max_pixels
        n = max(1, min(num_frames or 16, max_frames))
        requested = n
        frames = indices = fps = None
        data = None
        est = n_per_frame = 0
        reduced = False
        for _ in range(6):
            frames, indices, fps = vp.decode_video_frames(
                str(video_path), max_frames=max_frames,
                fixed_num_frames=n, target_fps=target_fps,
            )
            data = vp.preprocess_video_frames(
                frames, indices, fps,
                patch_size=self.proc.patch_size,
                merge_size=self.proc.merge_size,
                min_pixels=self.proc.min_pixels,
                max_pixels=max_pixels,
            )
            grid = data["image_grid_thw"]
            T = grid.shape[0]
            n_per_frame = int(grid[0, 1] * grid[0, 2]) // (self._merge * self._merge)
            est = T * n_per_frame + 256  # + text tokens
            if est <= self.max_input_tokens or n <= 4:
                break
            reduced = True
            n = max(4, n // 2)

        if est > self.max_input_tokens:
            raise ValueError(
                f"输入 token 超预算：每帧 {n_per_frame} × {T} 帧 ≈ {est} token，"
                f"超过上限 {self.max_input_tokens}。请降低采样帧数或视频分辨率。"
            )
        if reduced:
            log.info("input tokens over budget, auto-reduced frames %d -> %d "
                     "(%d tok/frame, est %d)", requested, len(indices), n_per_frame, est)
        return {
            "frames": frames, "indices": indices, "fps": fps, "data": data,
            "input_tokens_est": est, "frames_reduced": reduced, "frames_requested": requested,
        }

    # ------------------------------------------------------ streaming vision pass
    # Worker-side (touches mlx.core).

    def _vision_step(self, gh: int, gw: int, patch_positions: np.ndarray):
        """Vision forward builder for one frame of grid (1, gh, gw).

        A single-frame grid never splits into attention windows, so RoPE freqs
        and the (absent) mask are constants — freeze them once and compile the
        layer stack; repeated grids hit the cached step. Output is numerically
        identical to the plain path.
        """
        key = (gh, gw)
        ent = self._vision_static.get(key)
        if ent is None:
            tower = self.model.vision_tower
            pp = mx.array(patch_positions)
            freqs, mask = tower.prepare_static_inputs(mx.array([[1, gh, gw]]), pp)
            mx.eval(freqs)
            if self.compile_vision:
                step = mx.compile(
                    lambda px_: tower.forward_with_inputs(px_, freqs, mask))
            else:
                step = lambda px_: tower.forward_with_inputs(px_, freqs, mask)
            ent = {"step": step, "n_patches": int(pp.shape[0]),
                   "p_merged": int(pp.shape[0]) // (self._merge * self._merge)}
            self._vision_static[key] = ent
        return ent

    def vision_features(self, prep, on_frame: Optional[FrameCB] = None):
        """Per-frame vision prefill -> vision tokens [1, T, P_merged, D].

        Each frame is its own attention window (t=1), so this is exactly the
        model's image path run frame by frame — real streaming prefill.
        ``on_frame(t, second)`` fires after every completed frame.
        Returns ``(tokens, stats)`` where stats has total_s / per_frame_s.
        """
        t0 = time.time()
        data = prep["data"]
        grid = data["image_grid_thw"]
        gh, gw = int(grid[0, 1]), int(grid[0, 2])
        p_raw = gh * gw                                  # pixel rows per frame (pre-merge)
        p_merged = p_raw // (self._merge * self._merge)  # merged patches per frame
        T = grid.shape[0]
        seconds = data["frame_seconds"]

        dtype = self.model.vision_tower.patch_embedding.weight.dtype
        px_all = np.asarray(data["pixel_values"])
        pp_all = np.asarray(data["patch_positions"])
        ent = self._vision_step(gh, gw, pp_all[:p_raw])

        tokens = []
        per_frame_s = []
        for t in range(T):
            ft = time.time()
            vt = ent["step"](mx.array(px_all[t * p_raw:(t + 1) * p_raw]).astype(dtype))
            vt = vt.reshape(1, 1, p_merged, -1).astype(mx.float32)
            mx.eval(vt)
            tokens.append(vt)
            per_frame_s.append(time.time() - ft)
            if on_frame:
                on_frame(t, seconds[t])
        return (mx.concatenate(tokens, axis=1),  # [1, T, P, D]
                {"total_s": round(time.time() - t0, 3),
                 "per_frame_s": round(sum(per_frame_s) / max(len(per_frame_s), 1), 3)})

    # --------------------------------------------------------------- event gate
    # Worker-side.

    def gate_timeline(self, vision_tokens, threshold: float, on_prob: Optional[ProbCB] = None):
        """Incremental StreamMind scoring.

        The Mamba scan is causal and the ClsNet reads each time step as its own
        length-1 sequence, so P(speak) at step t depends only on frames 0..t —
        scoring prefixes gives the exact same numbers, one frame at a time.
        Returns ``(probs, stats)``.
        """
        t0 = time.time()
        if self.gate is None:
            return [None] * vision_tokens.shape[1], {"total_s": 0.0, "enabled": False}
        T = vision_tokens.shape[1]
        probs = []
        for t in range(T):
            p = self.gate.speak_probs(vision_tokens[:, :t + 1])  # [1, t+1]
            mx.eval(p)
            probs.append(float(p[0, -1].item()))
            if on_prob:
                on_prob(t, probs[-1])
        return probs, {"total_s": round(time.time() - t0, 3), "enabled": True}

    # ------------------------------------------------------------------ summary
    # Worker-side.

    def _video_chat_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        return self.proc.apply_chat_template(messages, add_generation_prompt=True)

    def _video_inputs(self, text: str, data):
        """Rebuild model inputs from cached preprocess (no re-decode)."""
        grid = data["image_grid_thw"]
        n_per_frame = int(grid[0, 1] * grid[0, 2]) // (self._merge * self._merge)
        text = self.proc._rewrite_video_pad_frames(text, n_per_frame, data["frame_seconds"])
        enc = self.proc.tokenizer.encode(text, add_special_tokens=False)
        return {
            "input_ids": mx.array([enc.ids]),
            "pixel_values": mx.array(data["pixel_values"]),
            "image_grid_thw": mx.array(grid),
            "patch_positions": mx.array(data["patch_positions"]),
        }

    def _image_chat_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        return self.proc.apply_chat_template(messages, add_generation_prompt=True)

    def _merged_embeds(self, ids: mx.array, image_features: mx.array) -> mx.array:
        """Merge vision features into the text embedding sequence exactly the way
        ``Model.__call__`` does (image-pad token positions replaced by features).

        ``ids``: [1, n]; ``image_features``: [..., D] with rows ordered like the
        image/video pad tokens appear left-to-right. Returns [1, n, D]."""
        cfg = self.model.config
        img_tok = int(cfg.image_token_id)
        vid_tok = int(getattr(cfg, "video_token_id", img_tok))
        embeds3d = self.model.language_model.model.embed_tokens(ids)
        feats2d = image_features.reshape(-1, image_features.shape[-1])
        return self.model.merge_input_ids_with_image_features(
            img_tok, vid_tok, feats2d, embeds3d, ids)

    @staticmethod
    def _apply_rep_penalty(logits_2d: mx.array, toks: list[int], penalty: float = 1.15) -> mx.array:
        """Down-weight already-generated tokens so greedy decoding climbs out of
        常见复述循环(如反复输出同一个短语)instead of running to the cap.
        ``logits_2d``: [1, vocab]."""
        if not toks:
            return logits_2d
        ids = list(dict.fromkeys(toks[-96:]))  # recent, deduped
        idx = mx.array(ids)
        vals = logits_2d[0, idx]
        logits_2d[0, idx] = mx.where(vals > 0, vals / penalty, vals * penalty)
        return logits_2d

    @staticmethod
    def _in_loop(toks: list[int]) -> bool:
        """True when the tail consists of ≥3 identical cycles of any length 4-32
        tokens — catches long phrase loops (e.g. "、编辑索引"×∞) that an exact
        6-token check misses."""
        n = len(toks)
        if n < 12:
            return False
        for L in range(4, 33):
            if n >= 3 * L and toks[-L:] == toks[-2 * L : -L] == toks[-3 * L : -2 * L]:
                return True
        return False

    def _greedy(self, ids, model_kwargs: dict, max_tokens: int,
                on_seg: Optional[SegCB] = None, stop_event=None,
                merged_embeds: Optional[mx.array] = None) -> dict:
        """Shared greedy decode (worker thread only). ``ids`` is [1, n];
        ``model_kwargs`` carries the visual inputs ({} for text-only); when
        ``merged_embeds`` is given it goes straight into the language model,
        skipping re-run of the vision path. ``on_seg`` receives incremental
        decoded segments (their concatenation equals the returned text).
        Returns ``{"text", "prefill_s", "decode_s", "tokens", "tokens_per_s"}``."""
        cache = [KVCache() for _ in self.model.layers]
        detok = StreamingDetokenizer(self.proc.tokenizer)
        t0 = time.time()
        if merged_embeds is not None:
            out = self.model.language_model(inputs=ids, cache=cache,
                                            inputs_embeds=merged_embeds)
        else:
            out = self.model(ids, cache=cache, **model_kwargs)
        y = mx.argmax(out.logits[:, -1, :], axis=-1)
        mx.eval(y)
        prefill_s = time.time() - t0

        toks: list[int] = []
        eos_hit = False

        def emit():
            seg = detok.last_segment
            if seg and on_seg:
                on_seg(seg)

        t1 = time.time()
        stop_requested = False
        for _ in range(max_tokens):
            if stop_event is not None and stop_event.is_set():
                stop_requested = True
                break
            tok = int(y.item())
            if tok in EOS_IDS:
                eos_hit = True
                break
            toks.append(tok)
            detok.add_token(tok)
            emit()
            # repetition guard: 3 identical cycles of any 4-32 token phrase
            if self._in_loop(toks):
                break
            out = self.model(y[None], cache=cache)
            y = mx.argmax(
                self._apply_rep_penalty(out.logits[:, -1, :], toks), axis=-1
            )
            mx.eval(y)
        detok.finalize()
        emit()
        decode_s = time.time() - t1
        return {
            "text": detok.text,
            "stopped": stop_requested,
            # the answer never reached its natural end (cap / repetition guard):
            # UIs should show a "truncated" hint instead of a broken sentence
            "truncated": not eos_hit and not stop_requested,
            "prefill_s": round(prefill_s, 3),
            "decode_s": round(decode_s, 3),
            "tokens": len(toks),
            "tokens_per_s": round(len(toks) / max(decode_s, 1e-9), 1),
        }

    def summarize(self, prep, prompt: str, max_tokens: int = 300,
                  on_seg: Optional[SegCB] = None, stop_event=None,
                  vision_tokens: Optional[mx.array] = None) -> dict:
        """Greedy full-video generation (worker thread). Builds the video inputs
        from a cached preprocess (no re-decode). Pass the cached ``vision_tokens``
        from :meth:`vision_features` to skip the duplicate vision forward.
        Returns the same stats dict as ``_greedy``."""
        data = prep["data"]
        kwargs = self._video_inputs(self._video_chat_text(prompt), data)
        ids = kwargs.pop("input_ids")
        merged = None
        if vision_tokens is not None:
            merged = self._merged_embeds(ids, vision_tokens)
            kwargs = {}
        return self._greedy(ids, kwargs, max_tokens, on_seg=on_seg,
                            stop_event=stop_event, merged_embeds=merged)

    def generate_image(self, image, prompt: str, max_tokens: int = 120,
                       on_seg: Optional[SegCB] = None,
                       max_pixels: Optional[int] = None) -> dict:
        """Single-image generation (worker thread). Returns the same stats dict.
        ``max_pixels`` optionally downscales the image first (cheaper prefill;
        fine for scene-level analysis, too coarse for reading small subtitles)."""
        if max_pixels is not None and image is not None:
            w, h = image.size
            if w * h > max_pixels:
                scale = (max_pixels / (w * h)) ** 0.5
                image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                                     Image.LANCZOS)
        text = self._image_chat_text(prompt)
        inp = self.proc(text, images=image)
        ids = mx.array(inp["input_ids"])
        kwargs = dict(
            pixel_values=mx.array(inp["pixel_values"]),
            image_grid_thw=mx.array(inp["image_grid_thw"]),
            patch_positions=mx.array(inp["patch_positions"]),
        )
        return self._greedy(ids, kwargs, max_tokens, on_seg=on_seg)

    def generate_text(self, prompt: str, max_tokens: int = 300,
                      on_seg: Optional[SegCB] = None) -> dict:
        """Text-only generation (worker thread). Returns the same stats dict."""
        text = self.proc.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True)
        ids = mx.array(self.proc.tokenizer.encode(text, add_special_tokens=False).ids)[None, :]
        return self._greedy(ids, {}, max_tokens, on_seg=on_seg)

    # ------------------------------------------------------------ single frame
    # Public API — blocks on the worker.

    def describe_frame(self, image_bytes: bytes, prompt: str = "用一句话描述当前屏幕画面。",
                       max_tokens: int = 80, max_pixels: Optional[int] = None) -> dict:
        """Analyze a single frame; blocks until done (compat API)."""
        fut = _Future()

        def task():
            try:
                fut.set_result(self.describe_frame_stream(
                    image_bytes, prompt, max_tokens, max_pixels=max_pixels))
            except Exception as exc:  # noqa: BLE001
                fut.set_exception(exc)

        self.submit(task)
        res = fut.result(timeout=600)
        return {"description": res["text"], "elapsed_s": round(res["timings"]["total_ms"] / 1000, 2)}

    def describe_frame_stream(self, image_bytes: bytes, prompt: str,
                              max_tokens: int = 64,
                              max_pixels: Optional[int] = 800_000,
                              on_seg: Optional[SegCB] = None) -> dict:
        """Fast single-frame path (worker thread) with per-stage timings.

        Fixed-shape vision step (compiled when the grid repeats) + merged-embeds
        prefill + streaming decode. ``max_pixels`` downscales big frames; pass
        ``None`` to analyze at native resolution.
        """
        t_start = time.time()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if max_pixels is not None:
            w, h = img.size
            if w * h > max_pixels:
                scale = (max_pixels / (w * h)) ** 0.5
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                                 Image.LANCZOS)

        inp = self.proc(self._image_chat_text(prompt), images=img)
        t_prep = time.time()
        ids = mx.array(inp["input_ids"])

        grid = inp["image_grid_thw"][0]
        gh, gw = int(grid[1]), int(grid[2])
        n_raw = gh * gw
        ent = self._vision_step(
            gh, gw, np.asarray(inp["patch_positions"])[:n_raw])
        dtype = self.model.vision_tower.patch_embedding.weight.dtype
        t_vision = time.time()
        feats = ent["step"](
            mx.array(np.asarray(inp["pixel_values"])).astype(dtype))
        mx.eval(feats)
        t_prefill = time.time()

        merged = self._merged_embeds(ids, feats)
        timings = {
            "pil_ms": round((t_prep - t_start) * 1000, 1),
            "prep_ms": round((t_vision - t_prep) * 1000, 1),
            "vision_ms": round((t_prefill - t_vision) * 1000, 1),
        }
        res = self._greedy(ids, {}, max_tokens, on_seg=on_seg, merged_embeds=merged)
        timings.update({
            "prefill_ms": round(res["prefill_s"] * 1000, 1),
            "decode_ms": round(res["decode_s"] * 1000, 1),
            "total_ms": round((time.time() - t_start) * 1000, 1),
        })
        res["timings"] = timings
        res["grid"] = [1, gh, gw]
        res["tokens_per_frame"] = ent["p_merged"]
        return res