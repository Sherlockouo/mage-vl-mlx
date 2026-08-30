# mage-vl-mlx

Optimized **MLX port of [microsoft/Mage-VL](https://huggingface.co/microsoft/Mage-VL)** —
a 5B image/video VLM (Qwen3-4B text backbone + from-scratch Mage-ViT "Codec-ViT"
vision encoder) — purpose-built for **real-time recognition on Apple Silicon**.

This repo packages the model plugin, the weight converter, and a tuned
inference engine whose sole job is **low-latency frame → description**:

```
帧进 → 描述出   ·   ~0.4-0.9s/帧 @1080px (M4 Max, 4-bit)   ·   历史自动入库
```

Runs standalone: `pip install -e .`, convert the weights once, then

```bash
python -m mage_vl.server        # web UI on http://127.0.0.1:8010
python -m mage_vl.cli --model ./mage-vl-mlx --image photo.jpg
python scripts/benchmark.py --model ./mage-vl-mlx
```

## Optimizations for MLX (measured, M4 Max 64GB, 4-bit)

| # | Optimization | Effect |
|---|---|---|
| 1 | **4-bit quantization** (group 64) | 3.1GB weights; decode is bandwidth-bound so 4-bit wins (~1.6× over 8-bit) |
| 2 | **Frozen-input compiled ViT** — single-frame grids have no attention windows, so RoPE/mask are constants; layer stack runs under `mx.compile` | ViT forward 79→68ms (bit-identical), hits every frame |
| 3 | **Merged-embeds prefill** — cached vision tokens are merged into text embeddings and fed straight to the LM | video/summary passes run the ViT **once**, not twice |
| 4 | **Streaming incremental detokenizer** (`detokenize.py`) — byte-safe for CJK/emoji splits | token-level streaming, no full-history re-decode |
| 5 | **Repetition penalty + N-gram loop guard** — 1.15 penalty over generated tokens; any 4-32 token phrase repeated 3× ends generation | kills greedy-decode loops that previously ran to the token cap |
| 6 | **Fixed-rate adaptive sampling** — client samples at `max(用户间隔, 服务端建议)`, server pushes per-frame `fps_hint` | 发送节拍不被编码耗时拉长,过载自动降速不积压 |
| 7 | **Newest-frame-wins flow control** — while a frame is analyzed later arrivals replace the queued one | latency bounded under load |
| 8 | **Quantized KV cache** was evaluated and **rejected** (slower at every context length on M4 Max — dequant cost > bandwidth saved) | documented so you don't re-try it |

Benchmark (`scripts/benchmark.py`, M4 Max, 4-bit): text decode **56.5 tok/s**,
image decode **52.6 tok/s**, image prefill **318 tok/s**, peak 4.1GB.
Single-frame live recognition: **~390-870ms** end-to-end depending on 清晰度
(1080px reads stock-ticker digits correctly — 480px reads **0/6**, 1080px **6/6**).

## Setup

```bash
pip install -e ".[server,video]"      # mlx/mlx-vlm pulled in automatically
python scripts/install_plugin.py      # link plugin into mlx_vlm (auto-done by server/CLI too)
```

> Weights are **not** included. Convert once from the upstream checkpoint:
> `python scripts/convert.py --help` (HF → MLX 4/8-bit; see scripts/README hints
> inside each script).

## Real-time server

```bash
MAGE_MODEL_DIR=./mage-vl-mlx python -m mage_vl.server     # http://127.0.0.1:8010
```

One WebSocket (`/ws/stream`): JPEG frames in → token stream out. The bundled
single-file web client (`web/index.html`, served at `/`) provides
camera / screen-share / demo clip / local file sources, a two-column waterfall
of result cards, 识别间隔 & 清晰度 controls, and a history replay panel
(every analyzed frame auto-records to SQLite `data/history.db` —
compressed thumbnail + description + per-stage timings + alert severity).

Environment: `MAGE_MODEL_DIR`, `MAGE_DATA_DIR`, `MAGE_PORT`, `MAGE_HOST`.

## Layout

```
mage_vl/              # the mlx-vlm plugin (config/vision/mage_vl/processing/…)
  engine.py           #   optimized inference engine (all MLX tuning lives here)
  server.py           #   FastAPI: WS stream + history REST + web client
  history_db.py       #   SQLite frame history (thumbnails/timings/severity)
  detokenize.py       #   incremental streaming detokenizer
  severity.py         #   alert-level parsing (告警:无/注意/紧急 + keyword fallback)
  _link.py            #   idempotently links plugin into mlx_vlm.models
scripts/
  convert.py          # HF checkpoint → MLX 4/8-bit
  benchmark.py        # prefill/decode/peak-mem benchmark
  smoke_test.py       # structural test, no weights needed
  install_plugin.py   # manual plugin link
web/index.html        # single-file client (no build step)
tests/                # unit tests for guards/detokenizer helpers
```

## Status

| Piece | State |
|---|---|
| Config / Mage-ViT vision encoder / projector + Qwen3 stack | ✅ code + structural test |
| HF → MLX 4/8-bit conversion | ✅ `scripts/convert.py` |
| Native processor (bit-exact vs HF slow processor) | ✅ |
| Optimized engine (compiled ViT, merged embeds, streaming detok, rep-guard) | ✅ |
| Real-time WS server + web client + SQLite history | ✅ |
| Codec video backend (external `cv-preinfer` engine) | portable glue only |
| StreamMind event gate (`streaming.py`) | ✅ ported (optional weights) |

## License

Apache-2.0 — see `LICENSE` / `NOTICE`. Weights come from the upstream
Apache-2.0 checkpoint and are converted locally.
