"""mage-vl-mlx server — optimized real-time recognition over one WebSocket.

    MAGE_MODEL_DIR=./mage-vl-mlx python -m mage_vl.server     # :8010

Endpoints:
    GET  /api/health                     model status
    GET  /api/history/sessions           recorded sessions
    GET  /api/history/sessions/{id}      session + frames
    GET  /api/history/sessions/{sid}/frames/{fid}/thumb.jpg
    DELETE /api/history/sessions/{id}
    WS   /ws/stream                      JPEG frames in, token stream out
    /                                    single-file web client (web/index.html)

Every analyzed frame is auto-recorded into SQLite (data/history.db).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .engine import MageVLEngine
from .history_db import HistoryDB, make_thumbnail
from .severity import parse_severity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mage_vl.server")

DEFAULT_PROMPT = "用一到两句话描述当前画面。"
DEFAULT_MAX_TOKENS = 8000
MAX_TOKENS_LIMIT = 8000
DEFAULT_MAX_PIXELS = 2_073_600  # 1080px — small text (stock tickers) needs it

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MAGE_MODEL_DIR", ROOT / "mage-vl-mlx"))
DATA_DIR = Path(os.environ.get("MAGE_DATA_DIR", ROOT / "data"))
WEB_DIR = ROOT / "web"
PORT = int(os.environ.get("MAGE_PORT", "8010"))

engine: MageVLEngine
db: HistoryDB


# ----------------------------------------------------------------- session


@dataclass
class StreamSession:
    ws: WebSocket
    loop: asyncio.AbstractEventLoop
    sid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    prompt: str = DEFAULT_PROMPT
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_pixels: int = DEFAULT_MAX_PIXELS
    inflight: Optional[int] = None
    pending: Optional[tuple[int, bytes]] = None
    skipped: int = 0
    next_fid: int = 0
    rec_sid: Optional[str] = None
    rec_saved: int = 0

    def send(self, payload: dict) -> None:
        asyncio.run_coroutine_threadsafe(self._send(payload), self.loop)

    async def _send(self, payload: dict) -> None:
        try:
            await self.ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:  # noqa: BLE001 — client gone
            pass

    async def on_frame(self, data: bytes) -> None:
        if not data:
            return
        fid = self.next_fid
        self.next_fid += 1
        if self.inflight is not None or self.pending is not None:
            self.skipped += 1
            self.pending = (fid, data)
            return
        self._launch(fid, data)

    def _launch(self, fid: int, data: bytes) -> None:
        self.inflight = fid
        sess = self

        def task():
            try:
                res = engine.describe_frame_stream(
                    data, sess.prompt, max_tokens=sess.max_tokens,
                    max_pixels=sess.max_pixels,
                    on_seg=lambda seg: sess.send(
                        {"type": "seg", "fid": fid, "text": seg}),
                )
                payload = {
                    "type": "result", "fid": fid, "text": res["text"],
                    "truncated": res.get("truncated", False),
                    "timings": res["timings"], "tokens": res["tokens"],
                    "tokens_per_s": res["tokens_per_s"],
                    "fps_hint_ms": max(120, round(res["timings"]["total_ms"] * 1.15)),
                    "skipped": sess.skipped,
                }
                if sess.rec_sid and db is not None:
                    severity = parse_severity(res["text"])
                    timings = dict(res["timings"])
                    if res.get("truncated"):
                        timings["truncated"] = True
                    frame_id = db.add_frame(
                        sess.rec_sid, fid, make_thumbnail(data), res["text"],
                        severity, res["tokens"], timings)
                    sess.rec_saved += 1
                    payload["severity"] = severity
                    payload["rec_saved"] = sess.rec_saved
                    if frame_id:
                        payload["frame_url"] = (
                            f"/api/history/sessions/{sess.rec_sid}"
                            f"/frames/{frame_id}/thumb.jpg")
                sess.send(payload)
            except Exception as exc:  # noqa: BLE001
                log.exception("frame %s failed", fid)
                sess.send({"type": "error", "fid": fid, "message": str(exc)[:300]})
            asyncio.run_coroutine_threadsafe(sess._finish(fid), sess.loop)

        engine.submit(task)

    async def _finish(self, fid: int) -> None:
        if self.inflight != fid:
            return
        self.inflight = None
        nxt, self.pending = self.pending, None
        if nxt is not None:
            self._launch(nxt[0], nxt[1])

    def start_recording(self) -> None:
        if self.rec_sid or db is None:
            return
        rec = db.create_session(
            f"识别 · {time.strftime('%m-%d %H:%M:%S')}", "stream", self.prompt)
        self.rec_sid = rec["id"]
        self.rec_saved = 0
        self.send({"type": "session", "state": "recording",
                   "rec_sid": self.rec_sid, "title": rec["title"]})

    def stop_recording(self) -> None:
        if not self.rec_sid or db is None:
            return
        db.close_session(self.rec_sid)
        self.rec_sid = None
        self.send({"type": "session", "state": "stopped", "frames": self.rec_saved})


# ---------------------------------------------------------------------- app

app = FastAPI(title="mage-vl-mlx", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.on_event("startup")
def load_model() -> None:
    global engine, db
    t0 = time.time()
    engine = MageVLEngine(str(MODEL_DIR))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = HistoryDB(DATA_DIR / "history.db")
    log.info("engine ready in %.1fs | model=%s", time.time() - t0, MODEL_DIR)


@app.get("/api/health")
def health():
    return {"status": "ok", "model": str(MODEL_DIR)}


@app.get("/api/history/sessions")
def history_sessions():
    return db.list_sessions()


@app.get("/api/history/sessions/{sid}")
def history_session(sid: str):
    sess = db.get_session(sid)
    if sess is None:
        raise HTTPException(404, f"session {sid!r} not found")
    out = dict(sess)
    out["frames"] = db.list_frames(sid)
    return out


@app.get("/api/history/sessions/{sid}/frames/{frame_id}/thumb.jpg")
def history_thumb(sid: str, frame_id: int):
    thumb = db.get_thumb(sid, frame_id)
    if thumb is None:
        raise HTTPException(404, "frame not found")
    return Response(content=thumb, media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=31536000, immutable"})


@app.delete("/api/history/sessions/{sid}")
def history_delete(sid: str):
    if not db.delete_session(sid):
        raise HTTPException(404, f"session {sid!r} not found")
    return {"status": "deleted"}


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    sess = StreamSession(ws=ws, loop=loop)
    await sess._send({"type": "hello", "sid": sess.sid, "prompt": sess.prompt,
                      "max_tokens": sess.max_tokens})
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await sess.on_frame(msg["bytes"])
            elif msg.get("text"):
                try:
                    ctl = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                op = ctl.get("op")
                if op == "ping":
                    await sess._send({"type": "pong"})
                elif op == "config":
                    if isinstance(ctl.get("prompt"), str) and ctl["prompt"].strip():
                        sess.prompt = ctl["prompt"].strip()
                    if isinstance(ctl.get("max_tokens"), int):
                        sess.max_tokens = max(16, min(ctl["max_tokens"], MAX_TOKENS_LIMIT))
                    if isinstance(ctl.get("max_pixels"), int):
                        sess.max_pixels = max(100_000, min(ctl["max_pixels"], 4_000_000))
                    await sess._send({"type": "config", "prompt": sess.prompt,
                                      "max_tokens": sess.max_tokens,
                                      "max_pixels": sess.max_pixels})
                elif op == "run_start":
                    sess.start_recording()
                elif op == "run_stop":
                    sess.stop_recording()
    except WebSocketDisconnect:
        pass
    finally:
        sess.stop_recording()
        log.info("session %s closed frames=%d skipped=%d",
                 sess.sid, sess.next_fid, sess.skipped)


# single-file web client
if (WEB_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("MAGE_HOST", "127.0.0.1"),
                port=PORT, log_level="warning")
