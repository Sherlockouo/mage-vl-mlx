"""SQLite persistence for realtime monitoring history.

Stores one row per analyzed frame: compressed JPEG thumbnail (server-side,
320px wide), the model description, per-stage timings, tokens and the parsed
alert severity. Sessions group frames and carry the industry preset used.

The DB lives at ``<data_dir>/history.db`` (gitignored alongside uploads).
sqlite3 connections are opened per call — cheap for this write rate (~1/s)
and free of cross-thread issues with FastAPI's threadpool.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    service     TEXT NOT NULL,
    prompt      TEXT NOT NULL DEFAULT '',
    created     REAL NOT NULL,
    ended       REAL,
    frame_count INTEGER NOT NULL DEFAULT 0,
    alert_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS frames (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    fid         INTEGER NOT NULL,
    ts          REAL NOT NULL,          -- wall clock at capture (epoch s)
    description TEXT NOT NULL DEFAULT '',
    severity    TEXT NOT NULL DEFAULT '无',
    tokens      INTEGER NOT NULL DEFAULT 0,
    timings     TEXT NOT NULL DEFAULT '{}',
    thumb       BLOB NOT NULL,
    created     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id, fid);
"""


class HistoryDB:
    def __init__(self, db_path: Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        c.row_factory = sqlite3.Row
        return c

    # ------------------------------------------------------------- sessions

    def create_session(self, title: str, service: str, prompt: str) -> dict:
        sid = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute(
                "INSERT INTO sessions (id, title, service, prompt, created) "
                "VALUES (?,?,?,?,?)",
                (sid, title, service, prompt, time.time()))
        return self.get_session(sid)

    def get_session(self, sid: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
            return dict(row) if row else None

    def list_sessions(self, limit: int = 100) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, title, service, created, ended, frame_count, alert_count "
                "FROM sessions ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def close_session(self, sid: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE sessions SET ended=? WHERE id=? AND ended IS NULL",
                      (time.time(), sid))

    def delete_session(self, sid: str) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM sessions WHERE id=?", (sid,))
            return cur.rowcount > 0

    # --------------------------------------------------------------- frames

    def add_frame(self, sid: str, fid: int, thumb_jpeg: bytes, description: str,
                  severity: str, tokens: int, timings: dict) -> Optional[int]:
        """Insert a frame row and bump the session counters atomically."""
        now = time.time()
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO frames (session_id, fid, ts, description, severity, "
                "tokens, timings, thumb, created) VALUES (?,?,?,?,?,?,?,?,?)",
                (sid, fid, now, description, severity, tokens,
                 json.dumps(timings or {}), thumb_jpeg, now))
            c.execute(
                "UPDATE sessions SET frame_count = frame_count + 1, "
                "alert_count = alert_count + ? WHERE id=?",
                (1 if severity in ("注意", "紧急") else 0, sid))
            return cur.lastrowid

    def list_frames(self, sid: str, limit: int = 1000) -> list[dict]:
        """Frames without thumbnails (thumbs load via /thumb.jpg)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, fid, ts, description, severity, tokens, timings "
                "FROM frames WHERE session_id=? ORDER BY fid LIMIT ?",
                (sid, limit)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["timings"] = json.loads(d.get("timings") or "{}")
                d["thumb_url"] = f"/api/history/sessions/{sid}/frames/{d['id']}/thumb.jpg"
                out.append(d)
            return out

    def get_thumb(self, sid: str, frame_id: int) -> Optional[bytes]:
        with self._conn() as c:
            row = c.execute(
                "SELECT thumb FROM frames WHERE session_id=? AND id=?",
                (sid, frame_id)).fetchone()
            return row["thumb"] if row else None


def make_thumbnail(frame_jpeg: bytes, width: int = 320, quality: int = 62) -> bytes:
    """Downscale an uploaded frame to a small JPEG for history storage."""
    import io

    from PIL import Image
    img = Image.open(io.BytesIO(frame_jpeg)).convert("RGB")
    if img.width > width:
        h = max(1, round(img.height * width / img.width))
        img = img.resize((width, h), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()
