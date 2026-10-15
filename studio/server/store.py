"""SQLite persistence for saved voices and generation history."""
import json
import sqlite3
import threading
import time
import uuid

from . import config

_lock = threading.Lock()
_conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row
_conn.executescript(
    """
    CREATE TABLE IF NOT EXISTS voices (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        audio_file TEXT NOT NULL,
        ref_text TEXT NOT NULL DEFAULT '',
        language TEXT NOT NULL DEFAULT '',
        source TEXT NOT NULL DEFAULT 'upload',
        instruct TEXT NOT NULL DEFAULT '',
        created_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS history (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        title TEXT NOT NULL,
        files TEXT NOT NULL,
        meta TEXT NOT NULL DEFAULT '{}',
        created_at REAL NOT NULL
    );
    """
)
_conn.commit()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _voice(row) -> dict:
    return dict(row) if row else None


def list_voices() -> list[dict]:
    with _lock:
        return [dict(r) for r in _conn.execute("SELECT * FROM voices ORDER BY created_at DESC")]


def get_voice(voice_id: str) -> dict | None:
    with _lock:
        return _voice(_conn.execute("SELECT * FROM voices WHERE id = ?", (voice_id,)).fetchone())


def find_voice(key: str) -> dict | None:
    """Look a voice up by id or (case-insensitive) name."""
    with _lock:
        row = _conn.execute(
            "SELECT * FROM voices WHERE id = ? OR lower(name) = lower(?) ORDER BY created_at DESC",
            (key, key),
        ).fetchone()
    return _voice(row)


def add_voice(name, audio_file, ref_text="", language="", source="upload", instruct="") -> dict:
    voice = dict(id=_new_id(), name=name, audio_file=audio_file, ref_text=ref_text,
                 language=language, source=source, instruct=instruct, created_at=time.time())
    with _lock:
        _conn.execute(
            "INSERT INTO voices VALUES (:id, :name, :audio_file, :ref_text, :language, :source, :instruct, :created_at)",
            voice,
        )
        _conn.commit()
    return voice


def update_voice(voice_id: str, **fields) -> dict | None:
    allowed = {k: v for k, v in fields.items() if k in ("name", "ref_text", "language") and v is not None}
    if allowed:
        sets = ", ".join(f"{k} = ?" for k in allowed)
        with _lock:
            _conn.execute(f"UPDATE voices SET {sets} WHERE id = ?", (*allowed.values(), voice_id))
            _conn.commit()
    return get_voice(voice_id)


def delete_voice(voice_id: str) -> dict | None:
    voice = get_voice(voice_id)
    if voice:
        with _lock:
            _conn.execute("DELETE FROM voices WHERE id = ?", (voice_id,))
            _conn.commit()
    return voice


def _history(row) -> dict:
    item = dict(row)
    item["files"] = json.loads(item["files"])
    item["meta"] = json.loads(item["meta"])
    return item


def add_history(kind: str, title: str, files: dict, meta: dict | None = None) -> dict:
    item = dict(id=_new_id(), kind=kind, title=title[:200], files=json.dumps(files),
                meta=json.dumps(meta or {}, ensure_ascii=False), created_at=time.time())
    with _lock:
        _conn.execute("INSERT INTO history VALUES (:id, :kind, :title, :files, :meta, :created_at)", item)
        _conn.commit()
    return _history(item)


def list_history(limit: int = 200) -> list[dict]:
    with _lock:
        rows = _conn.execute("SELECT * FROM history ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_history(r) for r in rows]


def get_history(item_id: str) -> dict | None:
    with _lock:
        row = _conn.execute("SELECT * FROM history WHERE id = ?", (item_id,)).fetchone()
    return _history(row) if row else None


def delete_history(item_id: str) -> dict | None:
    item = get_history(item_id)
    if item:
        with _lock:
            _conn.execute("DELETE FROM history WHERE id = ?", (item_id,))
            _conn.commit()
    return item
