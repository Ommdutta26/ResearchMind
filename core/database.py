"""
core/database.py
SQLite-backed storage: run history, user sessions, report feedback, eval scores.
"""

from __future__ import annotations

import time
import sqlite3

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import DB_PATH


# ── Cached connection + checkpointer ─────────────────────────────────────────

@st.cache_resource
def get_memory():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(conn), conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS history (
            id      TEXT PRIMARY KEY,
            user_id TEXT DEFAULT 'default',
            topic   TEXT,
            ts      TEXT,
            status  TEXT,
            cost    REAL,
            tokens  INTEGER,
            summary TEXT,
            eval_score REAL DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id        TEXT PRIMARY KEY,
            thread_id TEXT,
            rating    INTEGER,
            comment   TEXT,
            ts        TEXT
        );

        CREATE TABLE IF NOT EXISTS topic_queue (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id  TEXT,
            topic    TEXT,
            schedule TEXT,
            email    TEXT,
            active   INTEGER DEFAULT 1
        );
    """)
    conn.commit()


def save_history(
    conn:       sqlite3.Connection,
    tid:        str,
    topic:      str,
    status:     str,
    cost:       float,
    tokens:     int,
    summary:    str,
    user_id:    str = "default",
    eval_score: float | None = None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO history VALUES (?,?,?,?,?,?,?,?,?)",
        (
            tid,
            user_id,
            topic,
            time.strftime("%Y-%m-%d %H:%M"),
            status,
            cost,
            tokens,
            str(summary)[:500],
            eval_score,
        ),
    )
    conn.commit()


def save_feedback(
    conn:      sqlite3.Connection,
    thread_id: str,
    rating:    int,
    comment:   str = "",
) -> None:
    import uuid
    conn.execute(
        "INSERT OR REPLACE INTO feedback VALUES (?,?,?,?,?)",
        (
            str(uuid.uuid4()),
            thread_id,
            rating,
            comment,
            time.strftime("%Y-%m-%d %H:%M"),
        ),
    )
    conn.commit()


def load_history(
    conn:    sqlite3.Connection,
    limit:   int = 15,
    user_id: str = "default",
) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT id,topic,ts,status,cost,tokens,eval_score "
            "FROM history WHERE user_id=? ORDER BY ts DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [
            {
                "id":         r[0],
                "topic":      r[1],
                "ts":         r[2],
                "status":     r[3],
                "cost":       r[4],
                "tokens":     r[5],
                "eval_score": r[6],
            }
            for r in rows
        ]
    except Exception:
        return []
