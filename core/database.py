"""
core/database.py
SQLite-backed run history: init, save, load.
"""

import time
import sqlite3

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import DB_PATH


# ── Cached connection + checkpointer ────────────────────────────────────────
@st.cache_resource
def get_memory():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(conn), conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS history (
            id      TEXT PRIMARY KEY,
            topic   TEXT,
            ts      TEXT,
            status  TEXT,
            cost    REAL,
            tokens  INTEGER,
            summary TEXT
        )"""
    )
    conn.commit()


def save_history(
    conn: sqlite3.Connection,
    tid: str,
    topic: str,
    status: str,
    cost: float,
    tokens: int,
    summary: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO history VALUES (?,?,?,?,?,?,?)",
        (
            tid,
            topic,
            time.strftime("%Y-%m-%d %H:%M"),
            status,
            cost,
            tokens,
            str(summary)[:300],
        ),
    )
    conn.commit()


def load_history(conn: sqlite3.Connection, limit: int = 15) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT id,topic,ts,status,cost,tokens FROM history "
            "ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "topic": r[1],
                "ts": r[2],
                "status": r[3],
                "cost": r[4],
                "tokens": r[5],
            }
            for r in rows
        ]
    except Exception:
        return []
