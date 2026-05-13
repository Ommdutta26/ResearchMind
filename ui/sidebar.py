"""
ui/sidebar.py
Streamlit sidebar: model settings, PDF/YouTube ingestion, history,
scheduled research queue, and topic auto-suggest.
"""

from __future__ import annotations

import sqlite3
import streamlit as st

from core.config import GROQ_PRICING, DEFAULT_MODEL, DEFAULT_DEPTH, DEFAULT_LOOPS, DEFAULT_BUDGET
from core.database import load_history
from utils.finops import calc_cost


def render_sidebar(db_conn: sqlite3.Connection) -> None:
    with st.sidebar:
        st.markdown(
            '<div style="font-family:Syne,sans-serif;font-weight:700;'
            'font-size:1rem;color:#e8e6f0;letter-spacing:1px;margin-bottom:12px">'
            "⚙ Settings</div>",
            unsafe_allow_html=True,
        )

        # ── Model selection ───────────────────────────────────────────────
        st.session_state.sel_model = st.selectbox(
            "LLM Model",
            list(GROQ_PRICING.keys()),
            index=list(GROQ_PRICING.keys()).index(
                st.session_state.get("sel_model", DEFAULT_MODEL)
            ),
        )

        # ── Research depth ────────────────────────────────────────────────
        st.session_state.depth = st.select_slider(
            "Research depth",
            options=["quick", "deep", "exhaustive"],
            value=st.session_state.get("depth", DEFAULT_DEPTH),
        )

        # ── Reflection loops ──────────────────────────────────────────────
        st.session_state.max_loops = st.slider(
            "Max reflection loops", 1, 4,
            value=st.session_state.get("max_loops", DEFAULT_LOOPS),
        )

        # ── Token budget ──────────────────────────────────────────────────
        st.session_state.budget_k = st.slider(
            "Token budget (k)", 20, 200,
            value=st.session_state.get("budget_k", DEFAULT_BUDGET),
        )

        st.divider()

        # ── Ingestion: PDF ────────────────────────────────────────────────
        st.markdown("**📄 Ingest document**")
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded:
            import tempfile, os
            from utils.retrieval import ingest_pdf
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            with st.spinner("Ingesting PDF…"):
                try:
                    n = ingest_pdf(tmp_path)
                    st.success(f"✅ {n} chunks added to knowledge base")
                except Exception as e:
                    st.error(f"Ingestion error: {e}")
                finally:
                    os.unlink(tmp_path)

        # ── Ingestion: YouTube ────────────────────────────────────────────
        yt_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=…")
        if st.button("Ingest YouTube") and yt_url:
            from utils.retrieval import ingest_youtube
            with st.spinner("Fetching transcript…"):
                try:
                    n = ingest_youtube(yt_url)
                    st.success(f"✅ {n} chunks added")
                except Exception as e:
                    st.error(f"YouTube error: {e}")

        st.divider()

        # ── Topic auto-suggest ────────────────────────────────────────────
        st.markdown("**💡 Trending topics**")
        if st.button("Fetch trending", use_container_width=True):
            from core.models import get_tools
            try:
                tavily, _, _ = get_tools()
                raw = tavily.invoke({"query": "trending research topics 2025"})
                items = raw if isinstance(raw, list) else raw.get("results", [])
                topics = [
                    r.get("title", "")[:60]
                    for r in items[:5]
                    if isinstance(r, dict) and r.get("title")
                ]
                if topics:
                    st.session_state["suggested_topics"] = topics
            except Exception as e:
                st.warning(f"Suggest failed: {e}")

        for t in st.session_state.get("suggested_topics", []):
            if st.button(t, key=f"sug_{t}", use_container_width=True):
                st.session_state["topic_prefill"] = t

        st.divider()

        # ── Scheduled research ────────────────────────────────────────────
        st.markdown("**⏰ Schedule research**")
        sched_topic = st.text_input("Topic to schedule", key="sched_topic")
        sched_email = st.text_input("Email for delivery", key="sched_email")
        sched_freq  = st.selectbox("Frequency", ["daily", "weekly"], key="sched_freq")

        if st.button("Add to queue") and sched_topic and sched_email:
            try:
                db_conn.execute(
                    "INSERT INTO topic_queue (user_id, topic, schedule, email, active) "
                    "VALUES (?,?,?,?,1)",
                    ("default", sched_topic, sched_freq, sched_email),
                )
                db_conn.commit()
                st.success("✅ Scheduled!")
            except Exception as e:
                st.error(str(e))

        st.divider()

        # ── Run history ───────────────────────────────────────────────────
        st.markdown("**📚 Recent runs**")
        history = load_history(db_conn)
        if history:
            for row in history[:8]:
                eval_tag = (
                    f" · eval {row['eval_score']:.0%}"
                    if row.get("eval_score") is not None else ""
                )
                cost_str = (
                    f"${row['cost']:.4f}" if row.get("cost") else "—"
                )
                st.markdown(
                    f"<div style='font-size:.75rem;color:#8a87a0;padding:3px 0;"
                    f"border-bottom:1px solid #2a2a4a'>"
                    f"<b style='color:#c0bdd8'>{row['topic'][:30]}</b><br>"
                    f"{row['ts']} · {cost_str}{eval_tag}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No runs yet.")
