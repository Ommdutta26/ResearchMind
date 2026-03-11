"""
ui/sidebar.py
Renders the Streamlit sidebar: configuration, knowledge ingestion, history.
"""

from __future__ import annotations

import streamlit as st

from core.config import GROQ_PRICING
from core.database import load_history
from utils.retrieval import ingest_pdf, ingest_youtube


def render_sidebar(db_conn) -> None:
    """Render the full sidebar and update st.session_state in place."""

    with st.sidebar:

        # ── Configuration ─────────────────────────────────────────────────
        st.markdown("### ⚙️ Configuration")

        st.session_state.sel_model = st.selectbox(
            "Model",
            list(GROQ_PRICING.keys()),
            index=list(GROQ_PRICING.keys()).index(
                st.session_state.get("sel_model", "llama-3.3-70b-versatile")
            ),
        )

        st.session_state.depth = st.select_slider(
            "Search Depth",
            ["quick", "deep", "exhaustive"],
            value=st.session_state.get("depth", "deep"),
        )

        st.session_state.max_loops = st.slider(
            "Reflection Loops",
            0, 3,
            value=st.session_state.get("max_loops", 1),
        )

        st.session_state.budget_k = st.slider(
            "Token Budget (k)",
            10, 200,
            value=st.session_state.get("budget_k", 80),
        )

        st.markdown("---")

        # ── PDF Ingestion ─────────────────────────────────────────────────
        st.markdown("### 📚 Knowledge Sources")

        uploaded_pdfs = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_pdfs:
            for pdf in uploaded_pdfs:
                path = f"temp_{pdf.name}"
                with open(path, "wb") as f:
                    f.write(pdf.read())
                try:
                    chunks = ingest_pdf(path)
                    st.success(f"{pdf.name} ingested ({chunks} chunks)")
                except Exception as e:
                    st.error(f"PDF error: {e}")

        st.markdown("---")

        # ── YouTube Ingestion ─────────────────────────────────────────────
        youtube_url = st.text_input("YouTube Video URL")

        if st.button("Ingest YouTube Video"):
            if youtube_url:
                try:
                    chunks = ingest_youtube(youtube_url)
                    st.success(f"YouTube transcript ingested ({chunks} chunks)")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.markdown("---")

        # ── History ───────────────────────────────────────────────────────
        st.markdown("### History")

        for h in load_history(db_conn, limit=8):
            icon = {"done": "✅", "error": "❌"}.get(h["status"], "🔄")
            if st.button(
                f"{icon} {h['topic'][:18]} {h['ts'][5:]}",
                key=f"h_{h['id']}",
            ):
                st.session_state.thread_id = h["id"]
                st.session_state.phase     = h["status"]
                st.rerun()

        st.markdown("---")

        # ── New Session ───────────────────────────────────────────────────
        import uuid

        if st.button("🆕 New Session"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.phase     = "idle"
            st.session_state.node_logs = []
            st.rerun()
