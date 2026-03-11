"""
app.py
NexusResearch – main Streamlit entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
import uuid

import streamlit as st
from langgraph.types import Command

from core.config import DEFAULT_MODEL, DEFAULT_DEPTH, DEFAULT_LOOPS, DEFAULT_BUDGET
from core.database import get_memory, init_db, save_history
from agents.graph import build_graph
from ui.styles import inject_css
from ui.components import render_log, render_metrics, render_badge, render_header
from ui.sidebar import render_sidebar
from utils.finops import total_tokens, calc_cost

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔬",
    layout="wide",
)

inject_css()

# ── Shared singletons ─────────────────────────────────────────────────────────
memory, db_conn = get_memory()
init_db(db_conn)
graph = build_graph(memory)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "thread_id":  str(uuid.uuid4()),
    "phase":      "idle",
    "node_logs":  [],
    "sel_model":  DEFAULT_MODEL,
    "depth":      DEFAULT_DEPTH,
    "max_loops":  DEFAULT_LOOPS,
    "budget_k":   DEFAULT_BUDGET,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def cfg() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


# ── Sidebar ───────────────────────────────────────────────────────────────────
render_sidebar(db_conn)

# ── Header ────────────────────────────────────────────────────────────────────
render_header()

# ── Topic input ───────────────────────────────────────────────────────────────
topic = st.text_input("Enter Research Topic:", "NVIDIA Stock 2026 Projections")

col1, col2 = st.columns([3, 1])

with col1:
    start_disabled = st.session_state.phase in ("running", "paused")

    if st.button(
        "🚀 Start Agent",
        use_container_width=True,
        disabled=start_disabled,
    ):
        st.session_state.phase     = "running"
        st.session_state.node_logs = []

        init_state = {
            "topic":            topic,
            "queries":          [],
            "research_data":    [],
            "final_summary":    "",
            "usage_metadata":   [],
            "source_urls":      [],
            "draft_report":     "",
            "critique":         {},
            "reflection_count": 0,
            "node_log":         [],
            "start_time":       time.time(),
            "status":           "running",
        }

        log_box = st.empty()

        try:
            for event in graph.stream(init_state, cfg()):
                nd = event[list(event.keys())[0]]
                st.session_state.node_logs.extend(nd.get("node_log", []))
                with log_box.container():
                    render_log(st.session_state.node_logs)
        except Exception as e:
            st.session_state.phase = "error"
            st.error(f"Agent error: {e}")

        snap = graph.get_state(cfg())
        if snap.next:
            st.session_state.phase = "paused"
        elif st.session_state.phase != "error":
            st.session_state.phase = "done"

        st.rerun()

with col2:
    render_badge(st.session_state.phase)

# ── HITL panel ────────────────────────────────────────────────────────────────
snap = graph.get_state(cfg())
sv   = dict(snap.values) if snap.values else {}

if snap.next:
    st.session_state.phase = "paused"
    research = list(sv.get("research_data") or [])
    n        = len(research)

    st.markdown(
        """<div class="hitl-box">
          <b style="font-family:Syne,sans-serif;color:#ffab40;font-size:.8rem;letter-spacing:2px">
            ⏸ AWAITING YOUR APPROVAL</b>
          <div style="margin-top:6px">Agent collected research. Approve to generate the report.</div>
        </div>""",
        unsafe_allow_html=True,
    )

    if n == 0:
        st.error("⚠️ No data — check your TAVILY_API_KEY or connection.")
    else:
        st.success(f"✅ {n} snippets ready.")
        with st.expander("👁️ Preview Research Data", expanded=True):
            for i, s in enumerate(research[:3]):
                st.markdown(f"**Snippet {i+1}:**")
                st.code(str(s)[:400], language=None)

    urls = list(sv.get("source_urls") or [])
    if urls:
        st.markdown(
            "**Sources:** " + "  ".join(f"[{i+1}]({u})" for i, u in enumerate(urls[:6]))
        )

    extra = st.text_area(
        "Extra instructions (optional)",
        placeholder="e.g. Focus on risks, compare with AMD…",
        height=70,
    )

    a_col, r_col = st.columns(2)

    with a_col:
        if st.button("✅ Approve & Generate Report", use_container_width=True):
            lb2 = st.empty()
            try:
                for event in graph.stream(
                    Command(resume={"action": "approve", "extra": extra}), cfg()
                ):
                    nd = event[list(event.keys())[0]]
                    st.session_state.node_logs.extend(nd.get("node_log", []))
                    with lb2.container():
                        render_log(st.session_state.node_logs)
                st.session_state.phase = "done"
            except Exception as e:
                st.session_state.phase = "error"
                st.error(f"Resume error: {e}")
            st.rerun()

    with r_col:
        if st.button("❌ Reject", use_container_width=True):
            for _ in graph.stream(Command(resume={"action": "reject"}), cfg()):
                pass
            st.session_state.phase = "idle"
            st.rerun()

# ── Execution log ─────────────────────────────────────────────────────────────
if st.session_state.node_logs:
    st.markdown("---")
    with st.expander("📋 Execution Log", expanded=False):
        render_log(st.session_state.node_logs)

# ── Final report ──────────────────────────────────────────────────────────────
final_snap = graph.get_state(cfg())
fsv        = dict(final_snap.values) if final_snap.values else {}

if st.session_state.phase == "done" and fsv.get("final_summary"):
    st.markdown("---")
    st.markdown("#### 📄 Research Report")

    ul = list(fsv.get("usage_metadata") or [])
    if ul:
        render_metrics(ul, st.session_state.sel_model, fsv.get("start_time", time.time()))

    critique = fsv.get("critique") or {}
    if critique.get("score"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Quality Score",    f"{critique.get('score', '?')}/10")
        c2.metric("Reflection Loops", fsv.get("reflection_count", 0))
        c3.metric(
            "Status",
            "✅ Approved" if critique.get("approved") else "⚠️ Forced",
        )
        if critique.get("suggestion"):
            st.caption(f"💡 {critique['suggestion']}")

    summary = fsv.get("final_summary", "")
    st.markdown(f'<div class="report-box">{summary}</div>', unsafe_allow_html=True)

    st.download_button(
        "⬇️ Download Report (.md)",
        data=summary,
        file_name=f"research_{time.strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
    )

    st.markdown("---")

    if st.checkbox("📊 Show Token Usage (FinOps)"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Input Tokens",  sum(u.get("prompt_tokens", 0)     for u in ul))
        c2.metric("Output Tokens", sum(u.get("completion_tokens", 0) for u in ul))
        c3.metric("Total Tokens",  total_tokens(ul))
        c4.metric("Est. Cost",     f"${calc_cost(ul, st.session_state.sel_model):.5f}")
        st.caption(f"Model: `{st.session_state.sel_model}`")

    save_history(
        db_conn,
        st.session_state.thread_id,
        topic,
        "done",
        calc_cost(ul, st.session_state.sel_model),
        total_tokens(ul),
        summary,
    )
