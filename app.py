"""
app.py  —  NexusResearch v2
Multi-agent AI research pipeline with:
  - Domain classification & specialist prompts
  - Streaming LLM output (analyst node)
  - Pydantic structured outputs (critic, fact_checker)
  - Persistent Chroma vector store (cross-session memory)
  - Cosine-similarity grounding
  - Inline citation engine
  - Automated eval harness
  - Rich export (PDF + DOCX)
  - Report feedback (👍/👎 → SQLite)
  - Trending topic auto-suggest
  - Scheduled research queue
  - Pipeline progress bar
"""

import time
import uuid

import streamlit as st
from langgraph.types import Command

from core.config import DEFAULT_MODEL, DEFAULT_DEPTH, DEFAULT_LOOPS, DEFAULT_BUDGET
from core.database import get_memory, init_db, save_history
from agents.graph import build_graph
from ui.styles import inject_css
from ui.components import (
    render_log,
    render_metrics,
    render_badge,
    render_header,
    render_progress,
    render_eval_score,
    render_citation_map,
    render_feedback,
)
from ui.sidebar import render_sidebar
from utils.finops import total_tokens, calc_cost
from utils.export import export_pdf, export_docx

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexusResearch",
    page_icon="🔬",
    layout="wide",
)
inject_css()

# ── Singletons ────────────────────────────────────────────────────────────────
memory, db_conn = get_memory()
init_db(db_conn)
graph = build_graph(memory)

# ── Session defaults ──────────────────────────────────────────────────────────
_DEFAULTS = {
    "thread_id":        str(uuid.uuid4()),
    "phase":            "idle",
    "node_logs":        [],
    "sel_model":        DEFAULT_MODEL,
    "depth":            DEFAULT_DEPTH,
    "max_loops":        DEFAULT_LOOPS,
    "budget_k":         DEFAULT_BUDGET,
    "suggested_topics": [],
    "topic_prefill":    "",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def cfg() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


# ── Sidebar ───────────────────────────────────────────────────────────────────
render_sidebar(db_conn)

# ── Header ────────────────────────────────────────────────────────────────────
# Show domain from last known state if available
_snap0 = graph.get_state(cfg())
_sv0   = dict(_snap0.values) if _snap0.values else {}
render_header(domain=_sv0.get("domain"))

# ── Topic input ───────────────────────────────────────────────────────────────
prefill = st.session_state.pop("topic_prefill", "") or "NVIDIA Stock 2026 Projections"
topic   = st.text_input("Enter Research Topic:", prefill)

col1, col2 = st.columns([3, 1])

with col1:
    start_disabled = st.session_state.phase in ("running", "paused")

    if st.button("🚀 Start Research", use_container_width=True, disabled=start_disabled):
        st.session_state.phase     = "running"
        st.session_state.node_logs = []
        st.session_state.thread_id = str(uuid.uuid4())   # fresh thread per run

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
            "citation_map":     None,
            "eval_score":       None,
            "domain":           None,
        }

        log_box   = st.empty()
        prog_box  = st.empty()

        try:
            for event in graph.stream(init_state, cfg()):
                nd = event[list(event.keys())[0]]
                st.session_state.node_logs.extend(nd.get("node_log", []))
                with log_box.container():
                    render_log(st.session_state.node_logs[-6:])
                with prog_box.container():
                    render_progress(st.session_state.node_logs)
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

# ── Pipeline progress (always visible during / after run) ─────────────────────
if st.session_state.node_logs:
    render_progress(st.session_state.node_logs)

# ── HITL approval panel ───────────────────────────────────────────────────────
snap = graph.get_state(cfg())
sv   = dict(snap.values) if snap.values else {}

if snap.next:
    st.session_state.phase = "paused"
    research = list(sv.get("research_data") or [])
    n        = len(research)
    domain   = sv.get("domain", "general")

    st.markdown(
        f"""<div class="hitl-box">
          <b style="font-family:Syne,sans-serif;color:#ffab40;font-size:.8rem;letter-spacing:2px">
            ⏸ AWAITING YOUR APPROVAL</b>
          <div style="margin-top:6px">
            Agent collected <b>{n}</b> research snippets for
            <b>{sv.get('topic','')}</b>
            <span class="domain-tag">⬡ {domain.upper()}</span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    if n == 0:
        st.error("⚠️ No data collected — check your TAVILY_API_KEY or network.")
    else:
        st.success(f"✅ {n} snippets ready from {len(set(sv.get('source_urls') or []))} sources.")
        with st.expander("👁️ Preview research snippets", expanded=True):
            for i, s in enumerate(research[:3]):
                st.markdown(f"**Snippet {i+1}:**")
                st.code(str(s)[:400], language=None)

    urls = list(sv.get("source_urls") or [])
    if urls:
        st.markdown(
            "**Sources:** "
            + "  ".join(f"[{i+1}]({u})" for i, u in enumerate(urls[:8]))
        )

    extra = st.text_area(
        "Extra instructions (optional)",
        placeholder="e.g. Focus on downside risks, compare with AMD, add a 2-year outlook…",
        height=70,
    )

    a_col, r_col = st.columns(2)

    with a_col:
        if st.button("✅ Approve & Generate Report", use_container_width=True):
            lb2 = st.empty()
            pb2 = st.empty()
            try:
                for event in graph.stream(
                    Command(resume={"action": "approve", "extra": extra}), cfg()
                ):
                    nd = event[list(event.keys())[0]]
                    st.session_state.node_logs.extend(nd.get("node_log", []))
                    with lb2.container():
                        render_log(st.session_state.node_logs[-6:])
                    with pb2.container():
                        render_progress(st.session_state.node_logs)
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
    with st.expander("📋 Execution log", expanded=False):
        render_log(st.session_state.node_logs)

# ── Final report ──────────────────────────────────────────────────────────────
final_snap = graph.get_state(cfg())
fsv        = dict(final_snap.values) if final_snap.values else {}

if st.session_state.phase == "done" and fsv.get("final_summary"):
    st.markdown("---")

    # ── Metrics bar ───────────────────────────────────────────────────────
    ul = list(fsv.get("usage_metadata") or [])
    if ul:
        render_metrics(ul, st.session_state.sel_model, fsv.get("start_time", time.time()))

    # ── Eval + quality indicators ─────────────────────────────────────────
    critique = fsv.get("critique") or {}
    eval_score = fsv.get("eval_score")

    c1, c2, c3, c4 = st.columns(4)
    if critique.get("score"):
        c1.metric("Critic score",     f"{critique['score']}/10")
        c2.metric("Reflection loops", fsv.get("reflection_count", 0))
        c3.metric("Status", "✅ Approved" if critique.get("approved") else "⚠️ Forced")
    if eval_score is not None:
        c4.metric("Eval harness",     f"{eval_score:.0%}")

    render_eval_score(eval_score)

    if critique.get("suggestion"):
        st.caption(f"💡 Critic suggestion: {critique['suggestion']}")

    # ── Fact-check summary ────────────────────────────────────────────────
    if critique.get("supported_claims") or critique.get("unsupported_claims"):
        with st.expander("🔍 Fact-check results", expanded=False):
            col_s, col_u = st.columns(2)
            with col_s:
                st.markdown("**✅ Supported claims**")
                for c in (critique.get("supported_claims") or []):
                    st.markdown(f"- {c}")
            with col_u:
                st.markdown("**⚠️ Unsupported / weak claims**")
                for c in (critique.get("unsupported_claims") or []):
                    st.markdown(f"- {c}")

    # ── Domain display ────────────────────────────────────────────────────
    if fsv.get("domain"):
        st.markdown(
            f'<div style="margin:8px 0;font-size:.8rem;color:#8a87a0">'
            f'Domain classified as: <span class="domain-tag">⬡ {fsv["domain"].upper()}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Report body ───────────────────────────────────────────────────────
    st.markdown("#### 📄 Research Report")
    summary = fsv.get("final_summary", "")
    st.markdown(f'<div class="report-box">{summary}</div>', unsafe_allow_html=True)

    # ── Citation map ──────────────────────────────────────────────────────
    render_citation_map(fsv.get("citation_map") or {})

    # ── Export buttons ────────────────────────────────────────────────────
    st.markdown("##### ⬇️ Export")
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        st.download_button(
            "📝 Markdown",
            data=summary,
            file_name=f"report_{time.strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with dl_col2:
        if st.button("📄 Export PDF", use_container_width=True):
            try:
                with st.spinner("Generating PDF…"):
                    path = export_pdf(summary, fsv.get("topic", topic))
                with open(path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=f.read(),
                        file_name=path.split("/")[-1],
                        mime="application/pdf",
                    )
            except RuntimeError as e:
                st.error(str(e))

    with dl_col3:
        if st.button("📘 Export DOCX", use_container_width=True):
            try:
                with st.spinner("Generating DOCX…"):
                    path = export_docx(summary, fsv.get("topic", topic))
                with open(path, "rb") as f:
                    st.download_button(
                        "⬇️ Download DOCX",
                        data=f.read(),
                        file_name=path.split("/")[-1],
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
            except RuntimeError as e:
                st.error(str(e))

    # ── FinOps breakdown ──────────────────────────────────────────────────
    if st.checkbox("📊 FinOps breakdown"):
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Input tokens",  sum(u.get("prompt_tokens", 0)     for u in ul))
        fc2.metric("Output tokens", sum(u.get("completion_tokens", 0) for u in ul))
        fc3.metric("Total tokens",  total_tokens(ul))
        fc4.metric("Est. cost",     f"${calc_cost(ul, st.session_state.sel_model):.5f}")
        st.caption(f"Model: `{st.session_state.sel_model}`")

    # ── Feedback ──────────────────────────────────────────────────────────
    render_feedback(db_conn, st.session_state.thread_id)

    # ── Save to history ───────────────────────────────────────────────────
    save_history(
        db_conn,
        st.session_state.thread_id,
        topic,
        "done",
        calc_cost(ul, st.session_state.sel_model),
        total_tokens(ul),
        summary,
        eval_score=eval_score,
    )
