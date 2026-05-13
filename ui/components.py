"""
ui/components.py
Reusable Streamlit UI components for NexusResearch v2.
"""

from __future__ import annotations

import time
import streamlit as st

from utils.finops import total_tokens, calc_cost


# ── Header ────────────────────────────────────────────────────────────────────

def render_header(domain: str | None = None) -> None:
    domain_tag = (
        f'<span class="domain-tag">⬡ {domain.upper()}</span>'
        if domain else ""
    )
    st.markdown(
        f"""
        <div style="padding:24px 0 8px">
          <span style="font-family:Syne,sans-serif;font-size:1.9rem;
                       font-weight:700;color:#e8e6f0;letter-spacing:-1px">
            🔬 NexusResearch
          </span>{domain_tag}
          <div style="font-size:.8rem;color:#8a87a0;margin-top:2px;letter-spacing:.5px">
            AI-powered multi-agent research pipeline
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Status badge ──────────────────────────────────────────────────────────────

def render_badge(phase: str) -> None:
    labels = {
        "idle":    "⬤ Idle",
        "running": "⬤ Running",
        "paused":  "⬤ Awaiting",
        "done":    "⬤ Done",
        "error":   "⬤ Error",
    }
    st.markdown(
        f'<div style="padding-top:8px">'
        f'<span class="badge badge-{phase}">{labels.get(phase, phase)}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Node progress bar ─────────────────────────────────────────────────────────

NODE_ORDER = [
    "CLASSIFIER", "PLANNER", "SEARCHER", "EXPANDER",
    "ANALYST", "FACT CHECKER", "CONTRARIAN", "CRITIC",
    "REFINER", "GROUNDING", "CITATION", "EVAL", "FINALIZER",
]

def render_progress(node_logs: list[str]) -> None:
    """Show which pipeline stage we're at with a progress indicator."""
    completed = set()
    for log in node_logs:
        for node in NODE_ORDER:
            if f"[{node}]" in log.upper():
                completed.add(node)

    pct = len(completed) / len(NODE_ORDER)
    st.markdown(
        f"""
        <div style="margin:8px 0 4px;font-size:.75rem;color:#8a87a0">
            Pipeline progress — {int(pct*100)}%
        </div>
        <div class="eval-bar-wrap">
          <div class="eval-bar-fill"
               style="width:{int(pct*100)}%;
                      background:linear-gradient(90deg,#7c6fe0,#4caf82)">
          </div>
        </div>
        <div style="font-size:.7rem;color:#8a87a0;margin-top:3px">
          {' → '.join(
              f'<b style="color:#7c6fe0">{n}</b>' if n in completed else n
              for n in NODE_ORDER
          )}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Execution log ─────────────────────────────────────────────────────────────

def render_log(logs: list[str]) -> None:
    for entry in logs:
        css = "log-entry error" if "error" in entry.lower() else "log-entry"
        st.markdown(f'<div class="{css}">{entry}</div>', unsafe_allow_html=True)


# ── Token / cost metrics ──────────────────────────────────────────────────────

def render_metrics(ul: list[dict], model: str, start: float) -> None:
    elapsed = time.time() - start
    cols    = st.columns(4)
    cols[0].metric("Total Tokens",  total_tokens(ul))
    cols[1].metric("Est. Cost",     f"${calc_cost(ul, model):.5f}")
    cols[2].metric("Elapsed",       f"{elapsed:.1f}s")
    cols[3].metric("Model",         model.split("-")[0].upper())


# ── Eval score display ────────────────────────────────────────────────────────

def render_eval_score(score: float | None) -> None:
    if score is None:
        return
    pct   = int(score * 100)
    color = "#4caf82" if score >= 0.7 else "#ffab40" if score >= 0.5 else "#f25c5c"
    label = "Excellent" if score >= 0.8 else "Good" if score >= 0.6 else "Fair" if score >= 0.4 else "Needs Work"
    st.markdown(
        f"""
        <div style="margin:8px 0">
          <div style="font-size:.75rem;color:#8a87a0;margin-bottom:3px">
            AI Quality Evaluation — {pct}% <em>({label})</em>
          </div>
          <div class="eval-bar-wrap">
            <div class="eval-bar-fill"
                 style="width:{pct}%;background:{color}">
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Citation viewer ───────────────────────────────────────────────────────────

def render_citation_map(citation_map: dict) -> None:
    if not citation_map:
        return
    with st.expander("🔗 Citation Map", expanded=False):
        for fragment, url in list(citation_map.items())[:10]:
            st.markdown(
                f'<div style="margin:4px 0;font-size:.82rem">'
                f'<span style="color:#c0bdd8">"{fragment}…"</span> '
                f'→ <a href="{url}" target="_blank" style="color:#7c6fe0">{url[:60]}</a>'
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Feedback widget ───────────────────────────────────────────────────────────

def render_feedback(conn, thread_id: str) -> None:
    """Thumbs up / down feedback stored in SQLite."""
    from core.database import save_feedback

    st.markdown("---")
    st.markdown(
        '<div style="font-size:.8rem;color:#8a87a0;margin-bottom:6px">'
        "Rate this report</div>",
        unsafe_allow_html=True,
    )
    col_up, col_dn, col_note, col_send = st.columns([1, 1, 5, 1])

    rating  = st.session_state.get(f"rating_{thread_id}", 0)

    with col_up:
        if st.button("👍", key=f"up_{thread_id}"):
            st.session_state[f"rating_{thread_id}"] = 1
    with col_dn:
        if st.button("👎", key=f"dn_{thread_id}"):
            st.session_state[f"rating_{thread_id}"] = -1
    with col_note:
        comment = st.text_input(
            "Comment (optional)", key=f"cmt_{thread_id}", label_visibility="collapsed",
            placeholder="Any comments on the report quality?",
        )
    with col_send:
        if st.button("Send", key=f"fb_{thread_id}"):
            current = st.session_state.get(f"rating_{thread_id}", 0)
            if current != 0:
                save_feedback(conn, thread_id, current, comment)
                st.success("Thanks!")
            else:
                st.warning("Select 👍 or 👎 first.")
