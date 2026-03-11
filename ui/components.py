"""
ui/components.py
Reusable Streamlit rendering helpers: node cards, metrics bar, logs, report.
"""

from __future__ import annotations

import time

import streamlit as st

from utils.finops import total_tokens, calc_cost

# ── Colour map for node labels ────────────────────────────────────────────────
NODE_COLORS: dict[str, str] = {
    "PLANNER":    "#00e5ff",
    "SEARCHER":   "#ff6b35",
    "ANALYST":    "#7fff6b",
    "CRITIC":     "#ffab40",
    "REFINER":    "#c678dd",
    "FINALIZER":  "#00e676",
}

# ── Badge config ──────────────────────────────────────────────────────────────
BADGES: dict[str, tuple[str, str]] = {
    "idle":    ("IDLE",    "badge-done"),
    "running": ("RUNNING", "badge-run"),
    "paused":  ("WAITING", "badge-wait"),
    "done":    ("DONE",    "badge-done"),
    "error":   ("ERROR",   "badge-err"),
}


# ── Individual components ─────────────────────────────────────────────────────

def node_card(title: str, body: str, color: str = "#00e5ff") -> None:
    st.markdown(
        f'<div class="node-card" style="border-left-color:{color}">'
        f'<b style="font-family:Syne,sans-serif;font-size:.75rem;letter-spacing:1px">{title}</b>'
        f'<div style="margin-top:5px;font-size:.82rem">{body}</div></div>',
        unsafe_allow_html=True,
    )


def render_log(logs: list[str]) -> None:
    for log in logs[-10:]:
        node = log.split("]")[0].strip("[")
        body = log[len(node) + 2 :].strip()
        node_card(node, body, NODE_COLORS.get(node, "#00e5ff"))


def render_metrics(ul: list[dict], model: str, t0: float) -> None:
    tok  = total_tokens(ul)
    cost = calc_cost(ul, model)
    secs = time.time() - t0 if t0 else 0
    pct  = min(100, int(tok / max(st.session_state.budget_k * 1000, 1) * 100))

    st.markdown(
        f"""<div class="metric-row">
          <div class="metric-tile"><div class="val">{tok:,}</div><div class="lbl">Tokens</div></div>
          <div class="metric-tile"><div class="val">${cost:.4f}</div><div class="lbl">Cost</div></div>
          <div class="metric-tile"><div class="val">{secs:.1f}s</div><div class="lbl">Elapsed</div></div>
          <div class="metric-tile"><div class="val">{pct}%</div><div class="lbl">Budget</div></div>
        </div>
        <div class="prog-wrap"><div class="prog-fill" style="width:{pct}%"></div></div>""",
        unsafe_allow_html=True,
    )


def render_badge(phase: str) -> None:
    lbl, cls = BADGES.get(phase, ("IDLE", "badge-done"))
    st.markdown(f'<br><span class="badge {cls}">{lbl}</span>', unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """<div style="padding:20px 0 10px;border-bottom:1px solid #1e2530;margin-bottom:20px">
          <span style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800">
            🔬 Nexus<span style="color:#00e5ff">Research</span></span>
          <div style="color:#5a6478;font-size:.7rem;letter-spacing:2px;text-transform:uppercase;margin-top:2px">
            Multi-Agent · Reflection · HITL · FinOps</div>
        </div>""",
        unsafe_allow_html=True,
    )
