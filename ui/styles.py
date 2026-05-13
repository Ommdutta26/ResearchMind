"""
ui/styles.py
Global CSS injection for NexusResearch Streamlit app.
"""

import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&display=swap');

        /* ── Root palette ───────────────────────────────────────────────── */
        :root {
            --nexus-bg:      #0f0f1a;
            --nexus-card:    #1a1a2e;
            --nexus-border:  #2a2a4a;
            --nexus-accent:  #7c6fe0;
            --nexus-gold:    #ffab40;
            --nexus-green:   #4caf82;
            --nexus-red:     #f25c5c;
            --nexus-text:    #e8e6f0;
            --nexus-muted:   #8a87a0;
        }

        /* ── Base ───────────────────────────────────────────────────────── */
        .stApp { background: var(--nexus-bg); }

        /* ── HITL approval box ──────────────────────────────────────────── */
        .hitl-box {
            background:    linear-gradient(135deg, #1e1a3a 0%, #2a1f4a 100%);
            border:        1px solid var(--nexus-accent);
            border-radius: 12px;
            padding:       20px 24px;
            margin:        16px 0;
            color:         var(--nexus-text);
            font-size:     0.9rem;
            box-shadow:    0 0 24px rgba(124,111,224,0.15);
        }

        /* ── Report display box ─────────────────────────────────────────── */
        .report-box {
            background:    var(--nexus-card);
            border:        1px solid var(--nexus-border);
            border-left:   4px solid var(--nexus-accent);
            border-radius: 10px;
            padding:       28px 32px;
            color:         var(--nexus-text);
            font-size:     0.95rem;
            line-height:   1.75;
            white-space:   pre-wrap;
        }

        /* ── Execution log ──────────────────────────────────────────────── */
        .log-entry {
            font-family:   monospace;
            font-size:     0.8rem;
            color:         var(--nexus-muted);
            padding:       2px 0;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }
        .log-entry.error { color: var(--nexus-red); }

        /* ── Status badge ───────────────────────────────────────────────── */
        .badge {
            display:       inline-block;
            padding:       4px 14px;
            border-radius: 20px;
            font-size:     0.75rem;
            font-family:   'Syne', sans-serif;
            font-weight:   600;
            letter-spacing:1px;
            text-transform:uppercase;
        }
        .badge-idle    { background:#1e1e2e; color:var(--nexus-muted); border:1px solid var(--nexus-border); }
        .badge-running { background:#1a2e1a; color:var(--nexus-green); border:1px solid var(--nexus-green); }
        .badge-paused  { background:#2e2a1a; color:var(--nexus-gold);  border:1px solid var(--nexus-gold);  }
        .badge-done    { background:#1a2e2a; color:#4dd0a0;            border:1px solid #4dd0a0;            }
        .badge-error   { background:#2e1a1a; color:var(--nexus-red);   border:1px solid var(--nexus-red);   }

        /* ── Eval bar ───────────────────────────────────────────────────── */
        .eval-bar-wrap { background:#1a1a2e; border-radius:8px; height:10px; overflow:hidden; margin-top:4px; }
        .eval-bar-fill { height:100%; border-radius:8px; transition:width 0.8s ease; }

        /* ── Citation pill ──────────────────────────────────────────────── */
        .cite-pill {
            display:       inline-block;
            background:    rgba(124,111,224,0.15);
            border:        1px solid rgba(124,111,224,0.4);
            border-radius: 4px;
            padding:       1px 6px;
            font-size:     0.7rem;
            color:         var(--nexus-accent);
            margin-left:   3px;
            cursor:        pointer;
        }

        /* ── Domain badge ───────────────────────────────────────────────── */
        .domain-tag {
            font-size:     0.7rem;
            padding:       2px 10px;
            border-radius: 12px;
            background:    rgba(255,171,64,0.12);
            color:         var(--nexus-gold);
            border:        1px solid rgba(255,171,64,0.3);
            margin-left:   8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
