"""
ui/styles.py
Injects the custom CSS theme into the Streamlit page.
"""

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono&display=swap');

:root{
  --bg:#0a0c10;--surf:#111418;--border:#1e2530;
  --acc:#00e5ff;--acc2:#ff6b35;--acc3:#7fff6b;
  --txt:#e8eaf0;--muted:#5a6478;
  --warn:#ffab40;--ok:#00e676;--err:#ff5252
}

html,body,[class*="css"]{
  background:var(--bg)!important;
  color:var(--txt)!important;
  font-family:'DM Mono',monospace
}

.stButton>button{
  background:transparent!important;
  border:1px solid var(--acc)!important;
  color:var(--acc)!important;
  font-family:'Syne',sans-serif!important;
  font-weight:700!important;
  border-radius:6px!important;
  transition:all .2s!important
}
.stButton>button:hover{background:var(--acc)!important;color:var(--bg)!important}

.stTextInput>div>input,
.stTextArea>div>textarea{
  background:var(--surf)!important;
  border:1px solid var(--border)!important;
  color:var(--txt)!important;
  border-radius:6px!important
}

[data-testid="stSidebar"]{
  background:var(--surf)!important;
  border-right:1px solid var(--border)!important
}

.node-card{
  background:var(--surf);border:1px solid var(--border);
  border-left:3px solid var(--acc);border-radius:8px;
  padding:12px 16px;margin:6px 0
}

.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}
.metric-tile{
  flex:1;min-width:110px;background:var(--surf);
  border:1px solid var(--border);border-radius:8px;
  padding:12px;text-align:center
}
.metric-tile .val{
  font-family:'Syne',sans-serif;font-size:1.4rem;
  font-weight:800;color:var(--acc)
}
.metric-tile .lbl{
  font-size:.65rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:1px;margin-top:2px
}

.report-box{
  background:var(--surf);border:1px solid var(--border);
  border-radius:10px;padding:28px 32px;line-height:1.8
}

.hitl-box{
  background:#ffab4014;border:1px solid var(--warn);
  border-radius:8px;padding:18px;margin:12px 0
}

.badge{
  display:inline-block;padding:3px 10px;border-radius:20px;
  font-size:.65rem;font-weight:700;letter-spacing:1px;text-transform:uppercase
}
.badge-run {background:#00e5ff22;color:var(--acc); border:1px solid var(--acc)}
.badge-wait{background:#ffab4022;color:var(--warn);border:1px solid var(--warn)}
.badge-done{background:#00e67622;color:var(--ok); border:1px solid var(--ok)}
.badge-err {background:#ff525222;color:var(--err); border:1px solid var(--err)}

.prog-wrap{background:var(--border);border-radius:4px;height:4px;margin:6px 0;overflow:hidden}
.prog-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--acc),var(--acc2));transition:width .4s}

#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:1rem!important}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
