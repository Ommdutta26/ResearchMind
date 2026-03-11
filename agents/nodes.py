"""
agents/nodes.py
All LangGraph node functions for the NexusResearch agent pipeline.
"""

from __future__ import annotations

import time

import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential
from langgraph.types import interrupt, Command

from core.state import AgentState
from core.models import get_llm, get_tools, get_vector_store
from core.config import DEPTH_QUERY_COUNT
from utils.retrieval import bm25_search, rerank, build_bm25_index


# ── Retry wrapper ─────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8), reraise=True)
def safe_llm(llm_obj, prompt):
    return llm_obj.invoke(prompt)


# ── Planner ───────────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    model = st.session_state.get("sel_model", "llama-3.3-70b-versatile")
    depth = st.session_state.get("depth", "deep")
    n     = DEPTH_QUERY_COUNT.get(depth, 3)
    llm   = get_llm(model)

    res = safe_llm(
        llm,
        f"Topic: {state['topic']}\n"
        f"Generate exactly {n} specific search queries. "
        f"Return ONLY the queries, one per line, no numbering.",
    )

    queries = [q.strip() for q in res.content.strip().split("\n") if q.strip()][:n]
    usage   = res.response_metadata.get("token_usage", {})

    return {
        "queries":        queries,
        "usage_metadata": [usage],
        "node_log":       [f"[PLANNER] {len(queries)} queries planned"],
        "start_time":     state.get("start_time", time.time()),
        "status":         "running",
    }


# ── Query Expander ────────────────────────────────────────────────────────────

def query_expander_node(state: AgentState) -> dict:
    llm = get_llm()

    prompt = (
        "Based on this research data generate new deeper queries.\n\n"
        f"Data:\n{state['research_data'][:2000]}"
    )

    res     = llm.invoke(prompt)
    queries = res.content.split("\n")

    return {"queries": queries[:3]}


# ── Searcher ──────────────────────────────────────────────────────────────────

def searcher_node(state: AgentState) -> dict:
    _tavily, _wiki, _arxiv = get_tools()
    vector_store           = get_vector_store()

    # Rebuild BM25 on current vector store snapshot
    bm25_index, bm25_docs = build_bm25_index()

    new_data: list[str] = []
    urls:     list[str] = []
    errors:   list[str] = []

    for i, q in enumerate(state["queries"]):

        # ── Hybrid retrieval (FAISS + BM25) ──────────────────────────────
        try:
            vec_docs      = vector_store.similarity_search(q, k=2)
            bm_docs       = bm25_search(q, bm25_index, bm25_docs)
            combined_docs = list({d.page_content: d for d in vec_docs + bm_docs}.values())
            ranked_docs   = rerank(q, combined_docs)

            for d in ranked_docs:
                new_data.append(f"[HYBRID|Q{i+1}] {q}\n{d.page_content[:600]}")

        except Exception as e:
            errors.append(f"Hybrid Q{i+1}: {str(e)[:80]}")

        # ── Web search (Tavily) ───────────────────────────────────────────
        try:
            raw = _tavily.invoke({"query": q})

            if isinstance(raw, dict):
                items = raw.get("results") or raw.get("content") or [raw]
            elif isinstance(raw, list):
                items = raw
            elif isinstance(raw, str) and raw.strip():
                new_data.append(f"[WEB|Q{i+1}] {q}\n{raw[:600]}")
                items = []
            else:
                items = []

            for r in items[:2]:
                if isinstance(r, dict):
                    snippet = (
                        r.get("content") or r.get("snippet") or r.get("text") or str(r)
                    )[:600]
                    url = r.get("url") or r.get("source") or ""
                else:
                    snippet = str(r)[:600]
                    url     = ""

                new_data.append(f"[WEB|Q{i+1}] {q}\n{snippet}")
                if url:
                    urls.append(url)

        except Exception as e:
            errors.append(f"Web Q{i+1}: {str(e)[:80]}")

        # ── Arxiv ─────────────────────────────────────────────────────────
        try:
            if _arxiv:
                arxiv_result = _arxiv.run(q)
                if arxiv_result:
                    new_data.append(f"[ARXIV|Q{i+1}] {q}\n{str(arxiv_result)[:600]}")
                    urls.append("https://arxiv.org")
        except Exception as e:
            errors.append(f"Arxiv Q{i+1}: {str(e)[:80]}")

    # ── Wikipedia (once per topic) ────────────────────────────────────────
    if _wiki:
        try:
            wr = _wiki.invoke(state["topic"])
            if wr:
                new_data.append(f"[WIKIPEDIA] {state['topic']}\n{str(wr)[:600]}")
        except Exception:
            pass

    log = f"[SEARCHER] {len(new_data)} snippets"
    if errors:
        log += f" | Errors: {'; '.join(errors[:2])}"

    return {
        "research_data": new_data,
        "source_urls":   urls,
        "node_log":      [log],
        "status":        "running" if new_data else "error",
    }


# ── Analyst (HITL gate) ───────────────────────────────────────────────────────

def analyst_node(state: AgentState) -> dict:
    context        = "\n\n".join(state.get("research_data", []))
    human_response = interrupt(
        {
            "question": "Approve data?",
            "snippet":  context[:400],
            "n_sources": len(state.get("research_data", [])),
        }
    )

    action = (
        human_response.get("action", "reject")
        if isinstance(human_response, dict)
        else "reject"
    )
    extra = (
        human_response.get("extra", "")
        if isinstance(human_response, dict)
        else ""
    )

    if action != "approve":
        return {
            "final_summary": "Rejected by user.",
            "status":        "done",
            "node_log":      ["[ANALYST] Rejected"],
        }

    model  = st.session_state.get("sel_model", "llama-3.3-70b-versatile")
    llm    = get_llm(model, temp=0.3)
    prompt = (
        f"Write a comprehensive research report on: {state['topic']}\n\n"
        f"Research data:\n{context}\n\n"
        f"{('Extra: ' + extra) if extra else ''}\n"
        "Structure: Executive Summary, Key Findings, Detailed Analysis, Conclusion."
    )

    try:
        res   = safe_llm(llm, prompt)
        usage = res.response_metadata.get("token_usage", {})
    except Exception as e:
        return {
            "final_summary": f"Error: {e}",
            "status":        "error",
            "usage_metadata": [],
            "node_log":      [f"[ANALYST] Error: {e}"],
        }

    return {
        "draft_report":   res.content,
        "usage_metadata": [usage],
        "node_log":       [f"[ANALYST] Draft ready ({len(res.content)} chars)"],
        "status":         "running",
    }


# ── Fact Checker ──────────────────────────────────────────────────────────────

def fact_checker_node(state: AgentState) -> dict:
    llm      = get_llm()
    report   = state.get("draft_report", "")
    research = "\n".join(state.get("research_data", []))[:4000]

    prompt = f"""You are a fact-checking AI.
Your job is to verify claims in the report using the research data.

Report:
{report}

Research Data:
{research}

Identify unsupported or weak claims.

Return STRICTLY in this format:

SUPPORTED:
- claim

UNSUPPORTED:
- claim

If everything is supported write:

UNSUPPORTED:
none
"""

    try:
        res   = safe_llm(llm, prompt)
        text  = res.content
        usage = res.response_metadata.get("token_usage", {})
    except Exception as e:
        return {
            "node_log":      [f"[FACT CHECKER] Error: {e}"],
            "usage_metadata": [],
        }

    supported: list[str]   = []
    unsupported: list[str] = []
    mode: str | None       = None

    for line in text.split("\n"):
        l = line.strip()
        if l.startswith("SUPPORTED"):
            mode = "supported"
            continue
        if l.startswith("UNSUPPORTED"):
            mode = "unsupported"
            continue
        if l.startswith("-"):
            claim = l[1:].strip()
            if mode == "supported":
                supported.append(claim)
            elif mode == "unsupported":
                unsupported.append(claim)

    critique = {
        **state.get("critique", {}),
        "fact_checked":        True,
        "supported_claims":    supported[:5],
        "unsupported_claims":  unsupported[:5],
    }

    return {
        "critique":       critique,
        "usage_metadata": [usage],
        "node_log":       [
            f"[FACT CHECKER] Supported:{len(supported)} Unsupported:{len(unsupported)}"
        ],
    }


# ── Contrarian ────────────────────────────────────────────────────────────────

def contrarian_node(state: AgentState) -> dict:
    model = st.session_state.get("sel_model")
    llm   = get_llm(model, temp=0.7)

    prompt = (
        f"Topic: {state['topic']}\n\n"
        "Write a strong opposing or contrarian perspective.\n"
        "Highlight risks, weaknesses, and counter-arguments."
    )

    res = safe_llm(llm, prompt)

    return {
        "node_log":    ["[CONTRARIAN] Added risk perspective"],
        "draft_report": (
            state.get("draft_report", "")
            + "\n\n## Contrarian Perspective\n"
            + res.content
        ),
    }


# ── Critic ────────────────────────────────────────────────────────────────────

def critic_node(state: AgentState) -> dict:
    model = st.session_state.get("sel_model")
    llm   = get_llm(model)

    prompt = (
        f"Rate this report 1-10.\nDraft:\n"
        f"{state.get('draft_report','')[:2000]}\n\n"
        "Reply EXACTLY:\n"
        "SCORE: <n>\n"
        "APPROVED: <yes/no>\n"
        "GAPS: <issues or none>\n"
        "SUGGESTION: <tip>"
    )

    res   = safe_llm(llm, prompt)
    text  = res.content
    usage = res.response_metadata.get("token_usage", {})

    score      = 7
    approved   = True
    gaps: list = []
    suggestion = ""

    for line in text.split("\n"):
        l = line.strip()
        if l.startswith("SCORE:"):
            try:
                score = int(l.split(":")[1].strip().split()[0])
            except Exception:
                pass
        elif l.startswith("APPROVED:"):
            approved = "yes" in l.lower()
        elif l.startswith("GAPS:"):
            g = l.split(":", 1)[1].strip()
            gaps = [] if g.lower() == "none" else [g]
        elif l.startswith("SUGGESTION:"):
            suggestion = l.split(":", 1)[1].strip()

    critique = {
        "score":      score,
        "approved":   approved,
        "gaps":       gaps,
        "suggestion": suggestion,
    }

    return {
        "critique":         critique,
        "usage_metadata":   [usage],
        "reflection_count": state.get("reflection_count", 0) + 1,
        "node_log":         [f"[CRITIC] Score:{score}/10 Approved:{approved}"],
    }


# ── Refiner ───────────────────────────────────────────────────────────────────

def refiner_node(state: AgentState) -> dict:
    model    = st.session_state.get("sel_model", "llama-3.3-70b-versatile")
    llm      = get_llm(model, temp=0.2)
    critique = state.get("critique", {})

    prompt = (
        f"Improve this report.\nOriginal:\n{state.get('draft_report','')[:2000]}\n\n"
        f"Gaps: {critique.get('gaps', [])}\n"
        f"Suggestion: {critique.get('suggestion','')}\n\n"
        "Write improved version."
    )

    try:
        res   = safe_llm(llm, prompt)
        usage = res.response_metadata.get("token_usage", {})
    except Exception as e:
        return {"node_log": [f"[REFINER] Error: {e}"], "usage_metadata": []}

    return {
        "draft_report":   res.content,
        "usage_metadata": [usage],
        "node_log":       [f"[REFINER] Refined (loop {state.get('reflection_count', 1)})"],
    }


# ── Grounding ─────────────────────────────────────────────────────────────────

def grounding_node(state: AgentState) -> dict:
    draft    = state.get("draft_report", "")
    research = " ".join(state.get("research_data", []))

    unsupported = [
        s.strip()
        for s in draft.split(".")
        if s.strip() and s not in research
    ]

    return {
        "node_log": [f"[GROUNDING] {len(unsupported)} potentially unsupported claims"],
        "critique": {
            **state.get("critique", {}),
            "unsupported_claims": unsupported[:5],
        },
    }


# ── Finalizer ─────────────────────────────────────────────────────────────────

def finalizer_node(state: AgentState) -> dict:
    draft      = state.get("draft_report", "")
    sources    = state.get("source_urls", [])
    confidence = min(1.0, len(state.get("research_data", [])) / 10)

    report = draft + f"\n\nConfidence Score: {confidence:.2f}"
    src    = (
        ("\n\n## Sources\n" + "\n".join(f"- {u}" for u in sources))
        if sources
        else ""
    )

    return {
        "final_summary": report + src,
        "status":        "done",
        "node_log":      ["[FINALIZER] Report ready"],
    }


# ── Router ────────────────────────────────────────────────────────────────────

def route_critic(state: AgentState) -> str:
    critique         = state.get("critique", {})
    score            = critique.get("score", 10)
    reflection_count = state.get("reflection_count", 0)
    max_loops        = st.session_state.get("max_loops", 1)

    if reflection_count >= max_loops:
        return "finalizer"
    if score < 7:
        return "refiner"
    return "finalizer"
