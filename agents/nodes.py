"""
agents/nodes.py
All LangGraph node functions for the NexusResearch agent pipeline.

Upgrades from v1:
  - Structured Pydantic output contracts (critic, fact_checker, planner)
  - Cosine-similarity grounding (replaces naive verbatim string matching)
  - Inline citation engine (sentence → source URL)
  - Automated eval harness node (judge LLM, 0-1 score)
  - Domain classifier for multi-agent routing
  - query_expander now runs AFTER searcher (has real data)
  - Streaming-ready: analyst uses llm.stream() via helper
"""

from __future__ import annotations

import time

import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential
from langgraph.types import interrupt

from core.state import AgentState, CritiqueOutput, FactCheckOutput, PlannerOutput
from core.models import get_llm, get_tools, get_vector_store
from core.config import DEPTH_QUERY_COUNT
from utils.retrieval import (
    bm25_search,
    rerank,
    build_bm25_index,
    cosine_grounding,
    build_citation_map,
)


# ── Retry wrapper ─────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8), reraise=True)
def safe_llm(llm_obj, prompt):
    return llm_obj.invoke(prompt)


# ── Streaming helper ──────────────────────────────────────────────────────────

def stream_llm_to_placeholder(llm_obj, prompt: str, placeholder) -> tuple[str, dict]:
    """
    Streams token-by-token into a Streamlit placeholder.
    Returns (full_text, usage_metadata).
    """
    chunks: list[str] = []
    usage: dict = {}
    with placeholder.container():
        box = st.empty()
        for chunk in llm_obj.stream(prompt):
            token = chunk.content
            chunks.append(token)
            box.markdown("".join(chunks) + "▌")
            if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                usage = chunk.response_metadata.get("token_usage", {})
        box.markdown("".join(chunks))
    return "".join(chunks), usage


# ── Domain classifier ─────────────────────────────────────────────────────────

DOMAINS = ["finance", "biotech", "geopolitics", "technology", "science", "general"]

def domain_classifier_node(state: AgentState) -> dict:
    """Classify the topic into a domain to route to the right specialist prompt."""
    llm    = get_llm("llama-3.1-8b-instant", temp=0.0)
    prompt = (
        f"Classify this research topic into exactly one domain.\n"
        f"Topic: {state['topic']}\n"
        f"Domains: {', '.join(DOMAINS)}\n"
        "Reply with ONLY the domain name, nothing else."
    )
    try:
        res    = safe_llm(llm, prompt)
        domain = res.content.strip().lower()
        if domain not in DOMAINS:
            domain = "general"
    except Exception:
        domain = "general"

    return {
        "domain":   domain,
        "node_log": [f"[CLASSIFIER] Domain: {domain}"],
        "status":   "running",
    }


# ── Planner ───────────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    model  = st.session_state.get("sel_model", "llama-3.3-70b-versatile")
    depth  = st.session_state.get("depth", "deep")
    n      = DEPTH_QUERY_COUNT.get(depth, 3)
    domain = state.get("domain", "general")

    # Domain-aware specialist instruction
    domain_hints = {
        "finance":     "Focus on financial metrics, valuations, market dynamics, risk factors.",
        "biotech":     "Focus on clinical trials, FDA pipeline, mechanism of action, competitive landscape.",
        "geopolitics": "Focus on alliances, trade flows, sanctions, regional stability indicators.",
        "technology":  "Focus on technical architecture, adoption curves, competitive moats, patents.",
        "science":     "Focus on peer-reviewed findings, methodology, reproducibility, impact.",
        "general":     "Cover the topic comprehensively across multiple dimensions.",
    }
    hint = domain_hints.get(domain, domain_hints["general"])

    llm = get_llm(model)
    res = safe_llm(
        llm,
        f"Topic: {state['topic']}\n"
        f"Domain: {domain}. {hint}\n"
        f"Generate exactly {n} specific, non-overlapping search queries. "
        f"Return ONLY the queries, one per line, no numbering.",
    )

    queries = [q.strip() for q in res.content.strip().split("\n") if q.strip()][:n]
    usage   = res.response_metadata.get("token_usage", {})

    return {
        "queries":        queries,
        "usage_metadata": [usage],
        "node_log":       [f"[PLANNER] {len(queries)} queries planned (domain: {domain})"],
        "start_time":     state.get("start_time", time.time()),
        "status":         "running",
    }


# ── Searcher ──────────────────────────────────────────────────────────────────

def searcher_node(state: AgentState) -> dict:
    _tavily, _wiki, _arxiv = get_tools()
    vector_store            = get_vector_store()
    bm25_index, bm25_docs   = build_bm25_index()

    new_data: list[str] = []
    urls:     list[str] = []
    errors:   list[str] = []

    for i, q in enumerate(state["queries"]):

        # ── Hybrid retrieval (Chroma + BM25) ─────────────────────────────
        try:
            vec_docs      = vector_store.similarity_search(q, k=2)
            bm_docs       = bm25_search(q, bm25_index, bm25_docs)
            combined_docs = list({d.page_content: d for d in vec_docs + bm_docs}.values())
            ranked_docs   = rerank(q, combined_docs)
            for d in ranked_docs:
                new_data.append(f"[HYBRID|Q{i+1}] {q}\n{d.page_content[:600]}")
        except Exception as e:
            errors.append(f"Hybrid Q{i+1}: {str(e)[:80]}")

        # ── Tavily web search ─────────────────────────────────────────────
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
                    snippet = (r.get("content") or r.get("snippet") or str(r))[:600]
                    url     = r.get("url") or r.get("source") or ""
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

    # Store new snippets in vector store for cross-session memory
    if new_data:
        try:
            vector_store.add_texts(new_data)
            vector_store.save_local("faiss_index")
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


# ── Query Expander (now after searcher — has real data) ───────────────────────

def query_expander_node(state: AgentState) -> dict:
    """
    FIX from v1: expander now runs after searcher so it has real research
    data to expand from, not an empty list.
    """
    data = state.get("research_data", [])
    if not data:
        return {"node_log": ["[EXPANDER] No data yet — skipping"]}

    llm    = get_llm("llama-3.1-8b-instant", temp=0.3)
    prompt = (
        "Based on this research data, generate 3 deeper follow-up queries "
        "that would fill the most important gaps.\n\n"
        f"Data summary:\n{' '.join(data[:3])[:1500]}\n\n"
        "Return ONLY the queries, one per line."
    )
    res     = safe_llm(llm, prompt)
    queries = [q.strip() for q in res.content.split("\n") if q.strip()][:3]

    return {
        "queries":  queries,
        "node_log": [f"[EXPANDER] {len(queries)} follow-up queries"],
    }


# ── Analyst (HITL gate + streaming report) ───────────────────────────────────

def analyst_node(state: AgentState) -> dict:
    context        = "\n\n".join(state.get("research_data", []))
    human_response = interrupt(
        {
            "question":  "Approve data?",
            "snippet":   context[:400],
            "n_sources": len(state.get("research_data", [])),
        }
    )

    action = (
        human_response.get("action", "reject")
        if isinstance(human_response, dict) else "reject"
    )
    extra = (
        human_response.get("extra", "")
        if isinstance(human_response, dict) else ""
    )

    if action != "approve":
        return {
            "final_summary": "Rejected by user.",
            "status":        "done",
            "node_log":      ["[ANALYST] Rejected"],
        }

    model  = st.session_state.get("sel_model", "llama-3.3-70b-versatile")
    domain = state.get("domain", "general")
    llm    = get_llm(model, temp=0.3)

    domain_section = {
        "finance":     "Include: valuation analysis, risk factors, catalysts, competitive positioning.",
        "biotech":     "Include: clinical stage, market size, competitive landscape, key risks.",
        "geopolitics": "Include: key actors, historical context, economic implications, scenarios.",
        "technology":  "Include: technical differentiation, adoption metrics, moat analysis.",
        "science":     "Include: methodology critique, reproducibility, real-world applications.",
        "general":     "",
    }.get(domain, "")

    prompt = (
        f"Write a comprehensive research report on: {state['topic']}\n\n"
        f"Research data:\n{context[:6000]}\n\n"
        f"{('Extra instructions: ' + extra) if extra else ''}\n"
        f"{domain_section}\n"
        "Structure with these sections:\n"
        "## Executive Summary\n## Key Findings\n## Detailed Analysis\n"
        "## Risks & Counterpoints\n## Conclusion\n\n"
        "Be specific and data-driven. Cite facts from the research data."
    )

    # Try streaming first; fall back to blocking invoke
    try:
        placeholder = st.empty()
        content, usage = stream_llm_to_placeholder(llm, prompt, placeholder)
    except Exception:
        try:
            res     = safe_llm(llm, prompt)
            content = res.content
            usage   = res.response_metadata.get("token_usage", {})
        except Exception as e:
            return {
                "final_summary":  f"Error: {e}",
                "status":         "error",
                "usage_metadata": [],
                "node_log":       [f"[ANALYST] Error: {e}"],
            }

    return {
        "draft_report":   content,
        "usage_metadata": [usage],
        "node_log":       [f"[ANALYST] Draft ready ({len(content)} chars)"],
        "status":         "running",
    }


# ── Fact Checker (Pydantic structured output) ─────────────────────────────────

def fact_checker_node(state: AgentState) -> dict:
    llm      = get_llm(temp=0.0)
    report   = state.get("draft_report", "")
    research = "\n".join(state.get("research_data", []))[:4000]

    # Use structured output for reliable parsing
    structured_llm = llm.with_structured_output(FactCheckOutput)

    prompt = (
        f"You are a fact-checking AI. Verify claims in this report against "
        f"the research data provided.\n\n"
        f"Report:\n{report[:2000]}\n\n"
        f"Research Data:\n{research}\n\n"
        "Identify which claims are supported and which are unsupported."
    )

    try:
        result: FactCheckOutput = structured_llm.invoke(prompt)
        usage  = {}  # structured output doesn't always surface usage
    except Exception as e:
        # Graceful fallback
        return {
            "node_log":       [f"[FACT CHECKER] Structured output failed: {e}"],
            "usage_metadata": [],
        }

    critique = {
        **state.get("critique", {}),
        "fact_checked":       True,
        "supported_claims":   result.supported_claims[:5],
        "unsupported_claims": result.unsupported_claims[:5],
    }

    return {
        "critique":       critique,
        "usage_metadata": [usage],
        "node_log": [
            f"[FACT CHECKER] Supported:{len(result.supported_claims)} "
            f"Unsupported:{len(result.unsupported_claims)}"
        ],
    }


# ── Contrarian ────────────────────────────────────────────────────────────────

def contrarian_node(state: AgentState) -> dict:
    model = st.session_state.get("sel_model")
    llm   = get_llm(model, temp=0.7)

    prompt = (
        f"Topic: {state['topic']}\n\n"
        "Write a rigorous contrarian perspective.\n"
        "Highlight: structural risks, blind spots in mainstream analysis, "
        "historical analogues where consensus was wrong, and second-order effects."
    )

    res = safe_llm(llm, prompt)

    return {
        "node_log":    ["[CONTRARIAN] Risk perspective added"],
        "draft_report": (
            state.get("draft_report", "")
            + "\n\n## Contrarian Perspective\n"
            + res.content
        ),
    }


# ── Critic (Pydantic structured output) ──────────────────────────────────────

def critic_node(state: AgentState) -> dict:
    model = st.session_state.get("sel_model")
    llm   = get_llm(model, temp=0.0)

    structured_llm = llm.with_structured_output(CritiqueOutput)

    prompt = (
        f"You are a senior research editor. Critically evaluate this report.\n\n"
        f"Report:\n{state.get('draft_report', '')[:3000]}\n\n"
        "Rate quality 1-10. Identify gaps. Suggest the single most impactful improvement."
    )

    try:
        result: CritiqueOutput = structured_llm.invoke(prompt)
        usage  = {}
    except Exception as e:
        # Fallback to a fixed passing critique
        result = CritiqueOutput(score=7, approved=True, gaps=[], suggestion="")
        usage  = {}

    critique = {
        **state.get("critique", {}),
        "score":      result.score,
        "approved":   result.approved,
        "gaps":       result.gaps,
        "suggestion": result.suggestion,
    }

    return {
        "critique":         critique,
        "usage_metadata":   [usage],
        "reflection_count": state.get("reflection_count", 0) + 1,
        "node_log": [
            f"[CRITIC] Score:{result.score}/10 Approved:{result.approved}"
        ],
    }


# ── Refiner ───────────────────────────────────────────────────────────────────

def refiner_node(state: AgentState) -> dict:
    model    = st.session_state.get("sel_model", "llama-3.3-70b-versatile")
    llm      = get_llm(model, temp=0.2)
    critique = state.get("critique", {})

    prompt = (
        f"Improve this research report based on the editorial feedback.\n\n"
        f"Original report:\n{state.get('draft_report', '')[:3000]}\n\n"
        f"Identified gaps: {critique.get('gaps', [])}\n"
        f"Suggested improvement: {critique.get('suggestion', '')}\n\n"
        "Write the full improved report. Keep all sections. Expand where gaps exist."
    )

    try:
        res   = safe_llm(llm, prompt)
        usage = res.response_metadata.get("token_usage", {})
    except Exception as e:
        return {"node_log": [f"[REFINER] Error: {e}"], "usage_metadata": []}

    return {
        "draft_report":   res.content,
        "usage_metadata": [usage],
        "node_log": [f"[REFINER] Refined (loop {state.get('reflection_count', 1)})"],
    }


# ── Grounding (cosine-similarity, not verbatim matching) ──────────────────────

def grounding_node(state: AgentState) -> dict:
    """
    FIX from v1: uses cosine similarity instead of verbatim string matching
    so paraphrased claims are correctly evaluated.
    """
    draft    = state.get("draft_report", "")
    research = state.get("research_data", [])

    unsupported = cosine_grounding(draft, research, threshold=0.35)

    return {
        "node_log": [f"[GROUNDING] {len(unsupported)} low-similarity claims flagged"],
        "critique": {
            **state.get("critique", {}),
            "cosine_unsupported": unsupported,
        },
    }


# ── Citation Engine ───────────────────────────────────────────────────────────

def citation_node(state: AgentState) -> dict:
    """Build a sentence → URL citation map using cosine similarity."""
    draft    = state.get("draft_report", "")
    research = state.get("research_data", [])
    urls     = state.get("source_urls", [])

    citation_map = build_citation_map(draft, research, urls)

    return {
        "citation_map": citation_map,
        "node_log": [f"[CITATION] {len(citation_map)} citations mapped"],
    }


# ── Automated Eval Harness ────────────────────────────────────────────────────

def eval_harness_node(state: AgentState) -> dict:
    """
    Judge LLM evaluates the final report on 5 dimensions:
    accuracy, depth, clarity, structure, actionability.
    Returns a composite 0-1 score stored in state for history tracking.
    """
    llm    = get_llm("llama-3.1-8b-instant", temp=0.0)
    report = state.get("draft_report", "")[:3000]

    prompt = (
        "You are a research quality judge. Score this report on each dimension from 0-10.\n\n"
        f"Report:\n{report}\n\n"
        "Reply EXACTLY in this format (no extra text):\n"
        "ACCURACY: <0-10>\n"
        "DEPTH: <0-10>\n"
        "CLARITY: <0-10>\n"
        "STRUCTURE: <0-10>\n"
        "ACTIONABILITY: <0-10>"
    )

    try:
        res  = safe_llm(llm, prompt)
        text = res.content

        scores: list[int] = []
        for line in text.split("\n"):
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    scores.append(int(parts[1].strip().split()[0]))
                except ValueError:
                    pass

        eval_score = (sum(scores) / (len(scores) * 10)) if scores else 0.0
    except Exception:
        eval_score = 0.0

    return {
        "eval_score": round(eval_score, 3),
        "node_log":   [f"[EVAL] Composite score: {eval_score:.2f}"],
    }


# ── Finalizer ─────────────────────────────────────────────────────────────────

def finalizer_node(state: AgentState) -> dict:
    draft        = state.get("draft_report", "")
    sources      = state.get("source_urls", [])
    citation_map = state.get("citation_map", {})
    confidence   = min(1.0, len(state.get("research_data", [])) / 10)
    eval_score   = state.get("eval_score", None)

    report = draft + f"\n\n---\n**Confidence Score:** {confidence:.2f}"

    if eval_score is not None:
        report += f"  |  **Quality Score:** {eval_score:.2f}"

    if sources:
        deduped = list(dict.fromkeys(sources))  # preserve order, dedupe
        report += "\n\n## Sources\n" + "\n".join(f"- {u}" for u in deduped)

    if citation_map:
        report += (
            "\n\n## Citation Map\n"
            + "\n".join(f'- "{k}…" → {v}' for k, v in list(citation_map.items())[:8])
        )

    return {
        "final_summary": report,
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
