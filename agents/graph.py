"""
agents/graph.py
Builds and compiles the LangGraph StateGraph for NexusResearch v2.

New nodes vs v1:
  - domain_classifier  — runs first, routes planner to right specialist prompt
  - query_expander     — moved AFTER searcher (has real data now)
  - citation           — cosine-sim sentence→URL mapping
  - eval_harness       — judge LLM quality scoring before finalize
"""

import streamlit as st
from langgraph.graph import StateGraph, END

from core.state import AgentState
from agents.nodes import (
    domain_classifier_node,
    planner_node,
    searcher_node,
    query_expander_node,
    analyst_node,
    fact_checker_node,
    contrarian_node,
    critic_node,
    refiner_node,
    grounding_node,
    citation_node,
    eval_harness_node,
    finalizer_node,
    route_critic,
)


@st.cache_resource
def build_graph(_mem):
    b = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    b.add_node("classifier",     domain_classifier_node)
    b.add_node("planner",        planner_node)
    b.add_node("searcher",       searcher_node)
    b.add_node("query_expander", query_expander_node)   # after searcher now
    b.add_node("analyst",        analyst_node)
    b.add_node("fact_checker",   fact_checker_node)
    b.add_node("contrarian",     contrarian_node)
    b.add_node("critic",         critic_node)
    b.add_node("refiner",        refiner_node)
    b.add_node("grounding",      grounding_node)
    b.add_node("citation",       citation_node)
    b.add_node("eval_harness",   eval_harness_node)
    b.add_node("finalizer",      finalizer_node)

    # ── Entry point ───────────────────────────────────────────────────────
    b.set_entry_point("classifier")

    # ── Static edges ──────────────────────────────────────────────────────
    b.add_edge("classifier",     "planner")
    b.add_edge("planner",        "searcher")
    b.add_edge("searcher",       "query_expander")  # FIX: expander after searcher
    b.add_edge("query_expander", "analyst")         # HITL interrupt here
    b.add_edge("analyst",        "fact_checker")
    b.add_edge("fact_checker",   "contrarian")
    b.add_edge("contrarian",     "critic")

    # ── Conditional: critic → refiner loop or grounding ───────────────────
    b.add_conditional_edges(
        "critic",
        route_critic,
        {
            "refiner":   "refiner",
            "finalizer": "grounding",
        },
    )

    b.add_edge("refiner",      "critic")
    b.add_edge("grounding",    "citation")
    b.add_edge("citation",     "eval_harness")
    b.add_edge("eval_harness", "finalizer")
    b.add_edge("finalizer",    END)

    return b.compile(checkpointer=_mem)
