"""
agents/graph.py
Builds and compiles the LangGraph StateGraph for NexusResearch.
"""

import streamlit as st
from langgraph.graph import StateGraph, END

from core.state import AgentState
from agents.nodes import (
    planner_node,
    query_expander_node,
    searcher_node,
    analyst_node,
    fact_checker_node,
    contrarian_node,
    critic_node,
    refiner_node,
    grounding_node,
    finalizer_node,
    route_critic,
)


@st.cache_resource
def build_graph(_mem):
    """
    Compile the agent graph.  `_mem` is passed as an argument (not used
    directly inside the function body) so that Streamlit's cache can
    distinguish between different checkpointer instances.
    """
    b = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────
    b.add_node("planner",        planner_node)
    b.add_node("searcher",       searcher_node)
    b.add_node("query_expander", query_expander_node)
    b.add_node("fact_checker",   fact_checker_node)
    b.add_node("analyst",        analyst_node)
    b.add_node("contrarian",     contrarian_node)
    b.add_node("critic",         critic_node)
    b.add_node("refiner",        refiner_node)
    b.add_node("grounding",      grounding_node)
    b.add_node("finalizer",      finalizer_node)

    # ── Entry point ───────────────────────────────────────────────────────
    b.set_entry_point("planner")

    # ── Static edges ──────────────────────────────────────────────────────
    b.add_edge("planner",        "searcher")
    b.add_edge("searcher",       "query_expander")
    b.add_edge("query_expander", "analyst")
    b.add_edge("analyst",        "fact_checker")
    b.add_edge("fact_checker",   "contrarian")
    b.add_edge("contrarian",     "critic")

    # ── Conditional edge: critic → refiner or grounding ───────────────────
    b.add_conditional_edges(
        "critic",
        route_critic,
        {
            "refiner":   "refiner",
            "finalizer": "grounding",
        },
    )

    b.add_edge("refiner",   "critic")
    b.add_edge("grounding", "finalizer")
    b.add_edge("finalizer", END)

    return b.compile(checkpointer=_mem)
