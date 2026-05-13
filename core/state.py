"""
core/state.py
AgentState TypedDict + Pydantic output contracts for each node.
"""

from __future__ import annotations

import operator
from typing import TypedDict, Annotated, List, Dict, Optional
from pydantic import BaseModel, Field


# ── Pydantic node output contracts ────────────────────────────────────────────

class PlannerOutput(BaseModel):
    queries: List[str] = Field(description="List of specific search queries")


class CritiqueOutput(BaseModel):
    score:      int  = Field(ge=1, le=10, description="Quality score 1-10")
    approved:   bool = Field(description="Whether the report meets quality bar")
    gaps:       List[str] = Field(default_factory=list, description="Identified gaps")
    suggestion: str  = Field(default="", description="Top improvement suggestion")


class FactCheckOutput(BaseModel):
    supported_claims:   List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)


class CitationMap(BaseModel):
    """Maps sentence fragments to source URLs."""
    citations: Dict[str, str] = Field(
        default_factory=dict,
        description="sentence_fragment -> source_url",
    )


# ── Shared agent state ─────────────────────────────────────────────────────────

class AgentState(TypedDict):
    topic:            str
    queries:          List[str]
    research_data:    Annotated[List[str], operator.add]
    final_summary:    str
    usage_metadata:   Annotated[List[Dict], operator.add]
    source_urls:      Annotated[List[str], operator.add]
    draft_report:     str
    critique:         dict
    reflection_count: int
    node_log:         Annotated[List[str], operator.add]
    start_time:       float
    status:           str
    citation_map:     Optional[dict]   # sentence_fragment -> url
    eval_score:       Optional[float]  # automated eval harness score 0-1
    domain:           Optional[str]    # topic domain for sub-agent routing
