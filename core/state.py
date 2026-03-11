"""
core/state.py
Defines the shared AgentState TypedDict used by every LangGraph node.
"""

import operator
from typing import TypedDict, Annotated, List, Dict


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
