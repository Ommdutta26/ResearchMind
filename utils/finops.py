"""
utils/finops.py
FinOps helpers: token totals and cost estimation.
"""

from core.config import GROQ_PRICING


def total_tokens(usage_list: list[dict]) -> int:
    return sum(u.get("total_tokens", 0) for u in (usage_list or []))


def calc_cost(usage_list: list[dict], model: str) -> float:
    pricing = GROQ_PRICING.get(model, {"input": 0.59, "output": 0.79})
    t_in    = sum(u.get("prompt_tokens", 0)     for u in (usage_list or []))
    t_out   = sum(u.get("completion_tokens", 0) for u in (usage_list or []))
    return (t_in * pricing["input"] + t_out * pricing["output"]) / 1_000_000
