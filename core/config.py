import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys (loaded from .env — never hardcode these) ──────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")

# ── Model Pricing (per-million tokens, USD) ──────────────────────────────────
GROQ_PRICING: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {"input": 0.59,  "output": 0.79},
    "llama-3.1-8b-instant":    {"input": 0.05,  "output": 0.08},
    "gemma2-9b-it":            {"input": 0.20,  "output": 0.20},
}

DEFAULT_MODEL  = "llama-3.3-70b-versatile"
DEFAULT_DEPTH  = "deep"
DEFAULT_LOOPS  = 1
DEFAULT_BUDGET = 80   # kilo-tokens

DEPTH_QUERY_COUNT = {"quick": 3, "deep": 5, "exhaustive": 7}

# ── Embedding / Reranker model names ────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Chunking ────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# ── SQLite DB path ───────────────────────────────────────────────────────────
DB_PATH = "agent_memory.db"
