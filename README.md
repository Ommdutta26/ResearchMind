# NexusResearch

A multi-agent research system built on LangGraph. Given a topic, it plans domain-specific queries, retrieves data from multiple sources, drafts a report, verifies claims against evidence, critiques and refines the output, and produces a cited final report with an automated quality score.

Live demo: https://nexus-research-agent-latest.onrender.com/
**Project Demo:** https://drive.google.com/file/d/1aG7FciW8SFNLIYnoHuY0LL6etrNEf-Z2/view?usp=sharing

---

## Overview

The system is implemented as a directed graph of specialized agents (LangGraph `StateGraph`), each responsible for one stage of the research pipeline. State is passed between nodes as a typed `AgentState` object, with structured outputs enforced via Pydantic contracts at every stage that matters for downstream logic — draft critique, fact-checking, domain classification.

Retrieval is hybrid: dense vector search (Chroma, persistent across sessions), sparse BM25, and cross-encoder reranking. Claim verification runs at two independent layers — a fact-checker that compares extracted claims against retrieved evidence, and a grounding node that flags low-similarity sentences in the final draft using cosine similarity against the embedded research corpus.

The pipeline includes a human-in-the-loop gate after retrieval: the user reviews collected snippets and sources before the analyst drafts anything, and can approve (with optional steering instructions) or reject.

---

## Architecture

```
Topic Input
    │
    ▼
Domain Classifier      (finance / biotech / geopolitics / technology / science / general)
    │
    ▼
Planner                (domain-aware, non-overlapping search queries)
    │
    ▼
Searcher               (Tavily + Wikipedia + ArXiv + Chroma + BM25 + cross-encoder rerank)
    │
    ▼
Query Expander         (follow-up queries generated from retrieved data)
    │
    ▼
──────────────── HITL Gate: Approve / Reject ────────────────
    │ approved
    ▼
Analyst                (streaming report generation, domain-specific structure)
    │
    ▼
Fact Checker           (Pydantic FactCheckOutput: supported / unsupported claims)
    │
    ▼
Contrarian             (opposing perspectives, blind spots, second-order risks)
    │
    ▼
Critic                 (Pydantic CritiqueOutput, 1–10 score)
    │
    ├── score < 7 and loops remaining ──► Refiner ──► back to Critic
    │
    └── score ≥ 7 or max loops reached
            │
            ▼
        Grounding       (cosine similarity, draft sentence vs. research corpus)
            │
            ▼
        Citation Engine (sentence → best-matching source URL)
            │
            ▼
        Eval Harness    (judge LLM: accuracy, depth, clarity, structure, actionability)
            │
            ▼
        Finalizer       (assembles report + confidence score + citations + sources)
```

Each node reads and writes a shared `AgentState` (`core/state.py`). Nodes that feed conditional logic downstream (critic, fact-checker, domain classifier) return validated Pydantic models rather than raw text, so the graph's routing logic never depends on parsing LLM prose.

---

## Design decisions worth knowing

**Structured output over string parsing.** Early iterations of the critic and fact-checker parsed free-text LLM output with regex, which broke silently whenever the model changed phrasing. Both nodes now use `llm.with_structured_output()` against explicit Pydantic schemas, so a malformed response fails loudly (validation error) instead of silently producing a wrong score or an empty claim list.

**Cosine similarity for grounding, not exact match.** Grounding originally checked whether draft sentences appeared verbatim in retrieved chunks. This almost never matched, because the analyst paraphrases source material by design. Grounding now embeds each draft sentence and compares it against the corpus embeddings via cosine similarity, with a threshold below which a sentence is flagged as unsupported.

**Query expansion runs after retrieval, not before.** The query expander depends on real search results to generate meaningful follow-up queries. Running it before the searcher meant it operated on empty state and produced generic, non-additive queries. It now runs after the searcher node in the graph.

**LLM client caching keyed on `(model, temperature)`.** The LLM factory uses `lru_cache`, but caching only on model name caused nodes with different temperature requirements (e.g., a deterministic critic vs. a more exploratory analyst) to silently share a client and inherit the wrong temperature. The cache key now includes both.

**Persistent vector store.** Chroma is used instead of an in-memory store so that retrieved snippets accumulate across sessions — later research runs benefit from a growing local knowledge base instead of starting cold each time.

**Two independent verification layers.** Fact-checking (claim-level, against explicit evidence) and grounding (sentence-level, cosine similarity against the full corpus) catch different failure modes — the former catches invented facts, the latter catches subtle drift during paraphrasing that a claim-level check can miss.

---

## Features

- **Domain-aware prompting** — classifier routes each topic to specialist planner/analyst prompts (finance, biotech, geopolitics, technology, science, general)
- **Hybrid retrieval** — Chroma dense search + BM25 sparse retrieval + cross-encoder reranking (ms-marco MiniLM)
- **Multi-source ingestion** — Tavily web search, Wikipedia, ArXiv, uploaded PDFs, YouTube transcripts, persistent vector store
- **Reflection loop** — critic/refiner cycle with configurable max iterations and score threshold
- **Two-layer verification** — claim-level fact-checking and sentence-level grounding
- **Inline citation mapping** — each sentence in the final report is matched to its closest source URL
- **Automated evaluation** — judge LLM scores every report on accuracy, depth, clarity, structure, and actionability (0–1 composite)
- **Streaming generation** — analyst output streams token-by-token via `llm.stream()`
- **Human-in-the-loop gate** — review and approve/reject collected research before drafting begins
- **Export** — Markdown, PDF (ReportLab), DOCX (python-docx)
- **Feedback capture** — thumbs up/down + comment stored in SQLite
- **Cost tracking** — per-run token counts and estimated cost by model
- **Optional LangSmith tracing** — node-level timings, token counts, and prompt replay when `LANGCHAIN_API_KEY` is set
- **Scheduled research queue** — topics with frequency and delivery email, stored in SQLite, hookable to APScheduler

---

## Tech stack

| Layer | Technology |
|---|---|
| LLMs | Groq (Llama 3.3 70B, Llama 3.1 8B, Gemma 2 9B) |
| Orchestration | LangGraph (`StateGraph`, `interrupt`, `Command`) |
| Framework | LangChain |
| Vector store | Chroma (persistent) |
| Sparse retrieval | BM25 (rank-bm25) |
| Reranking | Cross-encoder, ms-marco MiniLM L6 |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Web search | Tavily |
| Knowledge sources | Wikipedia API, ArXiv API, YouTube Transcript API |
| Structured output | Pydantic v2, `llm.with_structured_output()` |
| PDF export | ReportLab |
| DOCX export | python-docx |
| Frontend | Streamlit |
| Database | SQLite (run history, feedback, topic queue) |
| Checkpointing | langgraph-checkpoint-sqlite |
| Observability | LangSmith (optional) |
| Scheduling | APScheduler (optional) |

---

## Installation

```bash
git clone https://github.com/yourusername/nexus-research-agent.git
cd nexus-research-agent
pip install -r requirements.txt
```

`langgraph-checkpoint-sqlite` is a separate package from `langgraph` and must be installed explicitly — it's included in `requirements.txt`.

Configure environment variables:

```bash
cp .env.example .env
```

```env
# Required
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here

# Optional — enables LangSmith tracing
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=nexus-research
```

Run:

```bash
streamlit run app.py
```

A `chroma_db/` directory is created on first run and persists the vector store across sessions.

---

## Workflow

1. Enter a research topic, or select a trending suggestion from the sidebar (sourced via Tavily).
2. The domain classifier labels the topic and selects specialist prompts.
3. The planner generates domain-aware search queries.
4. The searcher retrieves from five-plus sources and writes results into Chroma.
5. The query expander generates follow-up queries from the retrieved data.
6. The pipeline pauses at the HITL gate — review snippets and sources, then approve (optionally with steering instructions) or reject.
7. The analyst streams the report draft.
8. The fact-checker verifies extracted claims against evidence.
9. The contrarian adds risk and counter-perspective sections.
10. The critic scores the draft; if below threshold and loops remain, the refiner revises and the draft is re-scored.
11. Grounding flags any sentence below the similarity threshold.
12. The citation engine maps sentences to source URLs.
13. The eval harness scores the final report on five dimensions.
14. The finalizer assembles the report with confidence score, citation map, and source list.
15. Export as Markdown, PDF, or DOCX. Rate the report.

---

## Report structure

Every generated report includes:

- Executive summary
- Key findings
- Detailed analysis (domain-specific sections)
- Risks and counterpoints
- Contrarian perspective
- Conclusion
- Confidence score and eval harness score
- Citation map (sentence → source URL)
- Source list

---

## Project structure

```
nexus-research/
├── app.py                  # Streamlit entrypoint
├── requirements.txt
├── .env.example
│
├── agents/
│   ├── graph.py             # StateGraph definition
│   └── nodes.py             # Node implementations
│
├── core/
│   ├── config.py             # Constants, pricing table, env loading
│   ├── database.py           # SQLite: history, feedback, topic queue
│   ├── models.py              # LLM factory (lru_cache), Chroma client, tools
│   └── state.py               # AgentState TypedDict + Pydantic contracts
│
├── ui/
│   ├── components.py        # Progress bar, metrics, citations, feedback widget
│   ├── sidebar.py             # Settings, ingestion, scheduler, run history
│   └── styles.py               # Global CSS
│
└── utils/
    ├── export.py              # PDF (ReportLab) + DOCX (python-docx)
    ├── finops.py               # Token counting, cost estimation
    └── retrieval.py            # BM25, reranking, cosine grounding, citation matching
```

---

## Use cases

- Investment research (finance domain)
- Drug pipeline analysis (biotech domain)
- Geopolitical briefings
- Technical literature synthesis
- Academic research support
- Market intelligence reports

---

## Roadmap

- Knowledge graph generation (Neo4j / NetworkX)
- Multi-modal research (image and video understanding)
- Multi-user OAuth with per-user isolated namespaces
- Source reliability scoring
- Comparative research mode (parallel topic branches with diff)
- Long-term cross-session memory with topic clustering

---

## Author

**Omm Dutta**
AI/ML · Agentic Systems · Retrieval-Augmented Generation
