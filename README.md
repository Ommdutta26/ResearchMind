# 🔬 NexusResearch v2 – Multi-Agent AI Research System

🚀 **Live Demo:** https://nexus-research-agent-latest.onrender.com/
🚀 **Project Demo:** https://drive.google.com/file/d/1aG7FciW8SFNLIYnoHuY0LL6etrNEf-Z2/view?usp=sharing
NexusResearch is an **Agentic AI research system** that autonomously collects information from multiple sources, verifies facts, and generates structured research reports.

The system combines **multi-agent orchestration, hybrid retrieval, reflection loops, inline citations, and an automated eval harness** to produce high-quality research outputs — with full cost monitoring and human-in-the-loop validation.

---

## 🆕 What's New in v2

| Area | v1 | v2 |
|---|---|---|
| Vector store | FAISS (ephemeral, resets on restart) | **Chroma** (persistent on disk, accumulates across sessions) |
| LLM temperature | Silently cached wrong temp across nodes | Fixed with `lru_cache` keyed on `(model, temp)` |
| Node outputs | Raw dict with brittle string parsing | **Pydantic structured output** contracts |
| Grounding | Verbatim string match (never matched) | **Cosine similarity** against embedded research chunks |
| Query expander | Ran before searcher (empty data) | Runs **after searcher** (has real data) |
| New nodes | — | Domain classifier, Citation engine, Eval harness |
| LLM output | Blocking until full response | **Streaming token-by-token** in analyst node |
| Export | Markdown `.md` only | **Markdown + PDF + DOCX** |
| Feedback | None | 👍/👎 + comment stored in SQLite |
| Observability | None | **LangSmith tracing** (opt-in) |
| Topic discovery | Manual only | **Trending topic auto-suggest** via Tavily |
| Scheduling | None | **Research queue** (APScheduler-hookable) |
| Pipeline visibility | Text logs only | **Visual progress bar** per node |

---

## 🚀 Features

### 🤖 Multi-Agent Architecture

The system uses **LangGraph** to orchestrate multiple AI agents in a directed graph with conditional reflection loops:

| Agent | Role |
|---|---|
| **Domain Classifier** | Labels the topic (finance / biotech / geopolitics / technology / science / general) to unlock specialist prompts |
| **Planner** | Generates domain-aware, non-overlapping search queries |
| **Searcher** | Retrieves data from web, Wikipedia, ArXiv, and the local vector store |
| **Query Expander** | Generates deeper follow-up queries from *real* retrieved data |
| **Analyst** | Writes a structured streaming report with domain-specific sections |
| **Fact Checker** | Verifies claims against retrieved evidence using Pydantic structured output |
| **Contrarian** | Adds rigorous opposing perspectives, blind spots, and second-order risks |
| **Critic** | Rates report quality 1–10 with Pydantic structured output |
| **Refiner** | Improves the report based on critic feedback (configurable loops) |
| **Grounding** | Flags low-similarity sentences using cosine similarity |
| **Citation Engine** | Maps each sentence to its best-matching source URL |
| **Eval Harness** | Judge LLM scores accuracy, depth, clarity, structure, and actionability (0–1) |
| **Finalizer** | Assembles the final report with confidence score, citations, and source list |

---

### 🔎 Hybrid Retrieval (Advanced RAG)

NexusResearch uses a **hybrid retrieval pipeline**:

- **Chroma Vector Search** (persistent across sessions)
- **BM25 Sparse Retrieval**
- **Cross-Encoder Reranking** (ms-marco MiniLM)

All retrieved snippets are added back to the vector store, building a cumulative knowledge base over time.

---

### 📍 Domain-Aware Specialist Prompts

The domain classifier routes each topic to tailored planner and analyst prompts:

- **Finance** — valuations, risk factors, market dynamics, catalysts
- **Biotech** — clinical stage, FDA pipeline, competitive landscape
- **Geopolitics** — key actors, alliances, trade flows, regional stability
- **Technology** — technical moats, adoption curves, patents
- **Science** — methodology, reproducibility, peer-review quality
- **General** — comprehensive multi-dimensional coverage

---

### 🌐 Multi-Source Knowledge

The agent retrieves information from:

- Tavily Web Search
- Wikipedia
- ArXiv Research Papers
- Uploaded PDF Documents
- YouTube Video Transcripts
- Persistent Chroma Vector Knowledge Base

---

### 🔗 Inline Citation Engine

After report generation, each sentence is embedded and matched to its closest source URL via cosine similarity. The report displays a **Citation Map** linking specific claims to their evidence sources.

---

### 🤖 Automated Eval Harness

A judge LLM evaluates every report on 5 dimensions:

1. **Accuracy** — factual correctness
2. **Depth** — thoroughness of analysis
3. **Clarity** — readability and structure
4. **Structure** — section organization
5. **Actionability** — practical usefulness

A composite **0–1 score** is stored in run history and displayed with a colour-coded progress bar.

---

### ⚡ Streaming LLM Output

The analyst node streams tokens into the UI in real time via `llm.stream()` — users see the report being written word-by-word instead of waiting for a full blocking response.

---

### 🔁 Reflection & Self-Improvement

The system performs configurable reflection loops using **Pydantic-structured critic output**:

```
Draft Report → Critic (Pydantic) → Score < 7? → Refiner → repeat
                                 → Score ≥ 7 or max loops? → Grounding → Citation → Eval → Finalizer
```

---

### 🧠 Fact Verification (Two Layers)

**Fact Checker** — Pydantic `FactCheckOutput` with `supported_claims` and `unsupported_claims` lists

**Grounding Node** — Cosine similarity between each draft sentence and the research corpus. Replaces the v1 verbatim match that never worked because LLMs paraphrase.

---

### 👤 Human-in-the-Loop (HITL)

Before generating the final report, the system pauses and lets the user:

- Preview the collected research snippets and source URLs
- Approve with optional extra instructions (e.g. "Focus on downside risks")
- Reject to stop the pipeline

---

### ⬇️ Rich Export

Reports can be exported in three formats:

- **Markdown** `.md` — always available, instant download
- **PDF** — formatted with ReportLab (section headers, bullet points, metadata)
- **DOCX** — formatted Word document with branded header via python-docx

---

### 💬 Report Feedback

A 👍/👎 rating widget lets users rate report quality. Ratings and optional comments are stored in a `feedback` SQLite table for continuous improvement tracking.

---

### 📈 Pipeline Progress Bar

A visual step indicator shows which node is currently executing, with completed steps highlighted. Built from the node log stream — no extra instrumentation needed.

---

### 💡 Trending Topic Auto-Suggest

The sidebar fetches trending research topics via Tavily. Clicking a suggestion pre-fills the topic input — one click to start researching something timely.

---

### ⏰ Scheduled Research Queue

Users can add topics to a queue with frequency (daily/weekly) and an email address. The queue is stored in SQLite and hookable to APScheduler for automated delivery.

---

### 💰 FinOps (Cost Monitoring)

The system tracks per-run:

- Input / output / total token counts
- Estimated API cost (per-model pricing table)
- Execution time
- Eval harness score (stored in history)

---

### 🔭 LangSmith Tracing (Optional)

Set `LANGCHAIN_API_KEY` in `.env` to enable full LangSmith tracing — node-level timings, token counts per step, exact prompts, and replay for every run.

---

## 🏗 System Architecture

```
Topic Input
    ↓
Domain Classifier  (finance / biotech / geopolitics / technology / science / general)
    ↓
Planner            (domain-aware specialist queries)
    ↓
Searcher           (Tavily + Wikipedia + ArXiv + Chroma + BM25 + Reranker)
    ↓
Query Expander     (deeper follow-ups from real data)
    ↓
─────────────── HITL Gate ─── Approve / Reject ───────────────
    ↓ (Approved)
Analyst            (streaming domain-specific report)
    ↓
Fact Checker       (Pydantic FactCheckOutput)
    ↓
Contrarian         (risks, blind spots, second-order effects)
    ↓
Critic             (Pydantic CritiqueOutput, score 1-10)
    ↓
  score < 7 and loops remaining?
    ↓ Yes                    ↓ No
  Refiner ──────────────→ Grounding (cosine similarity)
                              ↓
                           Citation Engine (sentence → URL)
                              ↓
                           Eval Harness (judge LLM, 0-1 score)
                              ↓
                           Finalizer
                              ↓
                        Final Report + Export
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **LLMs** | Groq (Llama 3.3 70B, Llama 3.1 8B, Gemma 2 9B) |
| **Orchestration** | LangGraph (StateGraph, interrupt, Command) |
| **AI Framework** | LangChain |
| **Vector Store** | Chroma (persistent) |
| **Sparse Retrieval** | BM25 (rank-bm25) |
| **Reranking** | Cross-Encoder (ms-marco MiniLM L6) |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Web Search** | Tavily |
| **Knowledge Sources** | Wikipedia API, ArXiv API, YouTube Transcript API |
| **Structured Output** | Pydantic v2 with `llm.with_structured_output()` |
| **PDF Export** | ReportLab |
| **DOCX Export** | python-docx |
| **Frontend** | Streamlit |
| **Database** | SQLite (history, feedback, topic queue) |
| **Checkpointing** | langgraph-checkpoint-sqlite |
| **Observability** | LangSmith (optional) |
| **Scheduling** | APScheduler (optional) |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/nexus-research-agent.git
cd nexus-research-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `langgraph-checkpoint-sqlite` is a separate package from `langgraph` and must be installed explicitly.

### 3. Configure environment variables

Copy the template and fill in your keys:

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

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

A `chroma_db/` directory is created automatically on first run and persists your knowledge base across sessions.

---

## 📊 Example Workflow

1. Enter a research topic (or click a trending suggestion from the sidebar)
2. The **domain classifier** labels the topic and selects specialist prompts
3. The **planner** generates domain-aware queries
4. The **searcher** retrieves data from 5+ sources and stores it in Chroma
5. The **query expander** generates follow-up queries from real retrieved data
6. You review the collected snippets → **Approve** (with optional instructions) or **Reject**
7. The **analyst** writes the report with streaming token output
8. The **fact checker** verifies claims (Pydantic structured output)
9. The **contrarian** adds risk perspectives
10. The **critic** scores the report; the **refiner** improves it if score < 7
11. **Grounding** flags low-similarity sentences via cosine similarity
12. The **citation engine** maps sentences to source URLs
13. The **eval harness** scores accuracy, depth, clarity, structure, actionability
14. The **final report** is assembled with confidence score + citation map
15. Export as Markdown, PDF, or DOCX — rate the report with 👍/👎

---

## 📸 Report Structure

Every generated report contains:

- **Executive Summary**
- **Key Findings**
- **Detailed Analysis** (domain-specific sections)
- **Risks & Counterpoints**
- **Contrarian Perspective**
- **Conclusion**
- **Confidence Score** + **Eval Harness Score**
- **Citation Map** (sentence → source URL)
- **Source List**

---

## 🎯 Use Cases

- Automated investment research (finance domain)
- Drug pipeline analysis (biotech domain)
- Geopolitical briefings
- Technical literature synthesis
- Academic research support
- Market intelligence reports

---

## 📁 Project Structure

```
nexus-research/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── .env.example
├── CHANGELOG.md            # Full v1 → v2 migration notes
│
├── agents/
│   ├── graph.py            # LangGraph StateGraph definition
│   └── nodes.py            # All node functions
│
├── core/
│   ├── config.py           # Constants, pricing, env loading
│   ├── database.py         # SQLite: history, feedback, topic queue
│   ├── models.py           # LLM factory (lru_cache), Chroma, tools
│   └── state.py            # AgentState TypedDict + Pydantic contracts
│
├── ui/
│   ├── components.py       # Progress bar, metrics, citations, feedback
│   ├── sidebar.py          # Settings, ingestion, scheduler, history
│   └── styles.py           # Global CSS
│
└── utils/
    ├── export.py           # PDF (ReportLab) + DOCX (python-docx)
    ├── finops.py           # Token counting, cost estimation
    └── retrieval.py        # BM25, reranking, cosine grounding, citations
```

---

## 📈 Roadmap

- Knowledge graph generation (Neo4j / NetworkX)
- Multi-modal research (image + video understanding)
- Multi-user OAuth (per-user isolated namespaces)
- Source reliability scoring
- Comparative research mode (parallel topic branches + diff)
- Long-term cross-session memory with topic clustering

---

## 👨‍💻 Author

Developed by **Omm Dutta**

AI / ML Enthusiast | Agentic AI Systems | Retrieval-Augmented Generation

---

## ⭐ If you like this project

Please consider giving the repository a **star** ⭐
