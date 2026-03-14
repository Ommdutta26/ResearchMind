# 🔬 NexusResearch – Multi-Agent AI Research System
🚀 Live Demo:researchmind-production.up.railway.app

NexusResearch is an **Agentic AI research system** that autonomously collects information from multiple sources, verifies facts, and generates structured research reports.

The system combines **multi-agent orchestration, hybrid retrieval, and reflection loops** to produce high-quality research outputs while maintaining cost monitoring and human-in-the-loop validation.

---

# 🚀 Features

### 🤖 Multi-Agent Architecture

The system uses **LangGraph** to orchestrate multiple AI agents:

* Planner – generates research queries
* Searcher – retrieves data from multiple sources
* Analyst – writes structured research reports
* Fact Checker – verifies claims against retrieved evidence
* Contrarian – adds opposing perspectives
* Critic – evaluates report quality
* Refiner – improves the report using critique
* Grounding – detects unsupported claims
* Finalizer – produces the final research report

---

### 🔎 Hybrid Retrieval (Advanced RAG)

NexusResearch uses a **hybrid retrieval pipeline**:

* **FAISS Vector Search**
* **BM25 Sparse Retrieval**
* **Cross-Encoder Reranking**

This ensures higher relevance and reduces hallucinations.

---

### 🌐 Multi-Source Knowledge

The agent retrieves information from multiple sources:

* Tavily Web Search
* Wikipedia
* ArXiv Research Papers
* Uploaded PDF Documents
* YouTube Video Transcripts
* Local Vector Knowledge Base

---

### 📄 PDF Knowledge Ingestion

Users can upload research documents which are automatically:

1. Parsed
2. Chunked
3. Embedded
4. Stored in FAISS

This enables **custom domain knowledge integration**.

---

### 🎥 YouTube Knowledge Ingestion

Users can provide a YouTube link and the system will:

1. Extract the transcript
2. Convert it into knowledge chunks
3. Store it in the vector database
4. Use it during research

---

### 🔁 Reflection & Self-Improvement

The system performs **reflection loops**:

Draft Report
→ Critic Evaluation
→ Refiner Improvement

This iterative process improves report quality.

---

### 🧠 Fact Verification

The system includes two verification layers:

**Fact Checker**

* Uses LLM reasoning to verify claims

**Grounding Node**

* Detects unsupported statements

This helps reduce hallucinations.

---

### 👨‍💻 Human-in-the-Loop (HITL)

Before generating the final report, the system pauses and allows the user to:

* Approve the research data
* Reject the research
* Provide additional instructions

---

### 💰 FinOps (Cost Monitoring)

The system tracks:

* Token usage
* API cost
* Budget consumption
* Execution time

This provides **production-level cost awareness**.

---

# 🏗 System Architecture

```
Planner
   ↓
Searcher
   ↓
Hybrid Retrieval
   ↓
Query Expansion
   ↓
Analyst
   ↓
Fact Checker
   ↓
Contrarian
   ↓
Critic
   ↓
Refiner
   ↓
Grounding
   ↓
Finalizer
```

---

# 🛠 Tech Stack

**LLMs**

* Groq (Llama 3 models)

**AI Frameworks**

* LangChain
* LangGraph

**Retrieval**

* FAISS Vector Database
* BM25 Sparse Retrieval
* Cross-Encoder Reranking

**Data Sources**

* Tavily Search
* Wikipedia API
* ArXiv API
* YouTube Transcript API

**Embeddings**

* Sentence Transformers (MiniLM)

**Frontend**

* Streamlit

**Database**

* SQLite (session history & memory)

---

# 📦 Installation

### 1. Clone the repository

```
git clone https://github.com/yourusername/nexus-research-agent.git
cd nexus-research-agent
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Set API keys

```
export GROQ_API_KEY="your_groq_key"
export TAVILY_API_KEY="your_tavily_key"
```

---

# ▶️ Run the Application

```
streamlit run app.py
```

---

# 📊 Example Workflow

1. Enter a research topic
2. The planner generates search queries
3. The searcher retrieves multi-source data
4. User reviews collected research
5. Analyst generates report
6. Fact checker verifies claims
7. Critic evaluates report
8. Refiner improves it
9. Final report is produced

---

# 📸 Example Output

The system produces structured research reports containing:

* Executive Summary
* Key Findings
* Detailed Analysis
* Contrarian Perspective
* Source Citations
* Confidence Score

---

# 🎯 Use Cases

* Automated research assistants
* Academic research support
* Market intelligence
* Technical literature analysis
* Knowledge synthesis

---

# 📈 Future Improvements

* Knowledge graph generation
* Multi-modal research (images & videos)
* Long-term research memory
* Source reliability scoring

---

# 👨‍💻 Author

Developed by **Omm Dutta**

AI / ML Enthusiast | Agentic AI Systems | Retrieval-Augmented Generation

---

# ⭐ If you like this project

Please consider giving the repository a **star** ⭐
