"""
core/models.py
Cached singletons for LLMs, embedding model, reranker, and search tools.
"""

import streamlit as st
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from core.config import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    TAVILY_API_KEY,
)


# ── Embedding model ──────────────────────────────────────────────────────────
@st.cache_resource
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


# ── Reranker ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


# ── FAISS vector store (seed with a placeholder doc) ────────────────────────
@st.cache_resource
def get_vector_store() -> FAISS:
    emb = get_embedding_model()
    return FAISS.from_texts(["System initialized knowledge base."], emb)


# ── LLM factory ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm(model: str = "llama-3.3-70b-versatile", temp: float = 0.0) -> ChatGroq:
    return ChatGroq(model=model, temperature=temp)


# ── Search tools ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_tools():
    tavily = TavilySearch(max_results=3, tavily_api_key=TAVILY_API_KEY)

    try:
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
    except Exception:
        wiki = None

    try:
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2))
    except Exception:
        arxiv = None

    return tavily, wiki, arxiv
