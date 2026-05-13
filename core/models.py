"""
core/models.py
Cached singletons for LLMs, embedding model, reranker, and search tools.

FIX from v1: get_llm no longer uses @st.cache_resource because Streamlit
caches on the first call's args, silently returning the wrong temperature
when different nodes request different temps.  We use a module-level LRU
cache keyed on (model, temp) instead.
"""

from __future__ import annotations

import functools

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


# ── Embedding model ───────────────────────────────────────────────────────────

@st.cache_resource
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


# ── Reranker ──────────────────────────────────────────────────────────────────

@st.cache_resource
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


# ── Chroma persistent vector store ───────────────────────────────────────────
# Survives restarts; accumulates ingested PDFs and YT transcripts across sessions.

def get_vector_store():
    emb = get_embedding_model()

    try:
        vs = FAISS.load_local(
            "faiss_index",
            emb,
            allow_dangerous_deserialization=True
        )
    except:
        vs = FAISS.from_texts(
            ["System initialized knowledge base."],
            emb
        )
        vs.save_local("faiss_index")

    return vs


# ── LLM factory — LRU-cached per (model, temp) ───────────────────────────────

@functools.lru_cache(maxsize=16)
def get_llm(
    model: str = "llama-3.3-70b-versatile",
    temp:  float = 0.0,
) -> ChatGroq:
    """
    Returns a ChatGroq instance.  Cached by (model, temp) so different nodes
    that request different temperatures always get the right object.
    """
    return ChatGroq(model=model, temperature=temp)


# ── Search tools ──────────────────────────────────────────────────────────────

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
