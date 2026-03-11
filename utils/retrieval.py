"""
utils/retrieval.py
Hybrid retrieval helpers: BM25 index, FAISS search, cross-encoder reranking,
plus ingestion utilities for PDFs and YouTube transcripts.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from youtube_transcript_api import YouTubeTranscriptApi

from core.config import CHUNK_SIZE, CHUNK_OVERLAP
from core.models import get_vector_store, get_reranker


# ── BM25 ──────────────────────────────────────────────────────────────────────

def build_bm25_index() -> tuple[BM25Okapi, list[Document]]:
    """
    Build a BM25 index over whatever is currently stored in the FAISS vector
    store (fetches up to 20 docs via a generic similarity search).
    """
    vs   = get_vector_store()
    docs = vs.similarity_search("init", k=20)
    corpus = [doc.page_content.split() for doc in docs]
    return BM25Okapi(corpus), docs


def bm25_search(
    query: str,
    bm25_index: BM25Okapi,
    bm25_docs: list[Document],
    top_k: int = 3,
) -> list[Document]:
    scores = bm25_index.get_scores(query.split())
    top    = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [bm25_docs[i] for i in top if scores[i] > 0]


# ── Reranking ─────────────────────────────────────────────────────────────────

def rerank(query: str, docs: list[Document], top_k: int = 3) -> list[Document]:
    if not docs:
        return []
    reranker = get_reranker()
    pairs    = [[query, d.page_content] for d in docs]
    scores   = reranker.predict(pairs)
    scored   = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


# ── Ingestion helpers ─────────────────────────────────────────────────────────

def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )


def ingest_pdf(file_path: str) -> int:
    """Load a PDF, chunk it, and add chunks to the FAISS vector store."""
    loader = PyPDFLoader(file_path)
    pages  = loader.load()
    texts  = [p.page_content for p in pages]
    docs   = _splitter().create_documents(texts)
    get_vector_store().add_documents(docs)
    return len(docs)


def _extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    raise ValueError(f"Cannot parse YouTube video ID from URL: {url!r}")


def ingest_youtube(url: str) -> int:
    """Fetch a YouTube transcript, chunk it, and add to the FAISS vector store."""
    video_id   = _extract_video_id(url)
    api        = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)
    text       = " ".join(t.text for t in transcript)
    docs       = _splitter().create_documents([text])
    get_vector_store().add_documents(docs)
    return len(docs)


# ── Misc ──────────────────────────────────────────────────────────────────────

def add_to_vector_store(texts: list[str]) -> None:
    docs = [Document(page_content=t) for t in texts]
    get_vector_store().add_documents(docs)


def compute_confidence(docs_with_scores: list[tuple]) -> float:
    if not docs_with_scores:
        return 0.0
    scores    = [score for _, score in docs_with_scores]
    avg_score = sum(scores) / len(scores)
    return 1 / (1 + avg_score)
