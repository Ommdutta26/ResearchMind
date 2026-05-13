"""
utils/retrieval.py
Hybrid retrieval: BM25, FAISS/Chroma similarity search, cross-encoder reranking,
cosine-similarity grounding check, and ingestion helpers.
"""

from __future__ import annotations

import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from youtube_transcript_api import YouTubeTranscriptApi

from core.config import CHUNK_SIZE, CHUNK_OVERLAP
from core.models import get_vector_store, get_reranker, get_embedding_model


# ── BM25 ──────────────────────────────────────────────────────────────────────

def build_bm25_index() -> tuple[BM25Okapi, list[Document]]:
    vs   = get_vector_store()
    docs = vs.similarity_search("research", k=20)
    if not docs:
        return BM25Okapi([["init"]]), [Document(page_content="init")]
    corpus = [doc.page_content.split() for doc in docs]
    return BM25Okapi(corpus), docs


def bm25_search(
    query:      str,
    bm25_index: BM25Okapi,
    bm25_docs:  list[Document],
    top_k:      int = 3,
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


# ── Cosine-similarity grounding (replaces naive string matching) ───────────────
# FIX from v1: the old grounding_node split the draft into sentences and checked
# whether each sentence appeared verbatim in the joined research string.  That
# almost never matched because the LLM paraphrases.  We now embed each sentence
# and find the nearest research chunk; sentences with max cosine sim < threshold
# are flagged as potentially unsupported.

def cosine_grounding(
    draft:         str,
    research_data: list[str],
    threshold:     float = 0.35,
    top_k:         int   = 5,
) -> list[str]:
    """
    Returns up to top_k sentences from `draft` whose maximum cosine similarity
    to any chunk in `research_data` is below `threshold`.
    """
    if not research_data:
        return []

    emb_model = get_embedding_model()

    sentences = [s.strip() for s in draft.split(".") if len(s.strip()) > 30]
    if not sentences:
        return []

    sent_vecs     = np.array(emb_model.embed_documents(sentences))
    research_vecs = np.array(emb_model.embed_documents([r[:400] for r in research_data]))

    # Normalise for cosine similarity
    sent_norms     = np.linalg.norm(sent_vecs,     axis=1, keepdims=True) + 1e-9
    research_norms = np.linalg.norm(research_vecs, axis=1, keepdims=True) + 1e-9
    sent_unit      = sent_vecs     / sent_norms
    research_unit  = research_vecs / research_norms

    # (n_sentences, n_research) cosine similarity matrix
    sim_matrix = sent_unit @ research_unit.T
    max_sims   = sim_matrix.max(axis=1)

    unsupported = [
        sentences[i]
        for i, sim in enumerate(max_sims)
        if sim < threshold
    ]
    return unsupported[:top_k]


# ── Inline citation mapping ───────────────────────────────────────────────────

def build_citation_map(
    draft:    str,
    research: list[str],
    urls:     list[str],
) -> dict[str, str]:
    """
    Maps short sentence fragments to their best-matching source URL using
    cosine similarity.  Returns {fragment: url}.
    """
    if not urls or not research:
        return {}

    emb_model = get_embedding_model()
    sentences  = [s.strip() for s in draft.split(".") if len(s.strip()) > 40]
    if not sentences:
        return {}

    # Align research snippets with urls (best effort, cycle if fewer urls)
    research_with_urls = [
        (r, urls[i % len(urls)]) for i, r in enumerate(research)
    ]
    research_texts = [r for r, _ in research_with_urls]

    sent_vecs     = np.array(emb_model.embed_documents(sentences))
    research_vecs = np.array(emb_model.embed_documents([t[:400] for t in research_texts]))

    sent_norms     = np.linalg.norm(sent_vecs,     axis=1, keepdims=True) + 1e-9
    research_norms = np.linalg.norm(research_vecs, axis=1, keepdims=True) + 1e-9
    sim            = (sent_vecs / sent_norms) @ (research_vecs / research_norms).T

    citation_map: dict[str, str] = {}
    for i, sent in enumerate(sentences):
        best_idx = int(sim[i].argmax())
        if sim[i, best_idx] > 0.4:
            citation_map[sent[:60]] = research_with_urls[best_idx][1]

    return citation_map


# ── Ingestion helpers ─────────────────────────────────────────────────────────

def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )


def ingest_pdf(file_path: str, collection: str = "nexus_default") -> int:
    loader = PyPDFLoader(file_path)
    pages  = loader.load()
    texts  = [p.page_content for p in pages]
    docs   = _splitter().create_documents(texts)
    get_vector_store(collection).add_documents(docs)
    return len(docs)


def _extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    raise ValueError(f"Cannot parse YouTube video ID from URL: {url!r}")


def ingest_youtube(url: str, collection: str = "nexus_default") -> int:
    video_id   = _extract_video_id(url)
    api        = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)
    text       = " ".join(t.text for t in transcript)
    docs       = _splitter().create_documents([text])
    get_vector_store(collection).add_documents(docs)
    return len(docs)


def add_to_vector_store(texts: list[str], collection: str = "nexus_default") -> None:
    docs = [Document(page_content=t) for t in texts]
    get_vector_store(collection).add_documents(docs)
