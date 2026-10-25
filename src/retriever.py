"""
retriever.py — Article chunking and context retrieval engine.

Splits a source article into overlapping sentence-windows, then
retrieves the top-k most relevant chunks for a given claim/summary
sentence using either:
    • TF-IDF cosine similarity  (fast, zero-GPU)
    • Sentence-Transformer embeddings  (higher quality, needs model)
"""

import re
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')


def split_sentences(text: str) -> List[str]:
    """Regex-based sentence splitter (good enough for news articles)."""
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def make_chunks(
    sentences: List[str],
    chunk_size: int = 3,
    stride: int = 2,
) -> List[str]:
    """Create overlapping multi-sentence chunks from a list of sentences."""
    chunks: List[str] = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(sentences):
            break
        i += stride
    return chunks if chunks else [" ".join(sentences)]


# ---------------------------------------------------------------------------
# TF-IDF retriever
# ---------------------------------------------------------------------------

class TfidfRetriever:
    """Lightweight retriever using TF-IDF + cosine similarity."""

    def retrieve(
        self,
        article: str,
        query: str,
        top_k: int = 3,
        chunk_size: int = 3,
        stride: int = 2,
    ) -> List[Tuple[str, float]]:
        """
        Return the *top_k* article chunks most similar to *query*.

        Returns
        -------
        list of (chunk_text, similarity_score) sorted desc.
        """
        sents = split_sentences(article)
        chunks = make_chunks(sents, chunk_size, stride)
        if not chunks:
            return [(article, 0.0)]

        vectorizer = TfidfVectorizer()
        corpus = chunks + [query]
        tfidf = vectorizer.fit_transform(corpus)
        chunk_vecs = tfidf[:-1]
        query_vec = tfidf[-1:]
        sims = cosine_similarity(query_vec, chunk_vecs)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(chunks[i], float(sims[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Sentence-Transformer retriever (optional, higher quality)
# ---------------------------------------------------------------------------

class EmbeddingRetriever:
    """Semantic retriever using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def retrieve(
        self,
        article: str,
        query: str,
        top_k: int = 3,
        chunk_size: int = 3,
        stride: int = 2,
    ) -> List[Tuple[str, float]]:
        sents = split_sentences(article)
        chunks = make_chunks(sents, chunk_size, stride)
        if not chunks:
            return [(article, 0.0)]

        chunk_embs = self.model.encode(chunks, convert_to_numpy=True)
        query_emb = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(query_emb, chunk_embs)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(chunks[i], float(sims[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_retriever(use_embeddings: bool = False, model_name: str = "all-MiniLM-L6-v2"):
    if use_embeddings:
        return EmbeddingRetriever(model_name)
    return TfidfRetriever()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    article = (
        "Police in Arkansas wish to unlock an iPhone belonging to two teenagers. "
        "The FBI agreed to help in the homicide case. "
        "Syed Farook and his wife were shot dead after killing 14 people. "
        "Previously, the FBI asked Apple to write new software."
    )
    query = "the FBI helped unlock the san bernardino phone"
    retriever = TfidfRetriever()
    results = retriever.retrieve(article, query, top_k=2)
    for chunk, sim in results:
        print(f"[{sim:.3f}] {chunk[:120]}")
