"""Production Hybrid Search Engine combining Dense Vector Search and BM25 with RRF.

Implements Reciprocal Rank Fusion (RRF) and Cross-Encoder Re-ranking to maximize precision and recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from src.documents import ParsedDocument, parse_text_to_structured
from src.vector_store import InMemoryVectorStore, VectorChunk


def _tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer."""
    return re.findall(r"\w+", (text or "").lower())


class FallbackBM25:
    """Pure-Python Okapi BM25 implementation for zero-dependency portability."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / max(1, self.corpus_size)
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_lens = [len(doc) for doc in corpus]

        df: Dict[str, int] = {}
        for doc in corpus:
            freq: Dict[str, int] = {}
            for token in doc:
                freq[token] = freq.get(token, 0) + 1
            self.doc_freqs.append(freq)
            for token in freq:
                df[token] = df.get(token, 0) + 1

        for token, count in df.items():
            # Standard Lucene/Okapi smoothed IDF
            self.idf[token] = math.log((self.corpus_size - count + 0.5) / (count + 0.5) + 1.0)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for idx in range(self.corpus_size):
                tf = self.doc_freqs[idx].get(token, 0)
                if tf > 0:
                    denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_lens[idx] / self.avgdl))
                    scores[idx] += idf_val * ((tf * (self.k1 + 1)) / max(1e-6, denom))
        return scores


class BM25Index:
    """Sparse keyword search index using BM25."""

    def __init__(self, chunks: List[VectorChunk]):
        self.chunks = chunks
        self.tokenized_corpus = [_tokenize(c.content) for c in chunks]
        self._bm25 = None

        if self.tokenized_corpus:
            try:
                from rank_bm25 import BM25Okapi

                self._bm25 = BM25Okapi(self.tokenized_corpus)
            except Exception:
                self._bm25 = FallbackBM25(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 15) -> List[Tuple[VectorChunk, float]]:
        if not self.chunks or not self._bm25:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]


class CrossEncoderReranker:
    """Re-ranks candidate chunks using exact multi-token matches and term-proximity scoring."""

    def score(self, query: str, candidate_text: str) -> float:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return 0.0

        c_tokens = _tokenize(candidate_text)
        if not c_tokens:
            return 0.0

        c_text_lower = candidate_text.lower()
        q_text_lower = query.lower()

        # 1. Exact phrase match bonus
        exact_bonus = 3.0 if q_text_lower in c_text_lower else 0.0

        # 2. Bigram coverage
        q_bigrams = {f"{q_tokens[i]}_{q_tokens[i+1]}" for i in range(len(q_tokens) - 1)}
        c_bigrams = {f"{c_tokens[i]}_{c_tokens[i+1]}" for i in range(len(c_tokens) - 1)}
        bigram_overlap = len(q_bigrams.intersection(c_bigrams)) / max(1, len(q_bigrams)) if q_bigrams else 0.0

        # 3. Unigram coverage
        q_set = set(q_tokens)
        c_set = set(c_tokens)
        unigram_overlap = len(q_set.intersection(c_set)) / max(1, len(q_set))

        return (unigram_overlap * 2.0) + (bigram_overlap * 3.0) + exact_bonus


@dataclass
class HybridSearchResult:
    """Enriched search hit containing chunk, scores, and wider parent context."""

    chunk: VectorChunk
    rrf_score: float
    dense_score: float
    bm25_score: float
    rerank_score: float
    citation: str
    context_to_use: str  # Expanded parent window if available, else chunk content


class HybridRetriever:
    """Unified Dense + Sparse search engine using Reciprocal Rank Fusion & Re-ranking."""

    def __init__(
        self,
        vector_store: Optional[InMemoryVectorStore] = None,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store or InMemoryVectorStore()
        self.bm25_index: Optional[BM25Index] = None
        self.reranker = CrossEncoderReranker()
        self.rrf_k = rrf_k
        self.chunks: List[VectorChunk] = []

    def index_document(self, doc_or_text: ParsedDocument | str) -> int:
        """Indexes parsed document or raw text for both Dense and BM25 retrieval."""
        if isinstance(doc_or_text, str):
            doc = parse_text_to_structured("Document", doc_or_text)
        else:
            doc = doc_or_text

        self.vector_store.clear()
        self.chunks = self.vector_store.build_parent_child_chunks(doc)
        self.bm25_index = BM25Index(self.chunks)
        return len(self.chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_parent_context: bool = True,
    ) -> List[HybridSearchResult]:
        """Performs hybrid retrieval using RRF and cross-encoder re-ranking."""
        if not self.chunks:
            return []

        candidate_limit = min(25, max(top_k * 3, 10))

        # 1. Dense retrieval
        dense_hits = self.vector_store.search(query, top_k=candidate_limit)
        dense_ranks: Dict[str, Tuple[int, float]] = {}
        for rank, (chunk, score) in enumerate(dense_hits, start=1):
            dense_ranks[chunk.chunk_id] = (rank, score)

        # 2. BM25 Sparse retrieval
        bm25_hits = self.bm25_index.search(query, top_k=candidate_limit) if self.bm25_index else []
        bm25_ranks: Dict[str, Tuple[int, float]] = {}
        for rank, (chunk, score) in enumerate(bm25_hits, start=1):
            bm25_ranks[chunk.chunk_id] = (rank, score)

        # 3. Reciprocal Rank Fusion (RRF)
        all_chunk_ids = set(dense_ranks.keys()).union(set(bm25_ranks.keys()))
        chunk_map = {c.chunk_id: c for c in self.chunks}

        fused_candidates: List[Tuple[VectorChunk, float, float, float]] = []
        for c_id in all_chunk_ids:
            chunk = chunk_map.get(c_id)
            if not chunk:
                continue

            dense_rank, dense_score = dense_ranks.get(c_id, (9999, 0.0))
            bm25_rank, bm25_score = bm25_ranks.get(c_id, (9999, 0.0))

            rrf = 0.0
            if c_id in dense_ranks:
                rrf += 1.0 / (self.rrf_k + dense_rank)
            if c_id in bm25_ranks:
                rrf += 1.0 / (self.rrf_k + bm25_rank)

            fused_candidates.append((chunk, rrf, dense_score, bm25_score))

        # Sort by RRF score
        fused_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = fused_candidates[:candidate_limit]

        # 4. Cross-Encoder Re-ranking
        ranked_results: List[HybridSearchResult] = []
        for chunk, rrf_score, dense_score, bm25_score in top_candidates:
            rerank_score = self.reranker.score(query, chunk.content)
            final_composite_score = rrf_score * 100.0 + rerank_score

            context_to_use = (
                chunk.parent_content
                if (use_parent_context and chunk.parent_content)
                else chunk.content
            )

            ranked_results.append(
                HybridSearchResult(
                    chunk=chunk,
                    rrf_score=round(rrf_score, 5),
                    dense_score=round(dense_score, 4),
                    bm25_score=round(bm25_score, 4),
                    rerank_score=round(final_composite_score, 4),
                    citation=chunk.to_citation(),
                    context_to_use=context_to_use,
                )
            )

        ranked_results.sort(key=lambda r: r.rerank_score, reverse=True)
        return ranked_results[:top_k]

    def retrieve_legacy_tuples(
        self, article: str, query: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Drop-in replacement for old retriever.retrieve(article, query) interface."""
        self.index_document(article)
        results = self.retrieve(query, top_k=top_k, use_parent_context=False)
        return [(r.chunk.content, r.dense_score) for r in results]
