"""In-memory Vector Store with Parent-Child chunking and persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.documents import DocumentSection, ParsedDocument, parse_text_to_structured


@dataclass
class VectorChunk:
    """Child chunk indexed for embedding search with reference to its wider parent."""

    chunk_id: str
    parent_id: str
    content: str
    parent_content: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_citation(self) -> str:
        parts = []
        if self.page_number:
            parts.append(f"Page {self.page_number}")
        if self.section_title:
            parts.append(f"Section: {self.section_title}")
        return ", ".join(parts) if parts else "Document"


class SimpleEmbeddingProvider:
    """Flexible embedding generator supporting SentenceTransformers, OpenAI, or TF-IDF projection."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self._st_model = None
        self._openai_client = None
        self._fallback_vectorizer = None

        # Check for sentence_transformers
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(
                model_name or "all-MiniLM-L6-v2"
            )
        except Exception:
            pass

        # Check for OpenAI embeddings if key present
        openai_key = os.getenv("OPENAI_API_KEY")
        if not self._st_model and openai_key:
            try:
                import openai

                self._openai_client = openai.OpenAI(api_key=openai_key)
            except Exception:
                pass

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encodes list of texts into normalized embedding vectors."""
        if not texts:
            return []

        # 1. Sentence Transformers
        if self._st_model is not None:
            embeddings = self._st_model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()

        # 2. OpenAI text-embedding-3-small
        if self._openai_client is not None:
            try:
                resp = self._openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts,
                )
                return [d.embedding for d in resp.data]
            except Exception:
                pass

        # 3. Deterministic Scikit-Learn TF-IDF projection fallback
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self._fallback_vectorizer is None or len(texts) > 5:
            self._fallback_vectorizer = TfidfVectorizer(
                max_features=384,
                ngram_range=(1, 2),
                stop_words="english",
            )
            dense_matrix = self._fallback_vectorizer.fit_transform(texts).toarray()
        else:
            try:
                dense_matrix = self._fallback_vectorizer.transform(texts).toarray()
            except Exception:
                self._fallback_vectorizer.fit(texts)
                dense_matrix = self._fallback_vectorizer.transform(texts).toarray()

        norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = (dense_matrix / norms).tolist()
        return normalized

    def encode_query(self, query: str) -> List[float]:
        res = self.encode([query])
        return res[0] if res else []


class InMemoryVectorStore:
    """Production in-memory vector store supporting parent-child chunking & persistence."""

    def __init__(self, embedding_provider: Optional[SimpleEmbeddingProvider] = None):
        self.embedder = embedding_provider or SimpleEmbeddingProvider()
        self.chunks: List[VectorChunk] = []
        self._matrix: Optional[np.ndarray] = None

    def clear(self) -> None:
        self.chunks = []
        self._matrix = None

    def add_chunks(self, chunks: List[VectorChunk]) -> None:
        """Adds pre-built chunks to the store and indexes their embeddings."""
        if not chunks:
            return

        texts_to_embed = [c.content for c in chunks if c.embedding is None]
        if texts_to_embed:
            embeddings = self.embedder.encode(texts_to_embed)
            e_idx = 0
            for c in chunks:
                if c.embedding is None:
                    c.embedding = embeddings[e_idx]
                    e_idx += 1

        self.chunks.extend(chunks)
        self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        if not self.chunks:
            self._matrix = None
            return
        vecs = [c.embedding for c in self.chunks if c.embedding is not None]
        if vecs:
            self._matrix = np.array(vecs, dtype=np.float32)
            # Normalize for fast dot product cosine similarity
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            self._matrix = self._matrix / norms
        else:
            self._matrix = None

    def search(self, query: str, top_k: int = 5) -> List[Tuple[VectorChunk, float]]:
        """Returns top-k most similar chunks with cosine scores."""
        if not self.chunks or self._matrix is None:
            return []

        q_vec = np.array(self.embedder.encode_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = np.dot(self._matrix, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Tuple[VectorChunk, float]] = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))
        return results

    def build_parent_child_chunks(
        self,
        doc: ParsedDocument,
        parent_chunk_words: int = 650,
        child_chunk_words: int = 140,
        child_overlap_words: int = 30,
    ) -> List[VectorChunk]:
        """Creates fine-grained child chunks tied to broader parent context windows."""
        generated_chunks: List[VectorChunk] = []
        parent_idx = 0

        # Step 1: Build Parent Windows
        sections = doc.sections
        current_words: List[str] = []
        current_pages: List[int] = []
        current_titles: List[str] = []

        def flush_parent():
            nonlocal current_words, current_pages, current_titles, parent_idx
            if not current_words:
                return

            p_text = " ".join(current_words)
            p_id = f"parent_{parent_idx}"
            parent_idx += 1

            p_page = current_pages[0] if current_pages else None
            p_title = current_titles[-1] if current_titles else "General"

            # Step 2: Split Parent into Child Chunks
            words = current_words
            c_idx = 0
            for start in range(0, len(words), child_chunk_words - child_overlap_words):
                end = min(len(words), start + child_chunk_words)
                child_words = words[start:end]
                if not child_words:
                    continue

                child_text = " ".join(child_words)
                child_id = f"{p_id}_c{c_idx}"
                c_idx += 1

                chunk = VectorChunk(
                    chunk_id=child_id,
                    parent_id=p_id,
                    content=child_text,
                    parent_content=p_text,
                    page_number=p_page,
                    section_title=p_title,
                    metadata={"source_file": doc.filename, "child_words": len(child_words)},
                )
                generated_chunks.append(chunk)

            current_words = []
            current_pages = []
            current_titles = []

        for sec in sections:
            sec_words = sec.content.split()
            if sec.page_number and sec.page_number not in current_pages:
                current_pages.append(sec.page_number)
            if sec.section_title and sec.section_title not in current_titles:
                current_titles.append(sec.section_title)

            current_words.extend(sec_words)
            if len(current_words) >= parent_chunk_words:
                flush_parent()

        flush_parent()

        # If document had very few words, ensure at least 1 chunk is created
        if not generated_chunks and doc.raw_text.strip():
            raw_words = doc.raw_text.split()
            chunk = VectorChunk(
                chunk_id="parent_0_c0",
                parent_id="parent_0",
                content=" ".join(raw_words[:child_chunk_words]),
                parent_content=doc.raw_text,
                page_number=1,
                section_title="Overview",
                metadata={"source_file": doc.filename},
            )
            generated_chunks.append(chunk)

        # Ingest and embed generated chunks
        self.add_chunks(generated_chunks)
        return generated_chunks

    def save_to_file(self, filepath: str | Path) -> None:
        """Serializes vector store to JSON file."""
        data = [asdict(c) for c in self.chunks]
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str | Path) -> None:
        """Deserializes vector store from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = [VectorChunk(**item) for item in data]
        self.clear()
        self.chunks = chunks
        self._rebuild_matrix()
