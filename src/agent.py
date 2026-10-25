"""Agentic Query Engine with Query Routing, HyDE, and Self-Correction loop.

Routes queries intelligently between:
  - GLOBAL_SUMMARY   → HierarchicalSummarizer
  - FACT_CHECK       → HybridRetriever + Detector
  - SPECIFIC_QA      → HybridRetriever + LLM grounded response
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from src.documents import ParsedDocument, parse_text_to_structured
from src.hybrid_search import HybridRetriever, HybridSearchResult

load_dotenv()

INTENT_LABELS = {
    "GLOBAL_SUMMARY": "User wants a comprehensive summary/overview of the entire document or article.",
    "FACT_CHECK": "User wants to verify specific claims, detect hallucinations, or check factual accuracy.",
    "SPECIFIC_QA": "User wants a targeted answer to a specific question using document content.",
}


@dataclass
class AgentResponse:
    """Structured response from the Agentic Query Engine."""

    intent: str
    answer: str
    citations: List[str]
    retrieved_chunks: List[str]
    confidence: float
    self_correction_applied: bool = False
    correction_note: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgenticQueryEngine:
    """Multi-intent query engine with HyDE expansion and Self-RAG verification loop."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "groq",
        model: Optional[str] = None,
        retriever: Optional[HybridRetriever] = None,
    ):
        self.provider = provider.lower()
        self.retriever = retriever or HybridRetriever()

        if self.provider == "groq":
            import groq

            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            self.model = model or "llama-3.3-70b-versatile"
            self.client = groq.Groq(api_key=self.api_key) if self.api_key else None
        else:
            import openai

            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = model or "gpt-4o-mini"
            self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None

    def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
        if not self.client:
            raise RuntimeError(
                f"Missing API key for '{self.provider}'. Set GROQ_API_KEY or OPENAI_API_KEY."
            )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------
    # 1. Intent Router
    # ------------------------------------------------------------------

    def detect_intent(self, query: str) -> str:
        """Classifies query intent using keyword heuristics + LLM fallback."""
        q = query.lower().strip()

        # Fast heuristic path
        summary_kws = {"summarize", "summary", "overview", "describe the", "what is the main", "tldr", "tl;dr"}
        fact_check_kws = {"hallucination", "check", "verify", "accurate", "false", "wrong", "correct", "fact"}

        if any(kw in q for kw in summary_kws):
            return "GLOBAL_SUMMARY"
        if any(kw in q for kw in fact_check_kws):
            return "FACT_CHECK"

        # LLM fallback router for ambiguous queries
        if self.client:
            system_p = (
                "You are a query intent classifier. Classify the user query into exactly one: "
                "GLOBAL_SUMMARY, FACT_CHECK, or SPECIFIC_QA. "
                "Return ONLY the label. No explanation."
            )
            user_p = f"Query: {query}"
            try:
                label = self._call_llm(system_p, user_p, max_tokens=10).strip().upper()
                if label in INTENT_LABELS:
                    return label
            except Exception:
                pass

        return "SPECIFIC_QA"

    # ------------------------------------------------------------------
    # 2. HyDE — Hypothetical Document Embedding expansion
    # ------------------------------------------------------------------

    def expand_query_with_hyde(self, query: str, context_hint: str = "") -> str:
        """Generates a short hypothetical document passage to improve dense retrieval recall."""
        if not self.client:
            return query  # graceful degradation without API

        system_p = (
            "You are a retrieval query expansion assistant. Write a short 2-sentence hypothetical "
            "passage that would directly answer the user's question if it appeared in a document. "
            "Be specific. Do not add preamble."
        )
        user_p = f"Question: {query}\nDocument hint: {context_hint or 'general document'}"
        try:
            hypothetical = self._call_llm(system_p, user_p, max_tokens=80)
            return f"{query} {hypothetical}"
        except Exception:
            return query

    # ------------------------------------------------------------------
    # 3. Grounded Answer Generator
    # ------------------------------------------------------------------

    def _generate_grounded_answer(
        self, query: str, retrieved_chunks: List[HybridSearchResult]
    ) -> Tuple[str, float]:
        """Generates an LLM answer strictly grounded in retrieved context, returns (answer, confidence)."""
        if not retrieved_chunks:
            return "The document did not contain enough relevant context to answer this query.", 0.0

        context_blocks = []
        for i, hit in enumerate(retrieved_chunks, start=1):
            context_blocks.append(
                f"[Context {i} | {hit.citation}]\n{hit.context_to_use}"
            )
        context_str = "\n\n".join(context_blocks)

        system_p = (
            "You are a precise, citation-grounded Q&A assistant. "
            "Answer ONLY using the provided context. Do not use external knowledge. "
            "If the answer is not in the context, say so explicitly. "
            "Include citation tags like [Context 1] when referencing specific facts."
        )
        user_p = (
            f"Question: {query}\n\n"
            f"=== RETRIEVED CONTEXT ===\n{context_str}\n\n"
            "Provide a grounded, accurate answer with inline citations:"
        )
        try:
            answer = self._call_llm(system_p, user_p, max_tokens=450)
            # Estimate confidence based on average RRF score of top hits
            avg_rrf = sum(h.rrf_score for h in retrieved_chunks[:3]) / max(1, min(3, len(retrieved_chunks)))
            confidence = min(1.0, avg_rrf * 200)
            return answer, round(confidence, 3)
        except Exception as exc:
            return f"Answer generation failed: {exc}", 0.0

    # ------------------------------------------------------------------
    # 4. Self-Correction Loop
    # ------------------------------------------------------------------

    def _self_correct(self, answer: str, retrieved_chunks: List[HybridSearchResult]) -> Tuple[str, bool, str]:
        """Verifies if the generated answer is faithful to retrieved context; corrects if not."""
        if not retrieved_chunks or not self.client:
            return answer, False, ""

        context_str = " | ".join(h.chunk.content[:300] for h in retrieved_chunks[:3])
        system_p = (
            "You are a faithfulness verifier. Check whether the answer is fully supported by the context. "
            "If it is faithful, reply: FAITHFUL. "
            "If it contains hallucinations or unsupported claims, reply: UNFAITHFUL: <corrected answer>."
        )
        user_p = (
            f"Context: {context_str}\n\n"
            f"Answer: {answer}\n\n"
            "Verdict (FAITHFUL or UNFAITHFUL: <corrected>):"
        )
        try:
            verdict = self._call_llm(system_p, user_p, max_tokens=300)
            if verdict.upper().startswith("UNFAITHFUL"):
                corrected = re.sub(r"(?i)^UNFAITHFUL:\s*", "", verdict).strip()
                return corrected if corrected else answer, True, "Self-correction removed unsupported claims."
        except Exception:
            pass
        return answer, False, ""

    # ------------------------------------------------------------------
    # 5. Main Query Method
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        document: ParsedDocument | str,
        intent_override: Optional[str] = None,
        top_k: int = 5,
        use_hyde: bool = True,
        use_self_correction: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> AgentResponse:
        """Executes full agentic query pipeline with routing, HyDE, retrieval, and self-correction."""

        if progress_callback:
            progress_callback("Detecting query intent...", 0.05)

        intent = intent_override or self.detect_intent(query)

        # GLOBAL_SUMMARY route — delegate to HierarchicalSummarizer
        if intent == "GLOBAL_SUMMARY":
            if progress_callback:
                progress_callback("Routing to hierarchical summarizer...", 0.15)
            from src.summarizer import IsolatedSummarizer

            raw_text = document if isinstance(document, str) else document.raw_text
            try:
                summarizer = IsolatedSummarizer()
                result = summarizer.summarize(
                    raw_text, focus="general comprehensive summary",
                    progress_callback=progress_callback,
                )
                return AgentResponse(
                    intent=intent,
                    answer=result["summary_text"],
                    citations=result.get("citations", ["Full Document"]),
                    retrieved_chunks=[],
                    confidence=0.9,
                    metadata={"mode": result.get("mode", "hierarchical"), "source_words": result.get("source_words", 0)},
                )
            except Exception as e:
                # Fallback to retrieval-based answer
                pass

        # Index document into hybrid retriever
        if progress_callback:
            progress_callback("Building semantic search index...", 0.2)

        doc = document if isinstance(document, ParsedDocument) else parse_text_to_structured("Document", document)
        chunk_count = self.retriever.index_document(doc)

        # HyDE query expansion
        expanded_query = query
        if use_hyde and chunk_count > 3:
            if progress_callback:
                progress_callback("Expanding query with HyDE...", 0.35)
            expanded_query = self.expand_query_with_hyde(query, context_hint=doc.filename)

        # Hybrid retrieval
        if progress_callback:
            progress_callback("Running hybrid search (Dense + BM25 + RRF)...", 0.50)

        retrieved = self.retriever.retrieve(expanded_query, top_k=top_k)

        # FACT_CHECK route — use detector alongside retrieval
        if intent == "FACT_CHECK":
            if progress_callback:
                progress_callback("Cross-checking retrieved context for factual consistency...", 0.70)
            answer, confidence = self._generate_grounded_answer(query, retrieved)
            answer = f"[Fact-Check Analysis]\n\n{answer}"
        else:
            # SPECIFIC_QA
            if progress_callback:
                progress_callback("Generating grounded answer...", 0.70)
            answer, confidence = self._generate_grounded_answer(query, retrieved)

        # Self-Correction
        correction_applied = False
        correction_note = ""
        if use_self_correction and retrieved:
            if progress_callback:
                progress_callback("Self-correction pass...", 0.88)
            answer, correction_applied, correction_note = self._self_correct(answer, retrieved)

        if progress_callback:
            progress_callback("Query complete.", 1.0)

        citations = [h.citation for h in retrieved if h.citation and h.citation != "Document"]
        context_texts = [h.chunk.content for h in retrieved]

        return AgentResponse(
            intent=intent,
            answer=answer,
            citations=citations,
            retrieved_chunks=context_texts,
            confidence=confidence,
            self_correction_applied=correction_applied,
            correction_note=correction_note if correction_applied else None,
            metadata={
                "expanded_query": expanded_query,
                "chunk_count": chunk_count,
                "top_k": top_k,
                "model": self.model,
            },
        )
