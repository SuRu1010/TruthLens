"""Hierarchical Map-Reduce & Tree-based Summarizer for arbitrarily large documents.

Solves the token-limit bottleneck for 50-200+ page PDFs and DOCXs without silent truncation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from src.documents import DocumentSection, ParsedDocument, parse_text_to_structured


@dataclass
class ChunkSummaryNode:
    """Represents an intermediate summary node in the hierarchical tree."""

    chunk_id: int
    level: int
    title: str
    summary_text: str
    citations: List[str] = field(default_factory=list)
    word_count: int = 0


@dataclass
class HierarchicalSummaryResult:
    """Complete output of hierarchical summarization."""

    final_summary: str
    executive_summary: str
    key_findings: List[str]
    section_breakdowns: List[Dict[str, Any]]
    total_source_words: int
    summary_words: int
    compression_ratio: float
    tree_levels: int
    citations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalSummarizer:
    """Multi-stage map-reduce summarizer capable of synthesizing massive documents."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "groq",
        model: Optional[str] = None,
        target_chunk_words: int = 900,
        chunk_overlap_words: int = 100,
    ):
        self.provider = provider.lower().strip()
        self.target_chunk_words = target_chunk_words
        self.chunk_overlap_words = chunk_overlap_words

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

    def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
        """Invokes the configured LLM client with fallback handling."""
        if not self.client:
            raise RuntimeError(
                f"Missing API key for provider '{self.provider}'. "
                f"Set {'GROQ_API_KEY' if self.provider == 'groq' else 'OPENAI_API_KEY'}."
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

    def _chunk_document(self, doc: ParsedDocument) -> List[Tuple[str, List[str], str]]:
        """Groups document sections into coherent semantic chunks with citations.

        Returns list of tuples: (chunk_text, citations_list, chunk_title)
        """
        chunks: List[Tuple[str, List[str], str]] = []
        current_words: List[str] = []
        current_citations: List[str] = []
        current_title: str = "Overview"

        for sec in doc.sections:
            sec_text = sec.content.strip()
            if not sec_text:
                continue

            sec_words = sec_text.split()
            sec_citation = sec.to_citation()

            if sec.element_type == "heading" and sec_text:
                current_title = sec_text

            if len(current_words) + len(sec_words) > self.target_chunk_words and current_words:
                chunk_str = " ".join(current_words)
                chunks.append((chunk_str, list(set(current_citations)), current_title))

                # Retain overlap from end of current chunk
                overlap_words = current_words[-self.chunk_overlap_words :] if len(current_words) > self.chunk_overlap_words else []
                current_words = list(overlap_words)
                current_citations = [sec_citation] if sec_citation else []

            current_words.extend(sec_words)
            if sec_citation and sec_citation not in current_citations:
                current_citations.append(sec_citation)

        if current_words:
            chunk_str = " ".join(current_words)
            chunks.append((chunk_str, list(set(current_citations)), current_title))

        return chunks

    def summarize(
        self,
        doc_or_text: ParsedDocument | str,
        focus: str = "general",
        max_words: int = 350,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> HierarchicalSummaryResult:
        """Executes full hierarchical summarization over any document length."""
        if isinstance(doc_or_text, str):
            doc = parse_text_to_structured("Input Document", doc_or_text)
        else:
            doc = doc_or_text

        total_words = doc.total_words
        if total_words == 0:
            return HierarchicalSummaryResult(
                final_summary="Document is empty.",
                executive_summary="Document is empty.",
                key_findings=[],
                section_breakdowns=[],
                total_source_words=0,
                summary_words=0,
                compression_ratio=0.0,
                tree_levels=0,
                citations=[],
            )

        if progress_callback:
            progress_callback("Analyzing document structure...", 0.05)

        # Fast path for small documents (< 1,800 words)
        if total_words <= 1800:
            if progress_callback:
                progress_callback("Generating direct high-fidelity synthesis...", 0.5)
            return self._summarize_direct(doc, focus=focus, max_words=max_words)

        # Stage 1: Partition into semantic chunks
        raw_chunks = self._chunk_document(doc)
        total_chunks = len(raw_chunks)
        if progress_callback:
            progress_callback(f"Partitioned into {total_chunks} structured sections.", 0.15)

        # Stage 2: Map phase - summarize each chunk
        level_0_nodes: List[ChunkSummaryNode] = []
        for idx, (chunk_text, citations, title) in enumerate(raw_chunks, start=1):
            if progress_callback:
                pct = 0.15 + (0.50 * (idx / total_chunks))
                progress_callback(f"Synthesizing section {idx}/{total_chunks}: {title}", pct)

            summary_text = self._map_chunk(chunk_text, title, citations, focus=focus)
            level_0_nodes.append(
                ChunkSummaryNode(
                    chunk_id=idx,
                    level=0,
                    title=title,
                    summary_text=summary_text,
                    citations=citations,
                    word_count=len(summary_text.split()),
                )
            )

        # Stage 3: Hierarchical Tree Reduction (RAPTOR-inspired clustering)
        current_level_nodes = level_0_nodes
        current_level = 1

        while len(current_level_nodes) > 4:
            if progress_callback:
                progress_callback(f"Tree Reduction: Merging level {current_level} nodes...", 0.70)
            next_level_nodes: List[ChunkSummaryNode] = []
            cluster_size = 3
            for c_idx in range(0, len(current_level_nodes), cluster_size):
                batch = current_level_nodes[c_idx : c_idx + cluster_size]
                merged_title = f"{batch[0].title} - {batch[-1].title}"
                all_citations = []
                for b in batch:
                    all_citations.extend(b.citations)
                all_citations = list(dict.fromkeys(all_citations))

                combined_text = "\n\n".join(
                    f"### {node.title} (Source: {', '.join(node.citations)})\n{node.summary_text}"
                    for node in batch
                )
                reduced_text = self._reduce_cluster(combined_text, merged_title, focus=focus)
                next_level_nodes.append(
                    ChunkSummaryNode(
                        chunk_id=len(next_level_nodes) + 1,
                        level=current_level,
                        title=merged_title,
                        summary_text=reduced_text,
                        citations=all_citations,
                        word_count=len(reduced_text.split()),
                    )
                )
            current_level_nodes = next_level_nodes
            current_level += 1

        # Stage 4: Final Executive Synthesis
        if progress_callback:
            progress_callback("Synthesizing final executive summary...", 0.85)

        all_citations: List[str] = []
        for node in level_0_nodes:
            all_citations.extend(node.citations)
        unique_citations = list(dict.fromkeys(all_citations))

        final_summary, exec_summary, key_findings = self._synthesize_final(
            nodes=current_level_nodes,
            doc_title=doc.filename,
            focus=focus,
            max_words=max_words,
            citations=unique_citations,
        )

        section_breakdowns = [
            {
                "section": n.title,
                "citations": n.citations,
                "summary": n.summary_text,
            }
            for n in level_0_nodes
        ]

        summary_words = len(final_summary.split())
        ratio = round(summary_words / max(1, total_words), 4)

        if progress_callback:
            progress_callback("Hierarchical synthesis complete.", 1.0)

        return HierarchicalSummaryResult(
            final_summary=final_summary,
            executive_summary=exec_summary,
            key_findings=key_findings,
            section_breakdowns=section_breakdowns,
            total_source_words=total_words,
            summary_words=summary_words,
            compression_ratio=ratio,
            tree_levels=current_level,
            citations=unique_citations,
            metadata={"total_chunks": total_chunks, "model": self.model, "provider": self.provider},
        )

    def _summarize_direct(
        self, doc: ParsedDocument, focus: str, max_words: int
    ) -> HierarchicalSummaryResult:
        """Direct single-pass synthesis for small documents."""
        system_prompt = (
            "You are an expert executive analyst. Produce a rigorous, highly structured summary "
            "strictly grounded in the provided document. Never introduce external unverified facts. "
            "Use clear Markdown formatting with an Executive Summary and Key Highlights."
        )
        user_prompt = (
            f"Document Title: {doc.filename}\n"
            f"Focus: {focus}\n"
            f"Target length: approximately {max_words} words.\n\n"
            f"--- SOURCE CONTENT ---\n{doc.get_full_text_with_citations()}\n\n"
            "Format your response exactly as:\n"
            "## Executive Summary\n<2-3 concise paragraphs>\n\n"
            "## Key Findings\n- <bullet 1 with citation>\n- <bullet 2 with citation>\n- <bullet 3 with citation>"
        )
        res = self._call_llm(system_prompt, user_prompt, max_tokens=max_words * 3)

        exec_summary, findings = self._extract_sections_from_llm(res)
        all_citations = [f"Page {p}" for p in sorted(doc.page_texts.keys())]

        return HierarchicalSummaryResult(
            final_summary=res,
            executive_summary=exec_summary,
            key_findings=findings,
            section_breakdowns=[],
            total_source_words=doc.total_words,
            summary_words=len(res.split()),
            compression_ratio=round(len(res.split()) / max(1, doc.total_words), 4),
            tree_levels=1,
            citations=all_citations,
            metadata={"mode": "direct", "model": self.model},
        )

    def _map_chunk(self, chunk_text: str, title: str, citations: List[str], focus: str) -> str:
        """Map phase: extracts dense factual propositions from an individual chunk."""
        cite_tag = f"[{', '.join(citations)}]" if citations else ""
        system_prompt = (
            "You are a factual condensation engine. Extract the core arguments, factual propositions, "
            "quantities, metrics, dates, and conclusions from the provided text block. "
            "Do not embellish or add speculative details. Keep it dense and informative (120-180 words)."
        )
        user_prompt = (
            f"Section Title: {title} {cite_tag}\n"
            f"Focus Angle: {focus}\n\n"
            f"Content:\n{chunk_text}\n\n"
            "Produce a factual summary of this section preserving specific metrics and source citations:"
        )
        return self._call_llm(system_prompt, user_prompt, max_tokens=350)

    def _reduce_cluster(self, combined_text: str, cluster_title: str, focus: str) -> str:
        """Reduce phase: aggregates intermediate summaries into a higher-level synthesis."""
        system_prompt = (
            "You are a synthesis specialist. Merge the following section summaries into a unified, "
            "logically coherent composite summary. Retain all quantitative data and structural citations."
        )
        user_prompt = (
            f"Cluster: {cluster_title}\n"
            f"Focus: {focus}\n\n"
            f"Intermediate Summaries:\n{combined_text}\n\n"
            "Produce a cohesive 200-word synthesis of these sections:"
        )
        return self._call_llm(system_prompt, user_prompt, max_tokens=400)

    def _synthesize_final(
        self,
        nodes: List[ChunkSummaryNode],
        doc_title: str,
        focus: str,
        max_words: int,
        citations: List[str],
    ) -> Tuple[str, str, List[str]]:
        """Synthesizes top-level reduced nodes into an overarching executive summary."""
        sections_body = "\n\n".join(
            f"### {n.title} (Source: {', '.join(n.citations) if n.citations else 'Document'})\n{n.summary_text}"
            for n in nodes
        )
        system_prompt = (
            "You are a principal document synthesis intelligence. Given the synthesized sections of a comprehensive document, "
            "create an executive brief that integrates all key themes, decisions, outcomes, and metrics without omitting critical nuances. "
            "All assertions must be faithful to the source sections."
        )
        user_prompt = (
            f"Document Title: {doc_title}\n"
            f"Desired Focus: {focus}\n"
            f"Target Word Limit: ~{max_words} words\n\n"
            f"--- SYNTHESIZED SECTIONS ---\n{sections_body}\n\n"
            "Structure your output strictly as follows:\n"
            "## Executive Summary\n"
            "<Comprehensive executive narrative highlighting core takeaways>\n\n"
            "## Key Findings\n"
            "- <Bullet point with specific metric/finding and citation>\n"
            "- <Bullet point with specific metric/finding and citation>\n"
            "- <Bullet point with specific metric/finding and citation>\n"
            "- <Bullet point with specific metric/finding and citation>\n"
        )
        full_res = self._call_llm(system_prompt, user_prompt, max_tokens=max_words * 3)
        exec_summary, findings = self._extract_sections_from_llm(full_res)
        return full_res, exec_summary, findings

    def _extract_sections_from_llm(self, text: str) -> Tuple[str, List[str]]:
        """Parses the generated response into executive summary narrative and key findings list."""
        exec_summary = ""
        findings: List[str] = []

        if "## Executive Summary" in text:
            parts = text.split("## Key Findings")
            exec_part = parts[0].replace("## Executive Summary", "").strip()
            exec_summary = exec_part
            if len(parts) > 1:
                findings_part = parts[1].strip()
                for line in findings_part.splitlines():
                    clean_line = re.sub(r"^[-*0-9.]+\s*", "", line).strip()
                    if clean_line:
                        findings.append(clean_line)
        else:
            exec_summary = text.strip()
            findings = [line.strip() for line in text.splitlines() if line.strip().startswith("-")]

        return exec_summary, findings
