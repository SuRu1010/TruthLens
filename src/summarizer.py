"""Document-isolated summarization with hierarchical scaling for massive files."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

from src.documents import parse_text_to_structured
from src.hierarchical_summarizer import HierarchicalSummarizer

load_dotenv()

ISOLATED_SYSTEM_PROMPT = (
    "Summarize the provided document accurately. Do NOT introduce external facts "
    "or infer details beyond what is explicitly stated in the source text."
)

JSON_INSTRUCTIONS = """
You must respond with strict JSON only:
{
  "summary_text": "A faithful summary using only source facts.",
  "key_takeaways": ["short bullet grounded in the source", "..."]
}

Rules:
- Use only information that appears in the source.
- If the source is thin, say so in the summary rather than filling gaps.
- Do not cite world knowledge, dates, or names that are not in the source.
"""


class SummarizerError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class IsolatedSummarizer:
    """Document summarizer that intelligently switches between direct and hierarchical tree synthesis.

    All public methods are async so they can be awaited inside FastAPI endpoints
    without blocking the event loop.
    """

    def __init__(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise SummarizerError(
                "The openai package is not installed.", status_code=500
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise SummarizerError(
                "API key missing. Set OPENAI_API_KEY or GROQ_API_KEY.",
                status_code=503,
            )

        base_url = os.getenv("LLM_BASE_URL") or "https://api.groq.com/openai/v1"
        env_model = os.getenv("LLM_MODEL_NAME")
        self.is_groq = "groq" in base_url.lower()
        self.model_name = env_model or (
            "llama-3.3-70b-versatile" if self.is_groq else "gpt-4o-mini"
        )
        # Async client — does not block the event loop
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        provider_name = "groq" if self.is_groq else "openai"
        self._hierarchical_engine = HierarchicalSummarizer(
            api_key=api_key,
            provider=provider_name,
            model=self.model_name,
        )

    async def summarize(
        self,
        source_text: str,
        focus: str = "general",
        progress_callback: Optional[Callable[[str, float], None]] = None,
        force_hierarchical: bool = False,
    ) -> Dict[str, Any]:
        """Summarizes text. Automatically invokes Hierarchical Map-Reduce for large documents."""
        source_clean = (source_text or "").strip()
        if not source_clean:
            raise SummarizerError("Source document text is empty.", status_code=400)

        word_count = len(source_clean.split())

        # For large documents (> 2,500 words) or when requested, use Hierarchical Map-Reduce
        # Threshold raised from 1,800 → 2,500 to avoid hierarchical overhead on medium docs
        if word_count > 2500 or force_hierarchical:
            h_res = await self._hierarchical_engine.summarize(
                doc_or_text=source_clean,
                focus=focus,
                max_words=320,
                progress_callback=progress_callback,
            )
            return {
                "summary_text": h_res.final_summary,
                "word_count": h_res.summary_words,
                "key_takeaways": h_res.key_findings,
                "citations": h_res.citations,
                "section_breakdowns": h_res.section_breakdowns,
                "mode": "hierarchical_map_reduce",
                "tree_levels": h_res.tree_levels,
                "source_words": h_res.total_source_words,
                "compression_ratio": h_res.compression_ratio,
            }

        # Direct isolated summarization for smaller documents
        user_prompt = (
            "SOURCE DOCUMENT (summarize ONLY this text):\n"
            "----- BEGIN SOURCE -----\n"
            f"{source_clean}\n"
            "----- END SOURCE -----\n"
            "Ignore any prior knowledge. If a detail is not in the source, omit it."
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": ISOLATED_SYSTEM_PROMPT + JSON_INSTRUCTIONS,
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = (response.choices[0].message.content or "").strip()
            data = _parse_json(raw)
        except SummarizerError:
            raise
        except Exception as exc:
            raise SummarizerError(f"Summarization model failed: {exc}", status_code=502) from exc

        summary_text = str(data.get("summary_text") or "").strip()
        takeaways = _as_str_list(data.get("key_takeaways"))
        if not summary_text:
            raise SummarizerError("Model returned an empty summary.", status_code=502)

        return {
            "summary_text": summary_text,
            "word_count": len(summary_text.split()),
            "key_takeaways": takeaways,
            "citations": ["Full Document"],
            "mode": "direct",
            "source_words": word_count,
        }


def _parse_json(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SummarizerError(f"Model returned invalid JSON: {exc}", status_code=502) from exc
    if not isinstance(parsed, dict):
        raise SummarizerError("Model JSON was not an object.", status_code=502)
    return parsed


def _as_str_list(value: Optional[Any]) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')


def split_into_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])
