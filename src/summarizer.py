"""Document-isolated summarization — no external knowledge."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

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
    def __init__(self) -> None:
        try:
            from openai import OpenAI
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
        self.model_name = env_model or (
            "qwen/qwen3.8-27b" if "groq" in base_url.lower() else "gpt-4o-mini"
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def summarize(self, source_text: str) -> Dict[str, Any]:
        truncated = source_text[:24000]
        user_prompt = (
            "SOURCE DOCUMENT (summarize ONLY this text):\n"
            "----- BEGIN SOURCE -----\n"
            f"{truncated}\n"
            "----- END SOURCE -----\n"
            "Ignore any prior knowledge. If a detail is not in the source, omit it."
        )

        try:
            response = self.client.chat.completions.create(
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
