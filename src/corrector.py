"""
corrector.py — Auto-correction engine for hallucinated sentences in TruthLens.

Takes sentences flagged as Contradiction or Neutral and rewrites them
to be 100% factually entailed by the source context.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from groq import Groq


@dataclass
class CorrectionResult:
    original: str
    corrected: str
    label: str
    explanation: str


class HallucinationCorrector:
    """Uses Groq LLM to rewrite hallucinated sentences into grounded entailments."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "API key not found. Please set GROQ_API_KEY or OPENAI_API_KEY."
            )

        self.client = Groq(api_key=api_key)
        self.model_name = (
            model_name
            or os.getenv("LLM_MODEL_NAME")
            or "qwen/qwen3.8-27b"
        )

    def correct_sentence(
        self, article: str, sentence: str, label: str
    ) -> CorrectionResult:
        """
        Rewrites a single hallucinated sentence based on source context.
        """
        system_prompt = (
            "You are an expert factual editor for news summaries.\n"
            "Your task is to rewrite a target sentence flagged for hallucination "
            "so that it becomes 100% factually accurate according to the source article.\n\n"
            "Rules:\n"
            "1. Remove or fix any unverified facts, incorrect entities, or contradictions.\n"
            "2. Keep as much of the original sentence structure as possible while remaining strictly true to the text.\n"
            "3. Do NOT add new unverified details.\n"
            "4. Output JSON strictly with keys: 'corrected' and 'explanation'."
        )

        user_prompt = f"""Source Article:
{article}

Flagged Sentence: "{sentence}"
Flagged Reason: {label}

Provide the corrected version and a brief explanation of what was changed.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            res = json.loads(response.choices[0].message.content)
            return CorrectionResult(
                original=sentence,
                corrected=res.get("corrected", sentence),
                label=label,
                explanation=res.get("explanation", "Rewritten for factual consistency."),
            )

        except Exception as e:
            return CorrectionResult(
                original=sentence,
                corrected=sentence,
                label=label,
                explanation=f"Correction failed: {e}",
            )

    def correct_batch(
        self, article: str, flagged_items: List[Dict[str, str]]
    ) -> List[CorrectionResult]:
        """
        Batch-corrects multiple flagged sentences.
        flagged_items: List of dicts with 'sentence' and 'label'
        """
        return [
            self.correct_sentence(article, item["sentence"], item["label"])
            for item in flagged_items
        ]


# ---------------------------------------------------------------------------
# CLI Test Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_article = (
        "The Metropolitan Police have launched an investigation following a robbery "
        "at a jewelry store in central London on Tuesday afternoon. Two suspects "
        "escaped on motorbikes with high-value watches."
    )
    sample_hallucination = (
        "The FBI has launched an investigation into a bank heist in New York where three armed men stole gold."
    )

    print("--- Running Quick Fix Auto-Correction Test ---")
    corrector = HallucinationCorrector()
    result = corrector.correct_sentence(
        article=sample_article,
        sentence=sample_hallucination,
        label="Contradiction",
    )

    print(f"Original  : {result.original}")
    print(f"Corrected : {result.corrected}")
    print(f"Reason    : {result.explanation}")
    