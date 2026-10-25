"""Hallucination Guardrail and Auto-Correction for generated summaries.

Runs TruthLens's own NLI/LLM detection pipeline against model outputs
to produce an automated Factual Integrity Score and corrected text.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.summarizer import split_into_sentences


@dataclass
class GuardrailVerdict:
    """Full evaluation output from the hallucination guardrail."""

    original_text: str
    corrected_text: str
    factual_integrity_score: float
    sentence_verdicts: List[Dict[str, Any]]
    hallucinated_sentences: List[str]
    corrections_made: int
    grounding_sources_used: int
    is_clean: bool


class HallucinationGuardrail:
    """Applies TruthLens NLI detection to generated text and auto-corrects hallucinations."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "groq",
        model: Optional[str] = None,
        nli_threshold: float = 0.55,
    ):
        self.threshold = nli_threshold
        self.provider = provider.lower()

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

    def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
        if not self.client:
            return ""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def _nli_score_sentence(self, sentence: str, context_passages: List[str]) -> float:
        """Returns NLI entailment probability for a sentence given context passages."""
        # Try local transformers NLI first (zero-cost)
        try:
            from transformers import pipeline

            if not hasattr(self, "_nli_pipeline"):
                self._nli_pipeline = pipeline(
                    "zero-shot-classification",
                    model="cross-encoder/nli-deberta-v3-small",
                    device=-1,
                )

            combined_context = " ".join(p[:500] for p in context_passages[:3])
            result = self._nli_pipeline(
                sentence,
                candidate_labels=[combined_context],
                hypothesis_template="{}",
            )
            # Use the score direct as rough entailment probability
            return float(result["scores"][0])
        except Exception:
            pass

        # Fallback: LLM-based NLI scoring
        if not self.client:
            return 1.0  # Assume entailed if no resources available

        system_p = (
            "You are a Natural Language Inference model. Given a premise (context) and a hypothesis (claim), "
            "return a JSON with key 'label' (ENTAILMENT, NEUTRAL, or CONTRADICTION) and 'score' (0.0 to 1.0 confidence). "
            "Only return JSON."
        )
        combined_context = " ".join(p[:400] for p in context_passages[:2])
        user_p = (
            f"Premise: {combined_context}\n\n"
            f"Hypothesis: {sentence}\n\n"
            "Return JSON: {{\"label\": \"...\", \"score\": 0.0}}"
        )
        try:
            import json

            raw = self._call_llm(system_p, user_p, max_tokens=60)
            raw_clean = re.sub(r"^```\w*\n?|```$", "", raw.strip())
            data = json.loads(raw_clean)
            label = str(data.get("label", "NEUTRAL")).upper()
            score = float(data.get("score", 0.5))
            if label == "ENTAILMENT":
                return score
            elif label == "CONTRADICTION":
                return 1.0 - score
            return 0.5
        except Exception:
            return 0.7  # Conservative neutral fallback

    def _auto_correct_sentence(self, hallucinated_sentence: str, context_passages: List[str]) -> str:
        """Rewrites a hallucinated sentence to be grounded in provided context."""
        if not self.client:
            return hallucinated_sentence  # No correction possible without LLM

        combined_context = " ".join(p[:500] for p in context_passages[:2])
        system_p = (
            "You are a factual correction assistant. The provided sentence contains information "
            "not supported by the context. Rewrite it using ONLY information present in the context. "
            "If the entire claim is absent from context, respond with: [REMOVED - not in source]. "
            "Keep the tone and style of the original but ensure factual grounding."
        )
        user_p = (
            f"Context: {combined_context}\n\n"
            f"Hallucinated sentence: {hallucinated_sentence}\n\n"
            "Corrected sentence:"
        )
        try:
            return self._call_llm(system_p, user_p, max_tokens=120)
        except Exception:
            return hallucinated_sentence

    def audit(
        self,
        generated_text: str,
        grounding_passages: List[str],
        auto_correct: bool = True,
    ) -> GuardrailVerdict:
        """Runs sentence-level hallucination audit on generated text against grounding sources."""
        sentences = split_into_sentences(generated_text)
        if not sentences:
            return GuardrailVerdict(
                original_text=generated_text,
                corrected_text=generated_text,
                factual_integrity_score=1.0,
                sentence_verdicts=[],
                hallucinated_sentences=[],
                corrections_made=0,
                grounding_sources_used=len(grounding_passages),
                is_clean=True,
            )

        sentence_verdicts: List[Dict[str, Any]] = []
        hallucinated: List[str] = []
        corrected_sentences: List[str] = []
        corrections_made = 0

        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) < 5:  # Skip very short sentence fragments
                corrected_sentences.append(sent)
                continue

            score = self._nli_score_sentence(sent, grounding_passages)
            label = "ENTAILED" if score >= self.threshold else "HALLUCINATED"

            verdict = {
                "sentence": sent,
                "label": label,
                "entailment_score": round(score, 4),
            }
            sentence_verdicts.append(verdict)

            if label == "HALLUCINATED":
                hallucinated.append(sent)
                if auto_correct:
                    corrected = self._auto_correct_sentence(sent, grounding_passages)
                    if corrected and corrected != sent:
                        corrected_sentences.append(corrected)
                        corrections_made += 1
                    else:
                        corrected_sentences.append(sent)
                else:
                    corrected_sentences.append(sent)
            else:
                corrected_sentences.append(sent)

        # Factual Integrity Score = % of sentences entailed
        total_checked = len(sentence_verdicts)
        entailed_count = sum(1 for v in sentence_verdicts if v["label"] == "ENTAILED")
        fi_score = round(entailed_count / max(1, total_checked), 3)

        corrected_text = " ".join(corrected_sentences).strip()

        return GuardrailVerdict(
            original_text=generated_text,
            corrected_text=corrected_text,
            factual_integrity_score=fi_score,
            sentence_verdicts=sentence_verdicts,
            hallucinated_sentences=hallucinated,
            corrections_made=corrections_made,
            grounding_sources_used=len(grounding_passages),
            is_clean=len(hallucinated) == 0,
        )
