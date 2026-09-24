"""
detector.py — Hallucination detection engine.

Combines the retriever with an NLI classifier or LLM Judge to produce
per-sentence hallucination verdicts.

Supported backends
------------------
* ``nli``   – Use a HuggingFace NLI model (default: roberta-large-mnli).
* ``tfidf`` – Heuristic baseline using TF-IDF similarity.
* ``llm``   – Generative LLM-as-a-Judge (Groq / OpenAI / Ollama).
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from dotenv import load_dotenv
from src.retriever import TfidfRetriever, get_retriever

# Auto-load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

LABEL_MAP = {0: "Contradiction", 1: "Neutral", 2: "Entailment"}
LABEL_EXPLAIN = {
    0: "Intrinsic hallucination — contradicts source.",
    1: "Extrinsic hallucination — unsupported by source.",
    2: "Factually consistent with source.",
}


@dataclass
class SentenceVerdict:
    """Result for one summary sentence."""
    sentence: str
    label: int
    label_name: str
    confidence: float
    explanation: str
    evidence_chunk: str
    evidence_score: float


@dataclass
class DetectionResult:
    """Container for a full summary analysis."""
    verdicts: List[SentenceVerdict] = field(default_factory=list)

    @property
    def overall_label(self) -> int:
        if not self.verdicts:
            return 2
        return min(v.label for v in self.verdicts)

    @property
    def overall_label_name(self) -> str:
        return LABEL_MAP.get(self.overall_label, "Unknown")

    def summary_dict(self) -> Dict:
        return {
            "overall": self.overall_label_name,
            "sentences": [
                {
                    "sentence": v.sentence,
                    "label": v.label_name,
                    "confidence": round(v.confidence, 4),
                    "explanation": v.explanation,
                    "evidence": v.evidence_chunk[:200],
                }
                for v in self.verdicts
            ],
        }


# ---------------------------------------------------------------------------
# NLI-based detector
# ---------------------------------------------------------------------------

class NLIDetector:
    """
    Uses a HuggingFace NLI model (e.g. roberta-large-mnli) to classify
    (premise=article_chunk, hypothesis=summary_sentence) pairs.
    """

    NLI_LABEL_TO_OURS = {
        "CONTRADICTION": 0,
        "NEUTRAL": 1,
        "ENTAILMENT": 2,
        "contradiction": 0,
        "neutral": 1,
        "entailment": 2,
    }

    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        device: str = "cpu",
        use_embedding_retriever: bool = False,
    ):
        from transformers import pipeline as hf_pipeline

        self.nli = hf_pipeline(
            "text-classification",
            model=model_name,
            device=-1 if device == "cpu" else 0,
        )
        self.retriever = get_retriever(use_embeddings=use_embedding_retriever)

    def detect(
        self,
        article: str,
        summary_sentences: List[str],
        top_k: int = 3,
    ) -> DetectionResult:
        result = DetectionResult()
        for sent in summary_sentences:
            retrieved = self.retriever.retrieve(article, sent, top_k=top_k)
            premise = " ".join(chunk for chunk, _ in retrieved)
            best_chunk, best_score = retrieved[0]

            out = self.nli({"text": premise, "text_pair": sent})
            top_label_str = out["label"]
            top_score = out["score"]
            label_int = self.NLI_LABEL_TO_OURS.get(top_label_str, 1)

            result.verdicts.append(
                SentenceVerdict(
                    sentence=sent,
                    label=label_int,
                    label_name=LABEL_MAP[label_int],
                    confidence=top_score,
                    explanation=LABEL_EXPLAIN[label_int],
                    evidence_chunk=best_chunk,
                    evidence_score=best_score,
                )
            )
        return result


# ---------------------------------------------------------------------------
# Heuristic / TF-IDF-only detector
# ---------------------------------------------------------------------------

class TfidfHeuristicDetector:
    """
    Simple heuristic fallback using similarity scores.
    """

    def __init__(
        self,
        entailment_threshold: float = 0.25,
        neutral_threshold: float = 0.10,
    ):
        self.ent_thresh = entailment_threshold
        self.neu_thresh = neutral_threshold
        self.retriever = TfidfRetriever()

    def detect(
        self,
        article: str,
        summary_sentences: List[str],
        top_k: int = 3,
    ) -> DetectionResult:
        result = DetectionResult()
        for sent in summary_sentences:
            retrieved = self.retriever.retrieve(article, sent, top_k=top_k)
            best_chunk, best_score = retrieved[0]

            if best_score >= self.ent_thresh:
                label = 2
                conf = min(best_score * 2, 1.0)
            elif best_score >= self.neu_thresh:
                label = 0
                conf = 0.5
            else:
                label = 1
                conf = 1.0 - best_score

            result.verdicts.append(
                SentenceVerdict(
                    sentence=sent,
                    label=label,
                    label_name=LABEL_MAP[label],
                    confidence=conf,
                    explanation=LABEL_EXPLAIN[label],
                    evidence_chunk=best_chunk,
                    evidence_score=best_score,
                )
            )
        return result


# ---------------------------------------------------------------------------
# LLM-as-a-Judge Detector
# ---------------------------------------------------------------------------

class LLMDetector:
    """
    Uses a Generative LLM (Groq / OpenAI / Ollama) to judge factual consistency.
    """

    LABEL_TO_INT = {
        "CONTRADICTION": 0,
        "NEUTRAL": 1,
        "ENTAILMENT": 2,
    }

    SYSTEM_PROMPT = """You are an expert factual integrity and hallucination evaluator.
Compare the SUMMARY SENTENCE against the SOURCE EVIDENCE.

Determine which category best fits:
- ENTAILMENT: The summary sentence is completely supported by and factually accurate according to the source.
- CONTRADICTION: The summary sentence DIRECTLY conflicts with or contradicts facts explicitly stated in the source (Intrinsic hallucination).
- NEUTRAL: The summary sentence contains external, missing, or unverified claims NOT mentioned in the source (Extrinsic hallucination).

CRITICAL DISAMBIGUATION RULES:
1. If details or facts in the summary sentence are MISSING, UNMENTIONED, or ABSENT from the source evidence, classify as NEUTRAL, NOT CONTRADICTION.
2. Only classify as CONTRADICTION if there is an explicit, direct factual clash (e.g., source says "red", summary says "blue").

Respond strict raw JSON only:
{
  "verdict": "ENTAILMENT" | "CONTRADICTION" | "NEUTRAL",
  "confidence": 0.0 to 1.0,
  "explanation": "Concise 1-sentence explanation of your verdict."
}"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_embedding_retriever: bool = False,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")

        # 1. API Key resolution
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key found. Set OPENAI_API_KEY in your environment or .env file."
            )

        # 2. Base URL resolution (Default to Groq)
        base_url = os.getenv("LLM_BASE_URL") or "https://api.groq.com/openai/v1"

        # 3. Model Name resolution precedence:
        # env var > passed model_name (if not default NLI name) > default Groq/OpenAI model
        env_model = os.getenv("LLM_MODEL_NAME")
        if env_model:
            self.model_name = env_model
        elif model_name and model_name not in ["roberta-large-mnli", "gpt-4o-mini"]:
            self.model_name = model_name
        else:
            self.model_name = "qwen/qwen3.8-27b" if "groq" in base_url.lower() else "gpt-4o-mini"

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.retriever = get_retriever(use_embeddings=use_embedding_retriever)

    def detect(
        self,
        article: str,
        summary_sentences: List[str],
        top_k: int = 3,
    ) -> DetectionResult:
        result = DetectionResult()

        for sent in summary_sentences:
            retrieved = self.retriever.retrieve(article, sent, top_k=top_k)
            premise = " ".join(chunk for chunk, _ in retrieved)
            best_chunk, best_score = retrieved[0]

            user_prompt = f"SOURCE EVIDENCE:\n{premise}\n\nSUMMARY SENTENCE:\n{sent}"

            label_int = 2
            confidence = 1.0
            explanation = LABEL_EXPLAIN[2]

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                raw_text = response.choices[0].message.content.strip()

                # Clean markdown block wrapping if present
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:].strip()

                data = json.loads(raw_text)

                verdict_str = str(data.get("verdict", "ENTAILMENT")).strip().upper()
                label_int = self.LABEL_TO_INT.get(verdict_str, 2)
                confidence = float(data.get("confidence", 0.95))
                explanation = str(data.get("explanation", LABEL_EXPLAIN[label_int]))

            except Exception as e:
                warnings.warn(f"LLM API Call failed ({e}). Falling back to ENTAILMENT.")

            result.verdicts.append(
                SentenceVerdict(
                    sentence=sent,
                    label=label_int,
                    label_name=LABEL_MAP[label_int],
                    confidence=confidence,
                    explanation=f"[LLM-as-a-Judge] {explanation}",
                    evidence_chunk=best_chunk,
                    evidence_score=best_score,
                )
            )

        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_detector(
    backend: str = "nli",
    model_name: str = "roberta-large-mnli",
    device: str = "cpu",
    use_embedding_retriever: bool = False,
):
    """
    Factory function for detectors.
    backend choices: 'nli', 'tfidf', or 'llm'
    """
    if backend == "nli":
        return NLIDetector(model_name, device, use_embedding_retriever)
    elif backend == "tfidf":
        return TfidfHeuristicDetector()
    elif backend == "llm":
        # Let LLMDetector resolve model_name from LLM_MODEL_NAME env var or defaults
        llm_model = None if model_name == "roberta-large-mnli" else model_name
        return LLMDetector(llm_model, use_embedding_retriever)
    else:
        raise ValueError(f"Unknown backend {backend!r}, choose 'nli', 'tfidf', or 'llm'")