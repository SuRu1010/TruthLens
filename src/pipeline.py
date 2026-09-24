"""API-facing verification wrapper around existing detectors."""

from __future__ import annotations

import os
from typing import Any, Dict

from dotenv import load_dotenv

from src.detector import DetectionResult, get_detector
from src.summarizer import split_into_sentences

load_dotenv()

_detector = None


def get_app_detector():
    global _detector
    if _detector is not None:
        return _detector

    backend = os.getenv("DETECTOR_BACKEND", "llm")
    try:
        _detector = get_detector(backend=backend)
    except Exception:
        _detector = get_detector(backend="tfidf")
    return _detector


def verify_claim(claim: str, reference: str) -> Dict[str, Any]:
    claim = (claim or "").strip()
    reference = (reference or "").strip()

    if not claim:
        raise ValueError("Claim text is required.")

    if not reference:
        return {
            "verdict": "NEUTRAL",
            "confidence_score": 0.32,
            "summary": (
                "No reference source was provided. The claim cannot be grounded "
                "in a document, so it is treated as unsupported (extrinsic)."
            ),
            "key_factors": [
                "Missing reference text",
                "Missing reference file",
                "Verification requires an isolated source document",
            ],
        }

    sentences = split_into_sentences(claim)
    result: DetectionResult = get_app_detector().detect(reference, sentences)
    overall = result.overall_label_name.upper()
    confidences = [v.confidence for v in result.verdicts] or [0.0]
    confidence = sum(confidences) / len(confidences)

    factors = []
    for v in result.verdicts[:6]:
        factors.append(f"{v.label_name}: {v.explanation}")
    if result.verdicts:
        factors.append(f"Strongest evidence snippet: {result.verdicts[0].evidence_chunk[:180]}")

    summary = (
        f"Overall verdict {overall} across {len(result.verdicts)} claim sentence(s). "
        + (result.verdicts[0].explanation if result.verdicts else "")
    )

    return {
        "verdict": overall,
        "confidence_score": round(float(confidence), 4),
        "summary": summary.strip(),
        "key_factors": factors or ["No sentence-level factors produced."],
    }
