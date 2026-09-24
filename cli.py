"""
cli.py — Interactive command-line interface for TruthLens.

Usage
-----
    python cli.py                      # interactive REPL mode
    python cli.py --article FILE --summary FILE
    python cli.py --backend nli        # use NLI model (needs GPU or patience)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

from src.detector import get_detector, DetectionResult, LABEL_MAP

# ---------------------------------------------------------------------------
# Sentence splitter (reuse logic from retriever)
# ---------------------------------------------------------------------------
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')


def split_into_sentences(text: str):
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

_COLORS = {
    0: "\033[91m",  # red   — contradiction
    1: "\033[93m",  # yellow — neutral
    2: "\033[92m",  # green — entailment
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def print_result(result: DetectionResult) -> None:
    print()
    print(f"{_BOLD}{'═' * 60}")
    print(f"  TruthLens — Sentence-Level Hallucination Report")
    print(f"{'═' * 60}{_RESET}")
    print()

    for i, v in enumerate(result.verdicts, 1):
        color = _COLORS.get(v.label, "")
        tag = f"[{v.label_name.upper()}]"
        print(f"  {_BOLD}Sentence {i}:{_RESET}")
        wrapped = textwrap.fill(v.sentence, width=70, initial_indent="    ", subsequent_indent="    ")
        print(wrapped)
        print(f"    {color}{tag}{_RESET}  confidence={v.confidence:.3f}")
        print(f"    → {v.explanation}")
        print(f"    Evidence (sim={v.evidence_score:.3f}): {v.evidence_chunk[:120]}…")
        print()

    overall_color = _COLORS.get(result.overall_label, "")
    print(f"  {_BOLD}Overall verdict:{_RESET} "
          f"{overall_color}{result.overall_label_name.upper()}{_RESET}")
    print(f"{'═' * 60}")
    print()


# ---------------------------------------------------------------------------
# REPL mode
# ---------------------------------------------------------------------------

def repl(detector) -> None:
    print(f"\n{_BOLD}TruthLens — Interactive Hallucination Detector{_RESET}")
    print("Type 'quit' or 'exit' to leave.\n")

    while True:
        try:
            print(f"{_BOLD}Paste the SOURCE article (end with an empty line):{_RESET}")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            article = " ".join(lines).strip()
            if not article or article.lower() in ("quit", "exit"):
                break

            print(f"\n{_BOLD}Paste the SUMMARY to check (end with an empty line):{_RESET}")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            summary = " ".join(lines).strip()
            if not summary or summary.lower() in ("quit", "exit"):
                break

            sentences = split_into_sentences(summary)
            if not sentences:
                sentences = [summary]

            result = detector.detect(article, sentences)
            print_result(result)

        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break


# ---------------------------------------------------------------------------
# File mode
# ---------------------------------------------------------------------------

def file_mode(detector, article_path: str, summary_path: str) -> None:
    article = Path(article_path).read_text(encoding="utf-8").strip()
    summary = Path(summary_path).read_text(encoding="utf-8").strip()
    sentences = split_into_sentences(summary)
    if not sentences:
        sentences = [summary]
    result = detector.detect(article, sentences)
    print_result(result)
    # Also print JSON
    print(json.dumps(result.summary_dict(), indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TruthLens — AI Hallucination Detector CLI",
    )
    parser.add_argument("--backend", default="tfidf", choices=["nli", "tfidf", "llm"],
                        help="Detection backend (default: tfidf for quick demo)")
    parser.add_argument("--model", default="roberta-large-mnli",
                        help="HuggingFace NLI model name (for 'nli' backend)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--article", default=None, help="Path to article text file")
    parser.add_argument("--summary", default=None, help="Path to summary text file")
    args = parser.parse_args()

    detector = get_detector(
        backend=args.backend,
        model_name=args.model,
        device=args.device,
    )

    if args.article and args.summary:
        file_mode(detector, args.article, args.summary)
    else:
        repl(detector)


if __name__ == "__main__":
    main()
