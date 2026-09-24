"""
dataset.py — FRANK benchmark loader and label aggregation.

Loads human_annotations_sentence.json and converts multi-annotator
error tags into a 3-class NLI label via majority/priority voting.

Labels:
    0 = Contradiction  (Intrinsic hallucination)
    1 = Neutral        (Extrinsic hallucination)
    2 = Entailment     (Factually consistent)
"""

import json
import collections
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Error-tag taxonomy
# ---------------------------------------------------------------------------
INTRINSIC_TAGS = {"EntE", "CircE", "RelE", "CorefE", "LinkE", "OtherE"}
EXTRINSIC_TAGS = {"OutE"}
LABEL_NAMES = {0: "Contradiction", 1: "Neutral", 2: "Entailment"}


# ---------------------------------------------------------------------------
# Voting helpers
# ---------------------------------------------------------------------------

def _vote_for_annotator(tags: List[str]) -> Optional[int]:
    """Map a single annotator's tag list → NLI label (or None for GramE-only)."""
    if any(t in INTRINSIC_TAGS for t in tags):
        return 0  # contradiction / intrinsic
    if any(t in EXTRINSIC_TAGS for t in tags):
        return 1  # neutral / extrinsic
    if "NoE" in tags:
        return 2  # entailment / factual
    return None   # e.g. only GramE


def aggregate_annotation(sentence_annotation: Dict[str, List[str]]) -> Optional[int]:
    """Majority vote across annotators; ties → stricter (lower) label."""
    votes = []
    for _key, tags in sentence_annotation.items():
        v = _vote_for_annotator(tags)
        if v is not None:
            votes.append(v)
    if not votes:
        return None
    counter = collections.Counter(votes)
    max_count = max(counter.values())
    # on tie pick the stricter (lower numerical) label
    candidates = sorted(k for k, v in counter.items() if v == max_count)
    return candidates[0]


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class FrankDataset:
    """
    Iterable dataset of (article, summary_sentence, label) triples.

    Parameters
    ----------
    json_path : str | Path
        Path to ``human_annotations_sentence.json``.
    split : str | None
        ``'train'``, ``'valid'``, ``'test'``, or ``None`` for all.
    """

    def __init__(self, json_path: str | Path, split: Optional[str] = None):
        json_path = Path(json_path)
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)

        self.records: List[Dict] = []
        for rec in raw:
            if split and rec.get("split") != split:
                continue
            sents = rec.get("summary_sentences", [])
            annots = rec.get("summary_sentences_annotations", [])
            for sent, ann in zip(sents, annots):
                label = aggregate_annotation(ann)
                if label is None:
                    continue
                self.records.append({
                    "hash": rec["hash"],
                    "model_name": rec.get("model_name", ""),
                    "article": rec["article"],
                    "summary": rec.get("summary", ""),
                    "sentence": sent,
                    "label": label,
                    "split": rec.get("split", ""),
                })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return self.records[idx]

    def label_distribution(self) -> Dict[str, int]:
        c = collections.Counter(r["label"] for r in self.records)
        return {LABEL_NAMES[k]: v for k, v in sorted(c.items())}


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/human_annotations_sentence.json"
    for sp in (None, "valid", "test"):
        ds = FrankDataset(path, split=sp)
        print(f"split={sp!s:>5}  size={len(ds):>5}  dist={ds.label_distribution()}")
