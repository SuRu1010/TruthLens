"""
evaluate.py — Evaluation suite for the hallucination detector.

Runs the detector on the FRANK benchmark test (or valid) split and
reports per-class precision, recall, F1, confusion matrix, and accuracy.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from tqdm import tqdm

from src.dataset import FrankDataset, LABEL_NAMES
from src.detector import get_detector, DetectionResult


# ---------------------------------------------------------------------------
# Run evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    dataset_path: str | Path,
    split: str = "test",
    backend: str = "tfidf",
    model_name: str = "roberta-large-mnli",
    device: str = "cpu",
    limit: int | None = None,
    top_k: int = 3,
) -> Dict:
    """
    Evaluate the detector on the FRANK dataset.

    Returns
    -------
    dict with keys: accuracy, macro_f1, report (str), confusion_matrix,
    y_true, y_pred, elapsed seconds.
    """
    ds = FrankDataset(dataset_path, split=split)
    if not ds.records:
        raise RuntimeError(f"No records found for split={split!r}")

    detector = get_detector(backend=backend, model_name=model_name, device=device)

    y_true: List[int] = []
    y_pred: List[int] = []
    errors: List[Dict] = []

    records = ds.records[:limit] if limit else ds.records
    start = time.time()

    for rec in tqdm(records, desc=f"Evaluating ({backend})"):
        article = rec["article"]
        sentence = rec["sentence"]
        gold_label = rec["label"]

        result: DetectionResult = detector.detect(article, [sentence], top_k=top_k)
        pred_label = result.verdicts[0].label

        y_true.append(gold_label)
        y_pred.append(pred_label)

        if pred_label != gold_label:
            errors.append({
                "sentence": sentence,
                "gold": LABEL_NAMES[gold_label],
                "pred": LABEL_NAMES[pred_label],
                "confidence": result.verdicts[0].confidence,
            })

    elapsed = time.time() - start

    labels_list = sorted(LABEL_NAMES.keys())
    target_names = [LABEL_NAMES[i] for i in labels_list]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_list,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(y_true),
        "num_errors": len(errors),
        "sample_errors": errors[:10],
        "elapsed_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------

def print_results(results: Dict) -> None:
    print("=" * 60)
    print("  TruthLens — Hallucination Detection Evaluation")
    print("=" * 60)
    print(f"  Samples evaluated : {results['num_samples']}")
    print(f"  Overall accuracy  : {results['accuracy']:.2%}")
    print(f"  Macro F1 score    : {results['macro_f1']:.4f}")
    print(f"  Time elapsed      : {results['elapsed_seconds']:.1f}s")
    print("-" * 60)
    print(results["report"])
    print("-" * 60)
    cm = np.array(results["confusion_matrix"])
    labels = ["Contra", "Neutral", "Entail"]
    print("Confusion Matrix (rows=true, cols=pred):")
    header = "          " + "  ".join(f"{l:>8}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        vals = "  ".join(f"{v:>8}" for v in row)
        print(f"  {labels[i]:>7}  {vals}")
    print("=" * 60)

    if results["sample_errors"]:
        print("\nSample Misclassifications:")
        for e in results["sample_errors"][:5]:
            print(f"  GOLD={e['gold']:>14}  PRED={e['pred']:>14}  "
                  f"conf={e['confidence']:.3f}  | {e['sentence'][:80]}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate TruthLens on FRANK")
    parser.add_argument("--data", default="data/human_annotations_sentence.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--backend", default="tfidf", choices=["nli", "tfidf", "llm"], help="Detection backend (nli, tfidf, or llm)")
    parser.add_argument("--model", default="roberta-large-mnli")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--save", default=None, help="Save results JSON to path")
    args = parser.parse_args()

    results = evaluate(
        dataset_path=args.data,
        split=args.split,
        backend=args.backend,
        model_name=args.model,
        device=args.device,
        limit=args.limit,
        top_k=args.top_k,
    )
    print_results(results)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # numpy int64 is not JSON-serializable; convert
        out = {k: v for k, v in results.items() if k not in ("y_true", "y_pred")}
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved → {save_path}")