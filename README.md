# TruthLens — AI Hallucination Detector

A sentence-level hallucination detection tool built on the **FRANK** benchmark dataset.  
TruthLens classifies AI-generated summary sentences into three categories:

| Label | Meaning |
|---|---|
| **Entailment** ✅ | Factually consistent with the source |
| **Contradiction** ❌ | Intrinsic hallucination — contradicts the source |
| **Neutral** ⚠️ | Extrinsic hallucination — information not found in source |

## Architecture

```
Article → [Chunker] → [Retriever (TF-IDF / Embeddings)]
                              ↓
Summary → [Sentence Splitter] → [NLI Classifier / Heuristic]
                              ↓
                     Per-sentence Verdict
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Interactive mode (TF-IDF heuristic — fast, no GPU needed)
python cli.py

# Interactive mode with NLI model (slower, more accurate)
python cli.py --backend nli

# Evaluate on FRANK benchmark
python -m src.evaluate --data data/human_annotations_sentence.json --split test --backend tfidf
python -m src.evaluate --data data/human_annotations_sentence.json --split test --backend nli --limit 100
```

## Project Structure

```
TruthLens/
├── cli.py                  # Interactive CLI / File mode
├── requirements.txt
├── data/
│   └── human_annotations_sentence.json   # FRANK benchmark
└── src/
    ├── __init__.py
    ├── dataset.py           # FRANK loader + label aggregation
    ├── retriever.py         # TF-IDF / Embedding retriever
    ├── detector.py          # NLI + heuristic hallucination detector
    └── evaluate.py          # Evaluation suite (accuracy, F1, confusion matrix)
```

## Dataset

Uses the [FRANK Benchmark](https://github.com/artidoro/frank) — a human-annotated  
dataset of 2,246 article–summary pairs with sentence-level factuality annotations  
across 9 summarization models. Error taxonomy includes:

- **Intrinsic errors**: EntE, CircE, RelE, CorefE, LinkE, OtherE
- **Extrinsic errors**: OutE
- **No error**: NoE

## Detection Backends

| Backend | Speed | Quality | GPU Required |
|---|---|---|---|
| `tfidf` | ⚡ Fast | Baseline | No |
| `nli` | 🐢 Slower | High | Optional |
