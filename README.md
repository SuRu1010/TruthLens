```markdown
# 🔍 TruthLens: Advanced Level 3/4 Agentic RAG & Hallucination Guardrail

TruthLens is an enterprise-grade, agentic Retrieval-Augmented Generation (RAG) and hallucination verification system designed to extract insights from massive documents, eliminate factual drift, and enforce strict source grounding using hybrid search, multi-intent routing, and self-correction loops.

---

## ✨ Core Architecture & Features

- **Multi-Intent Agentic Router (`src/agent.py`)**: Automatically classifies user intent into `GLOBAL_SUMMARY`, `FACT_CHECK`, or `SPECIFIC_QA`, routing requests to specialized processing pathways.
- **Hierarchical Map-Reduce Summarizer (`src/hierarchical_summarizer.py`)**: Solves token-limit bottlenecks for 50–200+ page PDFs/DOCXs using tree-based chunk synthesis (RAPTOR-inspired clustering) without truncation.
- **Hybrid Dense + Sparse Search (`src/hybrid_search.py`)**: Combines Dense Vector Embeddings and Okapi BM25 keyword matching via **Reciprocal Rank Fusion (RRF)** and Cross-Encoder re-ranking.
- **HyDE Query Expansion (`src/agent.py`)**: Generates hypothetical document passages to significantly boost retrieval recall for complex questions.
- **Sentence-Level Guardrail & Self-RAG (`src/guardrails.py`, `src/corrector.py`)**: Audits generated text sentence-by-sentence against source contexts, computing a Factual Integrity Score (0–1) and auto-correcting hallucinations.

---

## 🛠️ Project Structure

```text
TruthLens/
├── main.py                  # FastAPI backend server (v2 endpoints)
├── requirements.txt         # Project dependencies including rank-bm25
├── cli.py                   # Interactive command-line interface
├── data/                    # Benchmark and reference datasets
└── src/
    ├── __init__.py
    ├── agent.py             # Agentic Query Engine (Intent + HyDE + Self-RAG)
    ├── hybrid_search.py     # Hybrid BM25 + Dense RRF Retriever
    ├── hierarchical_summarizer.py # Map-Reduce tree summarizer for massive files
    ├── guardrails.py        # Sentence-level hallucination audit & auto-correction
    ├── documents.py         # PDF and DOCX structural parser & chunker
    ├── vector_store.py      # In-memory parent-child vector store
    ├── corrector.py         # Hallucination correction engine
    ├── pipeline.py          # Core claim verification pipeline
    └── summarizer.py        # Isolated single-pass summarizer

```

---

## 🚀 Getting Started

### 1. Prerequisites

* **Python 3.10+** installed on your system.
* A valid **Groq API Key** (`GROQ_API_KEY`) or **OpenAI API Key** (`OPENAI_API_KEY`) to power the generation and verification loops.

### 2. Installation Guide

Clone the repository from GitHub and set up your local virtual environment:

```bash
git clone [https://github.com/SuRu1010/TruthLens.git](https://github.com/SuRu1010/TruthLens.git)
cd TruthLens

python -m venv venv
source venv/bin/activate  # On Windows PowerShell/CMD: venv\Scripts\activate

pip install -r requirements.txt

```

### 3. Environment Configuration

Create a `.env` file in the root directory of your project folder and add your API keys and configuration settings:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL_NAME=llama-3.3-70b-versatile

```

### 4. Running the FastAPI Server

Launch the backend API server locally on `http://localhost:8000`:

```bash
python main.py

```

---

## 🔌 API Endpoints Reference

The TruthLens FastAPI server exposes the following endpoints for integration:

- **`POST /api/query`**
  - **Description**: Advanced agentic query endpoint supporting automatic intent routing (`GLOBAL_SUMMARY`, `FACT_CHECK`, `SPECIFIC_QA`), HyDE query expansion, hybrid search retrieval, and self-correction loops.
  - **Parameters**: `query` (string), `reference_text` or `reference_file` (upload), `intent` (optional override), `top_k`.

- **`POST /api/summarize`**
  - **Description**: Hierarchical map-reduce summarizer designed to process raw text, PDFs, or DOCX documents of any length without truncation or data loss.
  - **Parameters**: `text` (string), `file` (UploadFile), `focus` (string).

- **`POST /api/guardrail`**
  - **Description**: Audits any LLM-generated text against source grounding passages or files, returning a Factual Integrity Score (0–1), identified hallucinations, and auto-corrected text output.
  - **Parameters**: `generated_text` (string), `grounding_text` or `grounding_file` (upload), `auto_correct` (boolean).

- **`POST /api/verify`**
  - **Description**: Core claim verification endpoint. Fully backward-compatible with version 1 pipelines for fast claim-to-source entailment checking.
  - **Parameters**: `claim` (string), `reference_text` or `reference_file` (upload).

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

```

```