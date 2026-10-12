🔍 TruthLens: Advanced Level 3/4 Agentic RAG & Hallucination Guardrail

TruthLens is an enterprise-grade, agentic Retrieval-Augmented Generation (RAG) and hallucination verification system designed to extract insights from massive documents, eliminate factual drift, and enforce strict source grounding using hybrid search, multi-intent routing, and self-correction loops.

✨ Core Architecture & Features

Multi-Intent Agentic Router (src/agent.py): Automatically classifies user intent into GLOBAL_SUMMARY, FACT_CHECK, or SPECIFIC_QA, routing requests to specialized processing pathways.

Hierarchical Map-Reduce Summarizer (src/hierarchical_summarizer.py): Solves token-limit bottlenecks for 50–200+ page PDFs/DOCXs using tree-based chunk synthesis (RAPTOR-inspired clustering) without truncation.

Hybrid Dense + Sparse Search (src/hybrid_search.py): Combines Dense Vector Embeddings and Okapi BM25 keyword matching via Reciprocal Rank Fusion (RRF) and Cross-Encoder re-ranking.

HyDE Query Expansion (src/agent.py): Generates hypothetical document passages to significantly boost retrieval recall for complex questions.

Sentence-Level Guardrail & Self-RAG (src/guardrails.py, src/corrector.py): Audits generated text sentence-by-sentence against source contexts, computing a Factual Integrity Score (0–1) and auto-correcting hallucinations.

🛠️ Project Structure

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


🚀 Getting Started

1. Prerequisites

Python 3.10+

A valid Groq API Key (GROQ_API_KEY) or OpenAI API Key (OPENAI_API_KEY)

2. Installation

Clone the repository and set up your virtual environment:

git clone https://github.com/SuRu1010/TruthLens.git
cd TruthLens

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt


3. Environment Configuration

Create a .env file in the root directory:

GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL_NAME=llama-3.3-70b-versatile


4. Running the FastAPI Server

Launch the backend API server on http://localhost:8000:

python main.py


🔌 API Endpoints

POST /api/query: Advanced agentic query endpoint supporting automatic intent routing, HyDE expansion, hybrid search retrieval, and self-correction.

POST /api/summarize: Hierarchical map-reduce summarizer for large documents or raw text.

POST /api/guardrail: Audits generated text against source grounding passages to compute a Factual Integrity Score and auto-correct unverified claims.

POST /api/verify: Core claim verification endpoint (backward-compatible with v1).

📄 License

Distributed under the MIT License.