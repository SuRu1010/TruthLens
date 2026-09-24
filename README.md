# TruthLens
A foundational Retrieval-Augmented Generation (RAG) system built to parse context from source documents and generate grounded answers using vector similarity search and LLMs.

# 🔍 TruthLens: Retrieval-Augmented Generation (RAG) System

TruthLens is a Retrieval-Augmented Generation (RAG) application designed to extract context from custom documents and generate accurate, context-grounded answers using vector embeddings and large language models (LLMs).

This repository currently implements a **Level 1 (Naive) RAG pipeline** to demonstrate baseline document retrieval and answer synthesis.

---

## ✨ Features

- **Document Ingestion:** Chunking and vectorizing raw text/documents into local storage.
- **Semantic Vector Search:** Retrieves the top $K$ most relevant context chunks for a given query.
- **Context-Grounded Generation:** Sends retrieved sources alongside user queries to an LLM to minimize hallucinated responses.
- **Free Developer Stack:** Built using open-source tools and free API tiers.

---

## 🛠️ System Architecture (Level 1)

User Query ──> [ Vector Similarity Search ] ──> Retrieve Top Context Chunks
│
▼
User Query + Chunks ───────────────────────────> [ Single-Pass LLM Call ]
│
▼
Delivered Response


---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- A free **Google AI Studio API Key** (or preferred LLM provider)

### 2. Installation

Clone the repository:
```bash
git clone [https://github.com/YOUR-USERNAME/TruthLens.git](https://github.com/YOUR-USERNAME/TruthLens.git)
cd TruthLens
Create a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

Bash
pip install google-genai chromadb
3. Environment Setup
Set your API key in your terminal:

Bash
# Linux / macOS
export GEMINI_API_KEY="your-api-key-here"

# Windows (CMD)
set GEMINI_API_KEY=your-api-key-here
4. Running the Pipeline
Execute the main script:

Bash
python main.py
🗺️ TruthLens Roadmap
[x] Level 1 RAG: Basic vector search and prompt-based response generation.

[ ] Level 2 Upgrade: Hybrid search (BM25 keyword matching + Dense Embeddings) and Reranking.

[ ] Level 3 Upgrade: Multi-pass Verification & Self-Correction agent loop to eliminate remaining hallucinations.

[ ] Level 3 Upgrade: Map-Reduce processing for handling long, multi-page PDFs.

📄 License
Distributed under the MIT License.


---

### Commands to Commit & Push:

Now run these commands in your terminal:

```bash
git init
git add .
git commit -m "Initial commit: TruthLens Level 1 RAG implementation"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/TruthLens.git
git push -u origin main
(Remember to swap YOUR-USERNAME with your actual GitHub username).
