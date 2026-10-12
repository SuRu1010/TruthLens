"""TruthLens API v2 — Level 3/4 Agentic RAG Backend."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.documents import (
    combine_reference,
    extract_text_from_bytes,
    parse_document_to_structured,
)
from src.summarizer import IsolatedSummarizer
from src.pipeline import verify_claim
from src.corrector import HallucinationCorrector

app = FastAPI(
    title="TruthLens API",
    version="2.0.0",
    description="Level 3/4 Agentic RAG — Hierarchical Summarization, Hybrid Search, Self-RAG hallucination detection.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_summarizer: Optional[IsolatedSummarizer] = None
_corrector: Optional[HallucinationCorrector] = None
_agent = None


def _get_summarizer() -> IsolatedSummarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = IsolatedSummarizer()
    return _summarizer


def _get_corrector() -> HallucinationCorrector:
    global _corrector
    if _corrector is None:
        _corrector = HallucinationCorrector()
    return _corrector


def _get_agent():
    global _agent
    if _agent is None:
        import os
        from src.agent import AgenticQueryEngine

        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        provider = "groq" if os.getenv("GROQ_API_KEY") else "openai"
        _agent = AgenticQueryEngine(api_key=api_key, provider=provider)
    return _agent


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "TruthLens API v2 — Advanced Agentic RAG",
        "features": [
            "Hierarchical Map-Reduce Summarization (no truncation)",
            "Hybrid BM25 + Dense Retrieval with RRF",
            "Agentic Query Engine with HyDE + Self-Correction",
            "Sentence-level Hallucination Guardrail",
        ],
    }


# ---------------------------------------------------------------------------
# Core: Claim Verification (unchanged from v1 — backward compatible)
# ---------------------------------------------------------------------------


@app.post("/api/verify")
async def verify_endpoint(
    claim: str = Form(...),
    reference_text: Optional[str] = Form(None),
    reference_file: Optional[UploadFile] = File(None),
):
    try:
        file_text: Optional[str] = None
        if reference_file and reference_file.filename:
            contents = await reference_file.read()
            file_text = extract_text_from_bytes(reference_file.filename, contents)

        reference = combine_reference(reference_text, file_text)
        if not reference:
            raise HTTPException(status_code=400, detail="Reference text or document is required.")

        return verify_claim(claim, reference)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Core: Summarize — now uses HierarchicalSummarizer for large docs
# ---------------------------------------------------------------------------


@app.post("/api/summarize")
async def summarize_endpoint(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    focus: Optional[str] = Form("general"),
):
    """Summarize text or PDF/DOCX. Automatically uses hierarchical map-reduce for large documents."""
    try:
        input_text = (text or "").strip()
        source_filename: Optional[str] = None

        if file and file.filename:
            contents = await file.read()
            source_filename = file.filename
            parsed_doc = parse_document_to_structured(file.filename, contents)
            input_text = (input_text + "\n\n" + parsed_doc.raw_text).strip()

        if not input_text:
            raise HTTPException(
                status_code=400, detail="Please provide text or upload a document."
            )

        word_count = len(input_text.split())
        res = _get_summarizer().summarize(
            input_text,
            focus=focus or "general",
        )

        return {
            **res,
            "source_filename": source_filename,
            "source_word_count": word_count,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# NEW: Agentic Query — Intent Routing + HyDE + Self-RAG
# ---------------------------------------------------------------------------


@app.post("/api/query")
async def agentic_query_endpoint(
    query: str = Form(...),
    reference_text: Optional[str] = Form(None),
    reference_file: Optional[UploadFile] = File(None),
    intent: Optional[str] = Form(None),
    top_k: int = Form(5),
):
    """Agentic query over a reference document.
    
    Routes between GLOBAL_SUMMARY, FACT_CHECK, and SPECIFIC_QA automatically.
    Applies HyDE query expansion and Self-RAG correction.
    """
    try:
        input_text = (reference_text or "").strip()
        source_filename: Optional[str] = None

        if reference_file and reference_file.filename:
            contents = await reference_file.read()
            source_filename = reference_file.filename
            parsed = parse_document_to_structured(reference_file.filename, contents)
            input_text = (input_text + "\n\n" + parsed.raw_text).strip()

        if not input_text:
            raise HTTPException(
                status_code=400,
                detail="A reference document or text is required for querying.",
            )

        agent = _get_agent()
        result = agent.query(
            query=query,
            document=input_text,
            intent_override=intent if intent in ("GLOBAL_SUMMARY", "FACT_CHECK", "SPECIFIC_QA") else None,
            top_k=top_k,
        )

        return {
            "intent": result.intent,
            "answer": result.answer,
            "citations": result.citations,
            "confidence": result.confidence,
            "self_correction_applied": result.self_correction_applied,
            "correction_note": result.correction_note,
            "source_filename": source_filename,
            "retrieved_chunks": result.retrieved_chunks[:3],
            "metadata": result.metadata,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# NEW: Guardrail Audit — Hallucination Integrity Check on generated text
# ---------------------------------------------------------------------------


@app.post("/api/guardrail")
async def guardrail_endpoint(
    generated_text: str = Form(...),
    grounding_text: Optional[str] = Form(None),
    grounding_file: Optional[UploadFile] = File(None),
    auto_correct: bool = Form(True),
):
    """Run hallucination guardrail on an LLM-generated text.
    
    Returns a Factual Integrity Score (0–1), hallucinated sentences, and auto-corrected text.
    """
    try:
        grounding = (grounding_text or "").strip()

        if grounding_file and grounding_file.filename:
            contents = await grounding_file.read()
            extracted = extract_text_from_bytes(grounding_file.filename, contents)
            grounding = (grounding + "\n\n" + extracted).strip()

        if not grounding:
            raise HTTPException(
                status_code=400, detail="Grounding source text or document is required."
            )

        from src.guardrails import HallucinationGuardrail
        import os

        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        provider = "groq" if os.getenv("GROQ_API_KEY") else "openai"
        guardrail = HallucinationGuardrail(api_key=api_key, provider=provider)

        grounding_passages = [p.strip() for p in grounding.split("\n\n") if p.strip()]
        verdict = guardrail.audit(
            generated_text=generated_text,
            grounding_passages=grounding_passages,
            auto_correct=auto_correct,
        )

        return {
            "factual_integrity_score": verdict.factual_integrity_score,
            "is_clean": verdict.is_clean,
            "corrections_made": verdict.corrections_made,
            "hallucinated_sentences": verdict.hallucinated_sentences,
            "corrected_text": verdict.corrected_text,
            "sentence_verdicts": verdict.sentence_verdicts,
            "grounding_sources_used": verdict.grounding_sources_used,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)