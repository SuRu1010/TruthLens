from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.documents import extract_text_from_bytes
from src.summarizer import IsolatedSummarizer
from src.pipeline import verify_claim
from src.corrector import HallucinationCorrector

app = FastAPI(title="TruthLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

summarizer = None
corrector = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        summarizer = IsolatedSummarizer()
    return summarizer

def get_corrector():
    global corrector
    if corrector is None:
        corrector = HallucinationCorrector()
    return corrector


@app.get("/")
def read_root():
    return {"status": "online", "message": "TruthLens API is active"}


@app.post("/api/verify")
async def verify_endpoint(
    claim: str = Form(...),
    reference_text: Optional[str] = Form(None),
    reference_file: Optional[UploadFile] = File(None)
):
    try:
        reference = reference_text or ""
        
        # Extract text from attached file if present
        if reference_file:
            contents = await reference_file.read()
            extracted = extract_text_from_bytes(reference_file.filename, contents)
            reference = f"{reference}\n{extracted}".strip()
            
        if not reference:
            raise HTTPException(status_code=400, detail="Reference text or document is required.")

        return verify_claim(claim, reference)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/summarize")
async def summarize_endpoint(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        input_text = text or ""
        source_filename = None
        
        if file:
            contents = await file.read()
            extracted = extract_text_from_bytes(file.filename, contents)
            input_text = f"{input_text}\n{extracted}".strip()
            source_filename = file.filename
            
        if not input_text:
            raise HTTPException(status_code=400, detail="Please provide text or upload a document.")

        res = get_summarizer().summarize(input_text)
        
        # Standardize keys expected by front-end
        if isinstance(res, dict):
            if source_filename:
                res["source_filename"] = source_filename
            return res
            
        return {
            "summary_text": getattr(res, "summary_text", str(res)),
            "word_count": getattr(res, "word_count", len(input_text.split())),
            "key_takeaways": getattr(res, "key_takeaways", []),
            "source_filename": source_filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)