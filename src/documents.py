"""In-memory PDF / DOCX text extraction for API uploads."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

ALLOWED_SUFFIXES = {".pdf", ".docx"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


class DocumentParseError(Exception):
    """Raised when an upload cannot be parsed into usable text."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _suffix(filename: str | None) -> str:
    if not filename:
        raise DocumentParseError("Uploaded file is missing a filename.")
    return Path(filename).suffix.lower()


def extract_text_from_bytes(filename: str, data: bytes) -> str:
    suffix = _suffix(filename)
    if suffix not in ALLOWED_SUFFIXES:
        raise DocumentParseError(
            f"Unsupported file type '{suffix or 'unknown'}'. Use .pdf or .docx."
        )
    if not data:
        raise DocumentParseError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise DocumentParseError("File exceeds the 15 MB upload limit.")

    if suffix == ".pdf":
        text = _extract_pdf(data)
    else:
        text = _extract_docx(data)

    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        raise DocumentParseError(
            "No extractable text found in the uploaded file."
        )
    return cleaned


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(BytesIO(data))
    except Exception as exc:
        raise DocumentParseError(f"Unable to read PDF: {exc}") from exc

    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _extract_docx(data: bytes) -> str:
    from docx import Document

    try:
        document = Document(BytesIO(data))
    except Exception as exc:
        raise DocumentParseError(f"Unable to read DOCX: {exc}") from exc

    parts: list[str] = [p.text for p in document.paragraphs if p.text]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def combine_reference(reference_text: str | None, file_text: str | None) -> str:
    chunks = [c.strip() for c in (reference_text or "", file_text or "") if c and c.strip()]
    return "\n\n".join(chunks).strip()
