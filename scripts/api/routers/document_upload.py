"""Document upload endpoints — parse lab results and genetic data from uploaded files.

Accepts PDF, DOCX, or plain-text uploads. Extracts structured health data via
regex patterns (and optional LLM fallback), then stores confirmed results as
Observation objects in erik_core.objects.
"""
from __future__ import annotations

import json
import re
import uuid
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from db.pool import get_connection

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# In-memory store for pending (unconfirmed) document uploads
# ---------------------------------------------------------------------------

_pending_uploads: dict[str, dict] = {}

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ConfirmRequest(BaseModel):
    document_id: str


# ---------------------------------------------------------------------------
# Known lab names for regex extraction
# ---------------------------------------------------------------------------

_KNOWN_LABS: dict[str, str] = {
    "nfl": "NfL",
    "neurofilament": "NfL",
    "neurofilament light": "NfL",
    "neurofilament light chain": "NfL",
    "creatinine": "Creatinine",
    "creatine kinase": "Creatine Kinase",
    "ck": "Creatine Kinase",
    "albumin": "Albumin",
    "fvc": "FVC",
    "forced vital capacity": "FVC",
    "alp": "ALP",
    "alkaline phosphatase": "ALP",
    "alt": "ALT",
    "ast": "AST",
    "bicarbonate": "Bicarbonate",
    "bun": "BUN",
    "blood urea nitrogen": "BUN",
    "calcium": "Calcium",
    "chloride": "Chloride",
    "co2": "CO2",
    "glucose": "Glucose",
    "hemoglobin": "Hemoglobin",
    "hematocrit": "Hematocrit",
    "potassium": "Potassium",
    "sodium": "Sodium",
    "total protein": "Total Protein",
    "wbc": "WBC",
    "white blood cell": "WBC",
}

# Build a regex alternation from known lab names (longest first to avoid partial matches)
_lab_names_sorted = sorted(_KNOWN_LABS.keys(), key=len, reverse=True)
_lab_pattern_str = "|".join(re.escape(n) for n in _lab_names_sorted)

# Pattern: <lab name> [optional parenthetical like (NfL)]: <number> <unit>
_LAB_RE = re.compile(
    rf"(?P<name>(?:{_lab_pattern_str})(?:\s*\([^)]*\))?)"
    r"\s*:\s*"
    r"(?P<value>\d+(?:\.\d+)?)"
    r"\s+"
    r"(?P<unit>[a-zA-Z/%]+(?:/[a-zA-Z]+)?)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Known ALS-related genes
# ---------------------------------------------------------------------------

_ALS_GENES = frozenset({
    "SOD1", "C9orf72", "TARDBP", "FUS", "ATXN2",
    "VCP", "OPTN", "TBK1", "NEK1",
})

_GENE_RE = re.compile(
    r"\b(" + "|".join(re.escape(g) for g in _ALS_GENES) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------

def _parse_lab_patterns(text: str) -> list[dict]:
    """Extract known lab values from text using regex.

    Pattern: ``name: value unit`` (e.g. "NfL: 5.82 pg/mL").
    Only extracts labs whose name matches a known ALS-relevant lab.

    Returns a list of dicts with keys: type, name, value, unit, confidence.
    """
    if not text or not text.strip():
        return []

    results: list[dict] = []
    for m in _LAB_RE.finditer(text):
        raw_name = m.group("name").strip()
        # Strip any parenthetical to find the canonical name
        clean_name = re.sub(r"\s*\([^)]*\)", "", raw_name).strip().lower()
        canonical = _KNOWN_LABS.get(clean_name, raw_name)

        results.append({
            "type": "lab",
            "name": canonical,
            "value": float(m.group("value")),
            "unit": m.group("unit"),
            "confidence": "high",
        })

    return results


def _parse_genetic_patterns(text: str) -> list[dict]:
    """Detect known ALS-related gene names in text.

    Returns a list of dicts with keys: type, name, value, unit, confidence.
    """
    if not text or not text.strip():
        return []

    seen: set[str] = set()
    results: list[dict] = []
    for m in _GENE_RE.finditer(text):
        gene = m.group(1).upper()
        if gene not in seen:
            seen.add(gene)
            results.append({
                "type": "genetic",
                "name": gene,
                "value": "detected",
                "unit": "",
                "confidence": "medium",
            })

    return results


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_text_from_pdf(data: bytes) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""


def _extract_text_from_docx(data: bytes) -> str:
    """Extract text from a DOCX file by reading word/document.xml."""
    try:
        with zipfile.ZipFile(BytesIO(data)) as zf:
            if "word/document.xml" not in zf.namelist():
                return ""
            xml_data = zf.read("word/document.xml")
            tree = ET.fromstring(xml_data)
            # Word XML namespace
            ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            texts = []
            for t_elem in tree.iter(f"{ns}t"):
                if t_elem.text:
                    texts.append(t_elem.text)
            return " ".join(texts)
    except Exception:
        return ""


def _extract_text_from_upload(data: bytes, filename: str) -> str:
    """Determine file type and extract text accordingly."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_text_from_pdf(data)
    elif lower.endswith(".docx"):
        return _extract_text_from_docx(data)
    else:
        # Assume plain text
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return data.decode("latin-1")
            except Exception:
                return ""


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------

def _try_llm_extraction(text: str) -> list[dict]:
    """Attempt LLM-based extraction as fallback when regex finds nothing."""
    try:
        from llm.inference import create_llm
    except ImportError:
        return []

    try:
        llm = create_llm()
        prompt = (
            "Extract any lab values or genetic test results from this medical document. "
            "Return a JSON array of objects with keys: type (lab|genetic), name, value, unit, confidence.\n\n"
            f"Document text:\n{text[:4000]}"
        )
        response = llm.generate(prompt, max_tokens=1000)
        # Try to parse JSON from the response
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            items = json.loads(match.group(0))
            # Mark LLM results with lower confidence
            for item in items:
                item["confidence"] = "low"
            return items
    except Exception:
        pass

    return []


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------

def extract_structured_data(text: str, dry_run: bool = False) -> dict:
    """Combine regex + genetic extraction on the given text.

    Returns a dict with: document_id, document_type, source, date, extracted,
    uncertain_fields.

    In dry_run mode, returns a minimal valid empty response (no parsing).
    """
    doc_id = f"doc:{uuid.uuid4().hex[:12]}"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if dry_run:
        return {
            "document_id": doc_id,
            "document_type": "unknown",
            "source": "upload",
            "date": today,
            "extracted": [],
            "uncertain_fields": [],
        }

    labs = _parse_lab_patterns(text)
    genes = _parse_genetic_patterns(text)
    extracted = labs + genes

    # If regex found nothing, try LLM fallback
    if not extracted:
        extracted = _try_llm_extraction(text)

    # Classify document type
    if labs and genes:
        doc_type = "comprehensive_panel"
    elif labs:
        doc_type = "lab_report"
    elif genes:
        doc_type = "genetic_report"
    else:
        doc_type = "unknown"

    # Fields where confidence is not "high" go into uncertain_fields
    uncertain = [item for item in extracted if item.get("confidence") != "high"]

    return {
        "document_id": doc_id,
        "document_type": doc_type,
        "source": "upload",
        "date": today,
        "extracted": extracted,
        "uncertain_fields": uncertain,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/upload/document")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF, DOCX, or text) and extract structured health data.

    Returns parsed lab values and genetic results for user confirmation.
    """
    # Validate file size
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    filename = file.filename or "unknown.txt"

    # Extract text from the uploaded file
    text = _extract_text_from_upload(data, filename)
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from the uploaded file.",
        )

    # Run extraction
    result = extract_structured_data(text)

    # Store pending for confirmation
    _pending_uploads[result["document_id"]] = result

    return result


@router.post("/upload/document/confirm")
def confirm_document(req: ConfirmRequest):
    """Confirm a previously uploaded document, integrating extracted data into the DB.

    Looks up the pending document by ID and inserts each extracted item as an
    Observation into erik_core.objects.
    """
    pending = _pending_uploads.pop(req.document_id, None)
    if pending is None:
        raise HTTPException(
            status_code=404,
            detail=f"No pending document found with id '{req.document_id}'.",
        )

    today = pending.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parsed_date = datetime.strptime(today, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    inserted_ids: list[str] = []

    with get_connection() as conn:
        for item in pending.get("extracted", []):
            item_type = item.get("type", "unknown")
            name_slug = re.sub(r"[^a-z0-9]+", "_", item["name"].lower()).strip("_")

            if item_type == "lab":
                obj_id = f"obs:lab:{name_slug}:{today}"
                body = {
                    "observation_kind": "lab_result",
                    "name": item["name"],
                    "value": item["value"],
                    "unit": item["unit"],
                    "confidence": item.get("confidence", "medium"),
                    "collection_date": today,
                    "source_document": req.document_id,
                    "uploaded_by": "family",
                }
            elif item_type == "genetic":
                obj_id = f"obs:genetic_profile:{name_slug}:{today}"
                body = {
                    "observation_kind": "genetic_result",
                    "gene": item["name"],
                    "status": item.get("value", "detected"),
                    "confidence": item.get("confidence", "medium"),
                    "collection_date": today,
                    "source_document": req.document_id,
                    "uploaded_by": "family",
                }
            else:
                continue

            conn.execute(
                """INSERT INTO erik_core.objects
                   (id, type, status, body, provenance_source_system, time_observed_at)
                   VALUES (%s, 'Observation', 'active', %s, 'document_upload', %s)
                   ON CONFLICT (id) DO UPDATE SET
                     body = EXCLUDED.body,
                     updated_at = NOW()""",
                (obj_id, json.dumps(body), parsed_date),
            )
            inserted_ids.append(obj_id)

        conn.commit()

    return {
        "status": "confirmed",
        "document_id": req.document_id,
        "items_stored": len(inserted_ids),
        "ids": inserted_ids,
    }
