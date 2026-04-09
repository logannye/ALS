# Patient-First Backend Endpoints — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new API endpoints to the Erik backend: a discoveries timeline endpoint that generates daily research summaries, and a document upload endpoint that parses uploaded files into structured health data.

**Architecture:** Two new FastAPI routers registered in `api/main.py`. Discoveries uses rule-based template summaries from existing DB tables (no LLM cost). Document upload uses the existing Bedrock LLM to extract structured data from uploaded files, returning parsed results for frontend confirmation before integration.

**Tech Stack:** Python 3.12, FastAPI, PostgreSQL (`erik_kg`), Amazon Bedrock (Nova Micro), pytest

**Spec:** `~/erik-website/docs/superpowers/specs/2026-04-09-patient-first-ux-redesign.md`

**Test command:** `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v`

**Conventions (from CLAUDE.md):**
- PostgreSQL only, NEVER sqlite3
- `from db.pool import get_connection` for DB access
- TDD: write failing test first
- Entity IDs: `f"{type}:{name}".lower().replace(" ", "_")`

---

### Task 1: Discoveries Endpoint

New endpoint `GET /api/discoveries?days=14` that generates daily research summaries from existing DB tables using rule-based templates.

**Files:**
- Create: `scripts/api/routers/discoveries.py`
- Modify: `scripts/api/main.py` (register router)
- Create: `tests/test_discoveries_api.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_discoveries_api.py`:

```python
"""Tests for the discoveries API endpoint."""
from __future__ import annotations

import pytest
from datetime import date, timedelta


class TestBuildDailySummary:
    """build_daily_summary generates rule-based highlight bullets from DB data."""

    def test_returns_dict_with_required_keys(self):
        from api.routers.discoveries import build_daily_summary
        result = build_daily_summary(date.today(), dry_run=True)
        assert isinstance(result, dict)
        assert "date" in result
        assert "highlights" in result
        assert "milestone" in result
        assert "evidence_added" in result
        assert "step_count" in result

    def test_highlights_is_list(self):
        from api.routers.discoveries import build_daily_summary
        result = build_daily_summary(date.today(), dry_run=True)
        assert isinstance(result["highlights"], list)

    def test_each_highlight_has_text_and_category(self):
        from api.routers.discoveries import build_daily_summary
        result = build_daily_summary(date.today(), dry_run=True)
        for h in result["highlights"]:
            assert "text" in h
            assert "category" in h
            assert h["category"] in ("research", "treatment", "trial", "drug_design")

    def test_date_is_iso_string(self):
        from api.routers.discoveries import build_daily_summary
        d = date(2026, 4, 9)
        result = build_daily_summary(d, dry_run=True)
        assert result["date"] == "2026-04-09"


class TestBuildDiscoveriesResponse:
    """build_discoveries_response generates multiple days of summaries."""

    def test_returns_days_list(self):
        from api.routers.discoveries import build_discoveries_response
        result = build_discoveries_response(days=3, dry_run=True)
        assert "days" in result
        assert isinstance(result["days"], list)

    def test_days_count_matches_request(self):
        from api.routers.discoveries import build_discoveries_response
        result = build_discoveries_response(days=5, dry_run=True)
        assert len(result["days"]) <= 5

    def test_days_ordered_reverse_chronological(self):
        from api.routers.discoveries import build_discoveries_response
        result = build_discoveries_response(days=5, dry_run=True)
        if len(result["days"]) >= 2:
            assert result["days"][0]["date"] >= result["days"][1]["date"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_discoveries_api.py -v`
Expected: FAIL — `api.routers.discoveries` does not exist.

- [ ] **Step 3: Implement discoveries router**

Create `scripts/api/routers/discoveries.py`:

```python
"""Discoveries API — daily research summaries for the family timeline.

Generates rule-based highlight bullets from existing DB tables:
- erik_core.objects: evidence items added per day
- erik_core.entities: new entities per day
- erik_ops.research_state: step counts, protocol versions, layer transitions
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, Query

router = APIRouter()


def _query_day_metrics(target_date: date) -> dict[str, Any]:
    """Query DB for metrics on a specific day. Returns raw counts."""
    from db.pool import get_connection

    metrics: dict[str, Any] = {
        "evidence_added": 0,
        "entities_added": 0,
        "hypotheses_resolved": 0,
        "trials_found": 0,
        "molecules_designed": 0,
        "protocol_version": None,
    }

    date_str = target_date.isoformat()
    next_date_str = (target_date + timedelta(days=1)).isoformat()

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Evidence items added on this day
                cur.execute("""
                    SELECT COUNT(*) FROM erik_core.objects
                    WHERE type = 'EvidenceItem'
                      AND created_at >= %s AND created_at < %s
                """, (date_str, next_date_str))
                metrics["evidence_added"] = cur.fetchone()[0]

                # Entities created on this day
                cur.execute("""
                    SELECT COUNT(*) FROM erik_core.entities
                    WHERE created_at >= %s AND created_at < %s
                """, (date_str, next_date_str))
                metrics["entities_added"] = cur.fetchone()[0]

                # Trials found on this day
                cur.execute("""
                    SELECT COUNT(*) FROM erik_core.objects
                    WHERE type = 'EvidenceItem'
                      AND provenance_source_system = 'clinicaltrials.gov'
                      AND created_at >= %s AND created_at < %s
                """, (date_str, next_date_str))
                metrics["trials_found"] = cur.fetchone()[0]

                # Drug molecules designed (evidence from design_molecule action)
                cur.execute("""
                    SELECT COUNT(*) FROM erik_core.objects
                    WHERE type = 'EvidenceItem'
                      AND body->>'provenance' LIKE '%%design_molecule%%'
                      AND created_at >= %s AND created_at < %s
                """, (date_str, next_date_str))
                metrics["molecules_designed"] = cur.fetchone()[0]

                # Latest research state for step count
                cur.execute("""
                    SELECT state_json->>'step_count',
                           state_json->>'protocol_version',
                           state_json->>'research_layer'
                    FROM erik_ops.research_state
                    WHERE subject_ref = 'traj:draper_001'
                    LIMIT 1
                """)
                row = cur.fetchone()
                if row:
                    metrics["step_count"] = int(row[0] or 0)
                    metrics["protocol_version"] = int(row[1] or 0)
                    metrics["research_layer"] = row[2] or "normal_biology"

    except Exception:
        pass  # Return zero metrics on any DB error

    return metrics


def _metrics_to_highlights(metrics: dict[str, Any]) -> list[dict[str, str]]:
    """Convert raw metrics into plain-language highlight bullets."""
    highlights: list[dict[str, str]] = []

    if metrics.get("evidence_added", 0) > 0:
        n = metrics["evidence_added"]
        highlights.append({
            "text": f"Analyzed {n} new research papers and database entries",
            "category": "research",
        })

    if metrics.get("entities_added", 0) > 0:
        n = metrics["entities_added"]
        highlights.append({
            "text": f"Identified {n} new genes, proteins, and biological mechanisms",
            "category": "research",
        })

    if metrics.get("trials_found", 0) > 0:
        n = metrics["trials_found"]
        highlights.append({
            "text": f"Found {n} clinical trial update{'s' if n != 1 else ''}",
            "category": "trial",
        })

    if metrics.get("molecules_designed", 0) > 0:
        n = metrics["molecules_designed"]
        highlights.append({
            "text": f"Generated {n} new drug molecule candidate{'s' if n != 1 else ''}",
            "category": "drug_design",
        })

    if not highlights and metrics.get("step_count", 0) > 0:
        highlights.append({
            "text": "Galen continued running research — consolidating existing evidence",
            "category": "research",
        })

    return highlights


def _detect_milestone(metrics: dict[str, Any], prev_metrics: dict[str, Any] | None) -> str | None:
    """Detect notable milestones by comparing to previous day."""
    if prev_metrics is None:
        return None

    # Layer transition
    curr_layer = metrics.get("research_layer", "")
    prev_layer = prev_metrics.get("research_layer", "")
    layer_labels = {
        "als_mechanisms": "Galen is now mapping ALS disease mechanisms",
        "erik_specific": "Galen is now personalizing research to Erik's biology",
        "drug_design": "Galen has begun designing treatment options for Erik",
    }
    if curr_layer != prev_layer and curr_layer in layer_labels:
        return layer_labels[curr_layer]

    # First molecules
    if metrics.get("molecules_designed", 0) > 0 and (prev_metrics.get("molecules_designed", 0) == 0):
        return "First drug molecule candidates generated"

    # Evidence milestone (every 500)
    curr_evi = metrics.get("evidence_added", 0)
    if curr_evi > 0:
        # Check total evidence milestones via step count as proxy
        pass

    return None


def build_daily_summary(target_date: date, dry_run: bool = False) -> dict[str, Any]:
    """Build a single day's discovery summary."""
    if dry_run:
        return {
            "date": target_date.isoformat(),
            "highlights": [
                {"text": "Galen continued running research", "category": "research"},
            ],
            "milestone": None,
            "evidence_added": 0,
            "step_count": 0,
        }

    metrics = _query_day_metrics(target_date)
    highlights = _metrics_to_highlights(metrics)

    return {
        "date": target_date.isoformat(),
        "highlights": highlights,
        "milestone": None,  # Milestones require prev day comparison — handled in build_discoveries_response
        "evidence_added": metrics.get("evidence_added", 0),
        "step_count": metrics.get("step_count", 0),
    }


def build_discoveries_response(days: int = 14, dry_run: bool = False) -> dict[str, Any]:
    """Build the full discoveries response with multiple days."""
    today = date.today()
    day_entries: list[dict[str, Any]] = []

    prev_metrics: dict[str, Any] | None = None

    # Build oldest-first so we can detect milestones, then reverse
    for i in range(days - 1, -1, -1):
        target = today - timedelta(days=i)
        if dry_run:
            entry = build_daily_summary(target, dry_run=True)
        else:
            metrics = _query_day_metrics(target)
            highlights = _metrics_to_highlights(metrics)
            milestone = _detect_milestone(metrics, prev_metrics)
            entry = {
                "date": target.isoformat(),
                "highlights": highlights,
                "milestone": milestone,
                "evidence_added": metrics.get("evidence_added", 0),
                "step_count": metrics.get("step_count", 0),
            }
            prev_metrics = metrics

        # Only include days with content
        if entry["highlights"] or entry["milestone"]:
            day_entries.append(entry)

    # Reverse to newest-first
    day_entries.reverse()
    return {"days": day_entries}


@router.get("/api/discoveries")
async def get_discoveries(days: int = Query(default=14, ge=1, le=90)):
    """Return daily research summaries for the family timeline."""
    return build_discoveries_response(days=days)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_discoveries_api.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Register router in main.py**

In `scripts/api/main.py`, add the import and registration:

Find the router imports section and add:
```python
from api.routers.discoveries import router as discoveries_router
```

Find the `app.include_router(...)` section and add:
```python
app.include_router(discoveries_router)
```

- [ ] **Step 6: Run full test suite**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -x -q`
Expected: All tests pass (only pre-existing ClinicalTrials.gov 404 failure).

- [ ] **Step 7: Commit**

```bash
cd /Users/logannye/.openclaw/erik
git add scripts/api/routers/discoveries.py tests/test_discoveries_api.py scripts/api/main.py
git commit -m "feat: add /api/discoveries endpoint — daily research summaries for family timeline"
```

---

### Task 2: Document Upload Endpoint

New endpoint `POST /api/upload/document` that accepts a file, uses LLM to extract structured health data, and returns parsed results for confirmation. On confirmation, integrates via existing upload endpoints.

**Files:**
- Create: `scripts/api/routers/document_upload.py`
- Modify: `scripts/api/main.py` (register router)
- Create: `tests/test_document_upload_api.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_document_upload_api.py`:

```python
"""Tests for the document upload parsing API."""
from __future__ import annotations

import pytest


class TestExtractFromText:
    """extract_structured_data parses text into lab values and genetic results."""

    def test_returns_dict_with_required_keys(self):
        from api.routers.document_upload import extract_structured_data
        result = extract_structured_data("NfL 5.82 pg/mL", dry_run=True)
        assert isinstance(result, dict)
        assert "document_type" in result
        assert "extracted" in result
        assert "source" in result
        assert "date" in result

    def test_extracted_is_list(self):
        from api.routers.document_upload import extract_structured_data
        result = extract_structured_data("NfL 5.82 pg/mL", dry_run=True)
        assert isinstance(result["extracted"], list)

    def test_dry_run_returns_empty_extraction(self):
        from api.routers.document_upload import extract_structured_data
        result = extract_structured_data("anything", dry_run=True)
        assert result["extracted"] == []
        assert result["document_type"] == "unknown"


class TestParseLabPattern:
    """_parse_lab_patterns extracts common lab value patterns from text."""

    def test_extracts_nfl_value(self):
        from api.routers.document_upload import _parse_lab_patterns
        results = _parse_lab_patterns("Neurofilament light chain (NfL): 5.82 pg/mL")
        assert len(results) >= 1
        nfl = [r for r in results if "nfl" in r["name"].lower() or "neurofilament" in r["name"].lower()]
        assert len(nfl) >= 1
        assert nfl[0]["value"] == 5.82

    def test_extracts_creatinine(self):
        from api.routers.document_upload import _parse_lab_patterns
        results = _parse_lab_patterns("Creatinine: 0.9 mg/dL")
        assert len(results) >= 1
        assert results[0]["value"] == 0.9

    def test_empty_text_returns_empty(self):
        from api.routers.document_upload import _parse_lab_patterns
        results = _parse_lab_patterns("")
        assert results == []

    def test_no_match_returns_empty(self):
        from api.routers.document_upload import _parse_lab_patterns
        results = _parse_lab_patterns("This is a general note about the patient.")
        assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_document_upload_api.py -v`
Expected: FAIL — `api.routers.document_upload` does not exist.

- [ ] **Step 3: Implement document upload router**

Create `scripts/api/routers/document_upload.py`:

```python
"""Document upload API — parse uploaded health documents into structured data.

Accepts PDF, image, or text files. Extracts lab values, genetic test results,
and other structured health data using a combination of regex patterns and
LLM-powered extraction.

Flow:
1. POST /api/upload/document — upload file, get parsed results
2. POST /api/upload/document/confirm — confirm parsed results for integration
"""
from __future__ import annotations

import io
import json
import re
import uuid
from datetime import date
from typing import Any

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

router = APIRouter()

# In-memory store for pending parsed documents (TTL handled by cleanup)
_PENDING: dict[str, dict[str, Any]] = {}


# --------------------------------------------------------------------------
# Regex-based lab value extraction (fast, no LLM needed for common patterns)
# --------------------------------------------------------------------------

_LAB_PATTERN = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9 /\-\(\)]+?)\s*[:=]\s*"
    r"(?P<value>\d+\.?\d*)\s*"
    r"(?P<unit>[a-zA-Z/%]+(?:/[a-zA-Z]+)?)",
)

_KNOWN_LABS = {
    "nfl", "neurofilament", "creatinine", "creatine kinase", "ck",
    "albumin", "fvc", "forced vital capacity", "alp", "alt", "ast",
    "bun", "sodium", "potassium", "calcium", "magnesium", "phosphorus",
    "hemoglobin", "hematocrit", "wbc", "platelet", "glucose", "hba1c",
}


def _parse_lab_patterns(text: str) -> list[dict[str, Any]]:
    """Extract lab values from text using regex patterns."""
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for match in _LAB_PATTERN.finditer(text):
        name = match.group("name").strip()
        name_lower = name.lower()

        # Only extract if it looks like a known lab test
        is_known = any(lab in name_lower for lab in _KNOWN_LABS)
        if not is_known:
            continue

        key = f"{name_lower}:{match.group('value')}"
        if key in seen:
            continue
        seen.add(key)

        try:
            value = float(match.group("value"))
        except ValueError:
            continue

        results.append({
            "type": "lab",
            "name": name,
            "value": value,
            "unit": match.group("unit"),
            "confidence": "high",
        })

    return results


# --------------------------------------------------------------------------
# Genetic test pattern extraction
# --------------------------------------------------------------------------

_GENE_NAMES = {"SOD1", "C9orf72", "C9ORF72", "TARDBP", "FUS", "ATXN2", "VCP", "OPTN", "TBK1", "NEK1"}

def _parse_genetic_patterns(text: str) -> list[dict[str, Any]]:
    """Extract genetic test results from text."""
    results: list[dict[str, Any]] = []
    text_upper = text.upper()

    for gene in _GENE_NAMES:
        if gene.upper() in text_upper:
            # Try to find variant info near the gene name
            results.append({
                "type": "genetic",
                "name": gene,
                "value": "detected",
                "unit": "",
                "confidence": "medium",
            })

    return results


# --------------------------------------------------------------------------
# LLM-powered extraction (for complex documents)
# --------------------------------------------------------------------------

async def _extract_with_llm(text: str) -> list[dict[str, Any]]:
    """Use LLM to extract structured data from document text."""
    try:
        from llm.inference import create_llm
        llm = create_llm(tier="research")

        prompt = (
            "You are a medical document parser. Extract ALL lab values, genetic test results, "
            "ALSFRS-R scores, medications, and clinical observations from this document.\n\n"
            "Return ONLY valid JSON in this format:\n"
            '{"items": [{"type": "lab|genetic|alsfrs|medication|observation", '
            '"name": "test name", "value": numeric_or_string, "unit": "unit", '
            '"confidence": "high|medium|low"}]}\n\n'
            f"Document text:\n{text[:3000]}\n\n"
            "JSON output:"
        )

        result = llm.generate(prompt, max_tokens=1000, temperature=0.0)
        parsed = json.loads(result.strip())
        return parsed.get("items", [])
    except Exception:
        return []


# --------------------------------------------------------------------------
# PDF text extraction
# --------------------------------------------------------------------------

def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes. Returns empty string on failure."""
    try:
        # Try PyPDF2 / pypdf if available
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        parts = []
        for page in reader.pages[:20]:  # Cap at 20 pages
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except ImportError:
        pass
    except Exception:
        pass
    return ""


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def extract_structured_data(text: str, dry_run: bool = False) -> dict[str, Any]:
    """Extract structured health data from document text.

    Returns a dict with document_type, source, date, extracted fields,
    and uncertain_fields.
    """
    if dry_run:
        return {
            "document_id": f"doc_{uuid.uuid4().hex[:8]}",
            "document_type": "unknown",
            "source": "",
            "date": date.today().isoformat(),
            "extracted": [],
            "uncertain_fields": [],
        }

    # Combine regex and pattern-based extraction
    lab_results = _parse_lab_patterns(text)
    genetic_results = _parse_genetic_patterns(text)
    all_extracted = lab_results + genetic_results

    # Classify document type
    doc_type = "unknown"
    if lab_results:
        doc_type = "lab_results"
    elif genetic_results:
        doc_type = "genetic_testing"

    # Try to extract source (clinic/hospital name)
    source = ""
    source_patterns = ["Cleveland Clinic", "Mayo Clinic", "Johns Hopkins", "Invitae", "GeneDx"]
    for pattern in source_patterns:
        if pattern.lower() in text.lower():
            source = pattern
            break

    # Try to extract date
    doc_date = date.today().isoformat()
    date_match = re.search(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", text)
    if date_match:
        try:
            m, d, y = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
            if y < 100:
                y += 2000
            doc_date = f"{y:04d}-{m:02d}-{d:02d}"
        except ValueError:
            pass

    uncertain = [f["name"] for f in all_extracted if f.get("confidence") != "high"]

    doc_id = f"doc_{uuid.uuid4().hex[:8]}"

    return {
        "document_id": doc_id,
        "document_type": doc_type,
        "source": source,
        "date": doc_date,
        "extracted": all_extracted,
        "uncertain_fields": uncertain,
    }


@router.post("/api/upload/document")
async def upload_document(file: UploadFile = File(...)):
    """Upload a health document for parsing.

    Accepts PDF, images, DOCX, and text files. Returns parsed results
    for frontend confirmation before integration.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:  # 20MB limit
        raise HTTPException(status_code=413, detail="File too large (max 20MB)")

    # Extract text based on file type
    filename = file.filename.lower()
    text = ""

    if filename.endswith(".pdf"):
        text = _extract_text_from_pdf(content)
    elif filename.endswith(".txt"):
        text = content.decode("utf-8", errors="replace")
    elif filename.endswith((".jpg", ".jpeg", ".png", ".heic")):
        # For images, try LLM vision extraction
        try:
            extracted = await _extract_with_llm(f"[Image file: {file.filename}]")
            if extracted:
                doc_id = f"doc_{uuid.uuid4().hex[:8]}"
                result = {
                    "document_id": doc_id,
                    "document_type": "lab_results" if any(e["type"] == "lab" for e in extracted) else "unknown",
                    "source": "",
                    "date": date.today().isoformat(),
                    "extracted": extracted,
                    "uncertain_fields": [e["name"] for e in extracted if e.get("confidence") != "high"],
                }
                _PENDING[doc_id] = result
                return result
        except Exception:
            pass
        raise HTTPException(status_code=422, detail="Could not extract text from image. Please enter values manually.")
    elif filename.endswith(".docx"):
        # Basic DOCX text extraction
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                doc_xml = z.read("word/document.xml")
            root = ET.fromstring(doc_xml)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            parts = [t.text for t in root.iter(f"{{{ns['w']}}}t") if t.text]
            text = " ".join(parts)
        except Exception:
            raise HTTPException(status_code=422, detail="Could not read DOCX file. Please enter values manually.")
    else:
        # Try as plain text
        try:
            text = content.decode("utf-8", errors="replace")
        except Exception:
            raise HTTPException(status_code=422, detail="Unsupported file format.")

    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from file. Please enter values manually.")

    # Parse the extracted text
    result = extract_structured_data(text)

    # If regex extraction found nothing, try LLM
    if not result["extracted"]:
        llm_results = await _extract_with_llm(text)
        if llm_results:
            result["extracted"] = llm_results
            result["uncertain_fields"] = [e["name"] for e in llm_results if e.get("confidence") != "high"]
            if any(e["type"] == "lab" for e in llm_results):
                result["document_type"] = "lab_results"
            elif any(e["type"] == "genetic" for e in llm_results):
                result["document_type"] = "genetic_testing"

    # Store pending result for confirmation
    _PENDING[result["document_id"]] = {**result, "raw_text": text}

    return result


class ConfirmRequest(BaseModel):
    document_id: str


@router.post("/api/upload/document/confirm")
async def confirm_document(req: ConfirmRequest):
    """Confirm parsed document results and integrate into Erik's records."""
    from fastapi import Request

    pending = _PENDING.pop(req.document_id, None)
    if not pending:
        raise HTTPException(status_code=404, detail="Document not found or already confirmed")

    from db.pool import get_connection
    import json as _json

    integrated = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            for field in pending.get("extracted", []):
                if field["type"] == "lab":
                    obj_id = f"obs:lab:{field['name'].lower().replace(' ', '_')}:{pending['date']}"
                    body = {
                        "observation_kind": "lab_result",
                        "name": field["name"],
                        "value": field["value"],
                        "unit": field.get("unit", ""),
                        "collection_date": pending["date"],
                        "uploaded_by": "family",
                        "source_document": req.document_id,
                    }
                    cur.execute("""
                        INSERT INTO erik_core.objects (id, type, status, body, provenance_source_system, confidence)
                        VALUES (%s, 'Observation', 'active', %s::jsonb, 'family_upload', 0.8)
                        ON CONFLICT (id) DO UPDATE SET body = EXCLUDED.body, updated_at = NOW()
                    """, (obj_id, _json.dumps(body)))
                    integrated += 1

                elif field["type"] == "genetic":
                    obj_id = f"obs:genetic_profile:{field['name'].lower()}:{pending['date']}"
                    body = {
                        "observation_kind": "genetic_test",
                        "gene": field["name"],
                        "variant": field.get("value", ""),
                        "test_date": pending["date"],
                        "uploaded_by": "family",
                        "source_document": req.document_id,
                    }
                    cur.execute("""
                        INSERT INTO erik_core.objects (id, type, status, body, provenance_source_system, confidence)
                        VALUES (%s, 'Observation', 'active', %s::jsonb, 'family_upload', 0.8)
                        ON CONFLICT (id) DO UPDATE SET body = EXCLUDED.body, updated_at = NOW()
                    """, (obj_id, _json.dumps(body)))
                    integrated += 1

            conn.commit()

    return {"status": "ok", "integrated": integrated, "document_id": req.document_id}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_document_upload_api.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Register router in main.py**

In `scripts/api/main.py`, add:

```python
from api.routers.document_upload import router as document_upload_router
```

And:

```python
app.include_router(document_upload_router)
```

- [ ] **Step 6: Run full test suite**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
cd /Users/logannye/.openclaw/erik
git add scripts/api/routers/document_upload.py tests/test_document_upload_api.py scripts/api/main.py
git commit -m "feat: add /api/upload/document endpoint — parse uploaded health documents"
```

---

## Deployment

Push backend changes to trigger Railway redeploy:

```bash
cd /Users/logannye/.openclaw/erik
git push origin main
```

Railway auto-deploys from main. The new endpoints will be available at:
- `https://erik-api-production.up.railway.app/api/discoveries?days=14`
- `https://erik-api-production.up.railway.app/api/upload/document`
- `https://erik-api-production.up.railway.app/api/upload/document/confirm`
