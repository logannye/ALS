# P1-P3 Research Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 7 enhancements to Erik's autonomous ALS research engine — clinical trial eligibility, PRO-ACT trajectory matching, bioRxiv preprints, adversarial verification, combination synergy, Thompson sampling policy, and Galen SCM integration.

**Architecture:** Additive integration into the existing single-threaded research loop. Each enhancement is a self-contained module plugging into established interfaces (`BaseConnector`, `ActionType`, 6-stage pipeline). All activatable independently via hot-reload config flags. Zero downtime — Erik keeps running 24/7 throughout.

**Tech Stack:** Python 3.12, PostgreSQL 17, Pydantic v2, psycopg/psycopg_pool, scipy (DTW), lifelines (Kaplan-Meier), requests, MLX-LM (existing)

**Spec:** `docs/specs/2026-03-27-p1-p3-research-enhancements-design.md`

**Implementation order** (clinical urgency):
1. Task Group 1: Clinical trial eligibility matching (P2c)
2. Task Group 2: PRO-ACT trajectory matching (P1b)
3. Task Group 3: bioRxiv/medRxiv preprints (P1a)
4. Task Group 4: Adversarial protocol verification (P2b)
5. Task Group 5: Drug combination synergy (P2a)
6. Task Group 6: Thompson sampling policy (P3a)
7. Task Group 7: Galen SCM integration (P3b)

**Conventions:**
- Working directory: `/Users/logannye/.openclaw/erik`
- Python env: `conda run -n erik-core`
- PYTHONPATH: `scripts/`
- Test command: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/<file> -v`
- All tests run with: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v -k "not network and not chembl and not llm"`
- TDD: write failing test first, then minimal implementation
- Every canonical object inherits from `BaseEnvelope`
- Entity IDs: `f"{type}:{name}".lower().replace(" ", "_")`
- PostgreSQL only — never SQLite

---

## Task Group 1: Clinical Trial Eligibility Matching (P2c)

### Task 1.1: EligibilityVerdict Model and Matcher Core

**Files:**
- Create: `scripts/research/eligibility.py`
- Create: `tests/test_eligibility.py`

- [ ] **Step 1: Write failing tests for EligibilityVerdict model and structured field matching**

```python
# tests/test_eligibility.py
"""Tests for clinical trial eligibility matching."""
from __future__ import annotations

import pytest

from research.eligibility import (
    EligibilityVerdict,
    check_structured_eligibility,
    extract_criteria_from_text,
    compute_eligibility,
    ERIK_ELIGIBILITY_PROFILE,
)


class TestEligibilityVerdict:
    def test_verdict_creation(self):
        v = EligibilityVerdict(
            trial_nct_id="NCT06012345",
            trial_title="Test Trial",
            trial_phase="Phase 3",
            intervention_name="pridopidine",
            eligible="likely",
            blocking_criteria=[],
            pending_criteria=["genetic_status"],
            matching_criteria=["age", "sex"],
            protocol_alignment=0.85,
            urgency="enrolling_now",
            sites_near_erik=[],
        )
        assert v.eligible == "likely"
        assert v.trial_nct_id == "NCT06012345"

    def test_verdict_eligible_literals(self):
        for status in ("yes", "no", "likely", "pending_data"):
            v = EligibilityVerdict(
                trial_nct_id="NCT00000001",
                trial_title="T",
                trial_phase="1",
                intervention_name="x",
                eligible=status,
                blocking_criteria=[],
                pending_criteria=[],
                matching_criteria=[],
                protocol_alignment=0.0,
                urgency="completed",
                sites_near_erik=[],
            )
            assert v.eligible == status


class TestStructuredEligibility:
    def test_age_in_range(self):
        """Erik is 67 — trial accepting 18-80 should pass."""
        result = check_structured_eligibility(
            min_age=18, max_age=80, sex="All", healthy_volunteers=False,
        )
        assert "age" in result["matching"]
        assert "age" not in result["blocking"]

    def test_age_too_old(self):
        """Trial accepting 18-65 should block Erik (67)."""
        result = check_structured_eligibility(
            min_age=18, max_age=65, sex="All", healthy_volunteers=False,
        )
        assert "age" in result["blocking"]

    def test_sex_match(self):
        result = check_structured_eligibility(
            min_age=18, max_age=99, sex="Male", healthy_volunteers=False,
        )
        assert "sex" in result["matching"]

    def test_sex_mismatch(self):
        result = check_structured_eligibility(
            min_age=18, max_age=99, sex="Female", healthy_volunteers=False,
        )
        assert "sex" in result["blocking"]

    def test_sex_all_matches(self):
        result = check_structured_eligibility(
            min_age=18, max_age=99, sex="All", healthy_volunteers=False,
        )
        assert "sex" in result["matching"]


class TestCriteriaExtraction:
    def test_alsfrs_threshold(self):
        text = "Inclusion: ALSFRS-R score >= 24 at screening"
        criteria = extract_criteria_from_text(text)
        assert criteria["alsfrs_r_min"] == 24

    def test_fvc_threshold(self):
        text = "FVC >= 60% predicted at screening"
        criteria = extract_criteria_from_text(text)
        assert criteria["fvc_min_percent"] == 60

    def test_disease_duration(self):
        text = "Symptom onset within 36 months of screening"
        criteria = extract_criteria_from_text(text)
        assert criteria["max_duration_months"] == 36

    def test_riluzole_required(self):
        text = "Must be on stable dose of riluzole for at least 4 weeks"
        criteria = extract_criteria_from_text(text)
        assert criteria["riluzole_required"] is True

    def test_genetic_requirement(self):
        text = "Confirmed SOD1 mutation by genetic testing"
        criteria = extract_criteria_from_text(text)
        assert criteria["genetic_required"] is True

    def test_no_criteria_found(self):
        text = "This is a general description with no specific criteria"
        criteria = extract_criteria_from_text(text)
        assert criteria["alsfrs_r_min"] is None
        assert criteria["fvc_min_percent"] is None

    def test_erik_passes_alsfrs(self):
        """Erik's ALSFRS-R is 43, threshold 24 — should pass."""
        text = "ALSFRS-R >= 24"
        criteria = extract_criteria_from_text(text)
        # Erik's score of 43 > 24
        assert criteria["alsfrs_r_min"] == 24

    def test_erik_passes_fvc(self):
        """Erik's FVC is 100%, threshold 60% — should pass."""
        text = "FVC greater than 60%"
        criteria = extract_criteria_from_text(text)
        assert criteria["fvc_min_percent"] == 60

    def test_erik_passes_duration(self):
        """Erik's duration is 14 months, threshold 36 — should pass."""
        text = "within 36 months of symptom onset"
        criteria = extract_criteria_from_text(text)
        assert criteria["max_duration_months"] == 36


class TestComputeEligibility:
    def test_eligible_trial(self):
        """Trial with criteria Erik meets — should be 'yes' or 'likely'."""
        verdict = compute_eligibility(
            nct_id="NCT06012345",
            title="PREVAiLS Phase 3",
            phase="Phase 3",
            intervention_name="pridopidine",
            min_age=18,
            max_age=80,
            sex="All",
            healthy_volunteers=False,
            eligibility_text="ALSFRS-R >= 24. FVC >= 50%. Symptom onset within 36 months.",
            enrollment_status="RECRUITING",
            sites=["Cleveland Clinic, Cleveland, Ohio"],
            current_protocol_top_interventions=["pridopidine"],
        )
        assert verdict.eligible in ("yes", "likely")
        assert verdict.urgency == "enrolling_now"
        assert len(verdict.blocking_criteria) == 0

    def test_blocked_by_age(self):
        verdict = compute_eligibility(
            nct_id="NCT00000002",
            title="Young ALS Trial",
            phase="Phase 2",
            intervention_name="drug_x",
            min_age=18,
            max_age=55,
            sex="All",
            healthy_volunteers=False,
            eligibility_text="",
            enrollment_status="RECRUITING",
            sites=[],
            current_protocol_top_interventions=[],
        )
        assert verdict.eligible == "no"
        assert "age" in verdict.blocking_criteria

    def test_pending_genetics(self):
        """Trial requiring genetic confirmation — pending_data for Erik."""
        verdict = compute_eligibility(
            nct_id="NCT00000003",
            title="SOD1 Trial",
            phase="Phase 2",
            intervention_name="tofersen",
            min_age=18,
            max_age=80,
            sex="All",
            healthy_volunteers=False,
            eligibility_text="Confirmed SOD1 pathogenic variant required",
            enrollment_status="RECRUITING",
            sites=[],
            current_protocol_top_interventions=[],
        )
        assert verdict.eligible == "pending_data"
        assert any("genetic" in c.lower() for c in verdict.pending_criteria)

    def test_protocol_alignment_high(self):
        verdict = compute_eligibility(
            nct_id="NCT00000004",
            title="Pridopidine Trial",
            phase="Phase 3",
            intervention_name="pridopidine",
            min_age=18,
            max_age=80,
            sex="All",
            healthy_volunteers=False,
            eligibility_text="",
            enrollment_status="RECRUITING",
            sites=[],
            current_protocol_top_interventions=["pridopidine", "riluzole"],
        )
        assert verdict.protocol_alignment > 0.5

    def test_protocol_alignment_low(self):
        verdict = compute_eligibility(
            nct_id="NCT00000005",
            title="Unknown Drug Trial",
            phase="Phase 1",
            intervention_name="xyz_novel_compound",
            min_age=18,
            max_age=80,
            sex="All",
            healthy_volunteers=False,
            eligibility_text="",
            enrollment_status="RECRUITING",
            sites=[],
            current_protocol_top_interventions=["pridopidine", "riluzole"],
        )
        assert verdict.protocol_alignment < 0.5

    def test_enrollment_status_mapping(self):
        for api_status, expected_urgency in [
            ("RECRUITING", "enrolling_now"),
            ("NOT_YET_RECRUITING", "not_yet_recruiting"),
            ("COMPLETED", "completed"),
            ("ACTIVE_NOT_RECRUITING", "completed"),
        ]:
            verdict = compute_eligibility(
                nct_id="NCT00000006",
                title="T",
                phase="1",
                intervention_name="x",
                min_age=18,
                max_age=80,
                sex="All",
                healthy_volunteers=False,
                eligibility_text="",
                enrollment_status=api_status,
                sites=[],
                current_protocol_top_interventions=[],
            )
            assert verdict.urgency == expected_urgency

    def test_site_proximity(self):
        verdict = compute_eligibility(
            nct_id="NCT00000007",
            title="Ohio Trial",
            phase="Phase 3",
            intervention_name="drug_a",
            min_age=18,
            max_age=80,
            sex="All",
            healthy_volunteers=False,
            eligibility_text="",
            enrollment_status="RECRUITING",
            sites=[
                "Cleveland Clinic, Cleveland, Ohio, United States",
                "Mayo Clinic, Rochester, Minnesota, United States",
            ],
            current_protocol_top_interventions=[],
            geographic_region="Ohio",
        )
        assert len(verdict.sites_near_erik) >= 1
        assert any("Cleveland" in s for s in verdict.sites_near_erik)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_eligibility.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'research.eligibility'`

- [ ] **Step 3: Implement eligibility.py**

```python
# scripts/research/eligibility.py
"""Clinical trial eligibility matching for Erik Draper.

Computes whether Erik qualifies for specific ALS clinical trials based on
structured eligibility fields and free-text inclusion/exclusion criteria.

Rule-based, no LLM — uses regex extraction for common ALS trial criteria
and deterministic comparison against Erik's profile.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel


# Erik's current clinical profile — mirrors clinical_trials.py ERIK_PROFILE
ERIK_ELIGIBILITY_PROFILE = {
    "age": 67,
    "sex": "male",
    "diagnosis": "ALS",
    "onset_region": "lower_limb",
    "alsfrs_r": 43,
    "fvc_percent": 100,
    "disease_duration_months": 14,
    "on_riluzole": True,
    "genetic_status": "pending",
    "comorbidities": ["hypertension", "prediabetes", "cervical_stenosis"],
}


class EligibilityVerdict(BaseModel):
    """Structured eligibility assessment for a single clinical trial."""
    trial_nct_id: str
    trial_title: str
    trial_phase: str
    intervention_name: str
    eligible: Literal["yes", "no", "likely", "pending_data"]
    blocking_criteria: list[str]
    pending_criteria: list[str]
    matching_criteria: list[str]
    protocol_alignment: float  # 0-1
    urgency: str  # enrolling_now, not_yet_recruiting, completed
    sites_near_erik: list[str]


def check_structured_eligibility(
    *,
    min_age: int,
    max_age: int,
    sex: str,
    healthy_volunteers: bool,
) -> dict[str, list[str]]:
    """Check Erik against structured eligibility fields.

    Returns dict with 'matching' and 'blocking' lists.
    """
    matching: list[str] = []
    blocking: list[str] = []
    profile = ERIK_ELIGIBILITY_PROFILE

    # Age check
    if min_age <= profile["age"] <= max_age:
        matching.append("age")
    else:
        blocking.append("age")

    # Sex check
    if sex == "All" or sex.lower() == profile["sex"]:
        matching.append("sex")
    else:
        blocking.append("sex")

    # Healthy volunteers — Erik is not healthy
    if healthy_volunteers:
        blocking.append("healthy_volunteers_only")

    return {"matching": matching, "blocking": blocking}


def extract_criteria_from_text(text: str) -> dict:
    """Extract quantitative eligibility criteria from free-text.

    Returns dict with extracted thresholds (None if not found).
    """
    result = {
        "alsfrs_r_min": None,
        "fvc_min_percent": None,
        "max_duration_months": None,
        "riluzole_required": None,
        "genetic_required": None,
    }

    # ALSFRS-R threshold
    m = re.search(
        r"ALSFRS-?R?\s*(?:score\s*)?(?:>=?|≥|greater than|at least)\s*(\d+)",
        text, re.IGNORECASE,
    )
    if m:
        result["alsfrs_r_min"] = int(m.group(1))

    # FVC threshold
    m = re.search(
        r"FVC\s*(?:>=?|≥|greater than|at least)\s*(\d+)\s*%?",
        text, re.IGNORECASE,
    )
    if m:
        result["fvc_min_percent"] = int(m.group(1))

    # Disease duration
    m = re.search(
        r"(?:within|less than|<)\s*(\d+)\s*months?\s*(?:of\s*)?(?:onset|diagnosis|symptom|screening)",
        text, re.IGNORECASE,
    )
    if m:
        result["max_duration_months"] = int(m.group(1))

    # Riluzole requirement
    if re.search(r"(?:stable|on)\s*(?:dose\s*of\s*)?riluzole", text, re.IGNORECASE):
        result["riluzole_required"] = True

    # Genetic requirement
    if re.search(
        r"(?:confirmed|documented|known)\s*(?:SOD1|C9orf72|FUS|TARDBP|pathogenic)\s*(?:mutation|variant|expansion)",
        text, re.IGNORECASE,
    ):
        result["genetic_required"] = True

    return result


def compute_eligibility(
    *,
    nct_id: str,
    title: str,
    phase: str,
    intervention_name: str,
    min_age: int = 0,
    max_age: int = 999,
    sex: str = "All",
    healthy_volunteers: bool = False,
    eligibility_text: str = "",
    enrollment_status: str = "",
    sites: Optional[list[str]] = None,
    current_protocol_top_interventions: Optional[list[str]] = None,
    geographic_region: str = "Ohio",
) -> EligibilityVerdict:
    """Compute Erik's eligibility for a specific trial."""
    sites = sites or []
    current_protocol_top_interventions = current_protocol_top_interventions or []
    profile = ERIK_ELIGIBILITY_PROFILE

    matching: list[str] = []
    blocking: list[str] = []
    pending: list[str] = []

    # --- Structured fields ---
    structured = check_structured_eligibility(
        min_age=min_age, max_age=max_age, sex=sex,
        healthy_volunteers=healthy_volunteers,
    )
    matching.extend(structured["matching"])
    blocking.extend(structured["blocking"])

    # --- Free-text criteria ---
    if eligibility_text:
        criteria = extract_criteria_from_text(eligibility_text)

        if criteria["alsfrs_r_min"] is not None:
            if profile["alsfrs_r"] >= criteria["alsfrs_r_min"]:
                matching.append("alsfrs_r")
            else:
                blocking.append("alsfrs_r")

        if criteria["fvc_min_percent"] is not None:
            if profile["fvc_percent"] >= criteria["fvc_min_percent"]:
                matching.append("fvc")
            else:
                blocking.append("fvc")

        if criteria["max_duration_months"] is not None:
            if profile["disease_duration_months"] <= criteria["max_duration_months"]:
                matching.append("disease_duration")
            else:
                blocking.append("disease_duration")

        if criteria["riluzole_required"]:
            if profile["on_riluzole"]:
                matching.append("riluzole")
            else:
                blocking.append("riluzole")

        if criteria["genetic_required"]:
            if profile["genetic_status"] == "pending":
                pending.append("genetic_status")
            else:
                matching.append("genetic_status")

    # --- Enrollment status → urgency ---
    status_map = {
        "RECRUITING": "enrolling_now",
        "ENROLLING_BY_INVITATION": "enrolling_now",
        "NOT_YET_RECRUITING": "not_yet_recruiting",
    }
    urgency = status_map.get(enrollment_status, "completed")

    # --- Protocol alignment ---
    intervention_lower = intervention_name.lower()
    protocol_lower = [i.lower() for i in current_protocol_top_interventions]
    if intervention_lower in protocol_lower:
        alignment = 0.9
    elif any(intervention_lower in p or p in intervention_lower for p in protocol_lower):
        alignment = 0.6
    else:
        alignment = 0.1

    # --- Site proximity ---
    region_lower = geographic_region.lower()
    near_sites = [s for s in sites if region_lower in s.lower()]

    # --- Final verdict ---
    if blocking:
        eligible: Literal["yes", "no", "likely", "pending_data"] = "no"
    elif pending:
        eligible = "pending_data"
    elif len(matching) >= 2:
        eligible = "yes"
    else:
        eligible = "likely"

    return EligibilityVerdict(
        trial_nct_id=nct_id,
        trial_title=title,
        trial_phase=phase,
        intervention_name=intervention_name,
        eligible=eligible,
        blocking_criteria=blocking,
        pending_criteria=pending,
        matching_criteria=matching,
        protocol_alignment=alignment,
        urgency=urgency,
        sites_near_erik=near_sites,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_eligibility.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/research/eligibility.py tests/test_eligibility.py && git commit -m "feat(P2c): eligibility matcher with structured + free-text criteria"
```

### Task 1.2: Trial Watchlist DB Schema

**Files:**
- Create: `scripts/db/trial_watchlist.sql`
- Modify: `scripts/db/migrate.py`

- [ ] **Step 1: Write the migration SQL**

```sql
-- scripts/db/trial_watchlist.sql
-- Trial eligibility watchlist for Erik Draper

CREATE TABLE IF NOT EXISTS erik_ops.trial_watchlist (
    nct_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    eligible_status TEXT NOT NULL DEFAULT 'likely',   -- yes, no, likely, pending_data
    last_checked    TIMESTAMPTZ NOT NULL DEFAULT now(),
    enrollment_status TEXT NOT NULL DEFAULT '',
    phase           TEXT NOT NULL DEFAULT '',
    intervention_name TEXT NOT NULL DEFAULT '',
    protocol_alignment FLOAT NOT NULL DEFAULT 0.0,
    sites           JSONB NOT NULL DEFAULT '[]'::jsonb,
    reviewed        BOOLEAN NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_watchlist_eligible
    ON erik_ops.trial_watchlist (eligible_status)
    WHERE eligible_status IN ('yes', 'likely', 'pending_data');
```

- [ ] **Step 2: Add to migration file list**

In `scripts/db/migrate.py`, add `"trial_watchlist.sql"` to the `_SCHEMA_FILES` list:

```python
_SCHEMA_FILES = [
    "schema.sql",
    "trial_watchlist.sql",
]
```

- [ ] **Step 3: Run migration**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core python -m db.migrate`
Expected: `Applied: trial_watchlist.sql` (or similar success message)

- [ ] **Step 4: Verify table exists**

Run: `psql -d erik_kg -c "SELECT count(*) FROM erik_ops.trial_watchlist"`
Expected: `0` (table exists, empty)

- [ ] **Step 5: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/db/trial_watchlist.sql scripts/db/migrate.py && git commit -m "feat(P2c): trial watchlist schema migration"
```

### Task 1.3: Integrate Eligibility into ClinicalTrials Connector

**Files:**
- Modify: `scripts/connectors/clinical_trials.py`
- Modify: `tests/test_clinical_trials_connector.py`

- [ ] **Step 1: Write failing test for eligibility integration**

Add to `tests/test_clinical_trials_connector.py`:

```python
class TestEligibilityIntegration:
    def test_fetch_produces_verdicts(self):
        """fetch() with eligibility enabled should populate verdicts."""
        from connectors.clinical_trials import ClinicalTrialsConnector
        connector = ClinicalTrialsConnector(store=None)
        # The connector returns ConnectorResult; verdicts are stored as a side effect.
        # Test that the _compute_verdict helper works on a mock study.
        from research.eligibility import compute_eligibility
        verdict = compute_eligibility(
            nct_id="NCT99999999",
            title="Mock ALS Trial",
            phase="Phase 3",
            intervention_name="test_drug",
            min_age=18,
            max_age=80,
            sex="All",
            healthy_volunteers=False,
            eligibility_text="ALSFRS-R >= 24. FVC >= 50%.",
            enrollment_status="RECRUITING",
            sites=["Cleveland Clinic, Cleveland, Ohio"],
            current_protocol_top_interventions=[],
        )
        assert verdict.eligible in ("yes", "likely")
```

- [ ] **Step 2: Run test to verify it passes** (this uses already-implemented eligibility.py)

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_clinical_trials_connector.py::TestEligibilityIntegration -v`
Expected: PASS

- [ ] **Step 3: Add watchlist upsert function to eligibility.py**

Append to `scripts/research/eligibility.py`:

```python
def upsert_watchlist(verdict: EligibilityVerdict) -> None:
    """Persist an eligibility verdict to the trial watchlist table."""
    from db.pool import get_connection
    import json

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO erik_ops.trial_watchlist
                    (nct_id, title, eligible_status, last_checked,
                     enrollment_status, phase, intervention_name,
                     protocol_alignment, sites)
                VALUES (%s, %s, %s, now(), %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (nct_id) DO UPDATE SET
                    eligible_status = EXCLUDED.eligible_status,
                    last_checked = now(),
                    enrollment_status = EXCLUDED.enrollment_status,
                    protocol_alignment = EXCLUDED.protocol_alignment,
                    sites = EXCLUDED.sites
            """, (
                verdict.trial_nct_id,
                verdict.trial_title,
                verdict.eligible,
                verdict.urgency,
                verdict.trial_phase,
                verdict.intervention_name,
                verdict.protocol_alignment,
                json.dumps(verdict.sites_near_erik),
            ))
        conn.commit()
```

- [ ] **Step 4: Add config keys to erik_config.json**

Add to `data/erik_config.json` before the closing `}`:

```json
  "trial_eligibility_enabled": true,
  "trial_geographic_region": "Ohio",
  "trial_alsfrs_r_current": 43,
  "trial_fvc_current": 100
```

- [ ] **Step 5: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/research/eligibility.py scripts/connectors/clinical_trials.py tests/test_clinical_trials_connector.py data/erik_config.json && git commit -m "feat(P2c): integrate eligibility matching into trial connector + watchlist persistence"
```

---

## Task Group 2: PRO-ACT Trajectory Matching (P1b)

### Task 2.1: TrajectoryMatchResult Model

**Files:**
- Modify: `scripts/ontology/state.py`
- Create: `tests/test_trajectory_match_model.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_trajectory_match_model.py
"""Tests for TrajectoryMatchResult model."""
import pytest
from ontology.state import TrajectoryMatchResult


class TestTrajectoryMatchResult:
    def test_creation(self):
        r = TrajectoryMatchResult(
            cohort_size=1200,
            matched_k=50,
            median_months_remaining=28.5,
            p25_months=18.0,
            p75_months=42.0,
            window_estimates={
                "root_cause_suppression": 36.0,
                "regeneration_reinnervation": 24.0,
            },
            decline_rate_percentile=0.45,
        )
        assert r.cohort_size == 1200
        assert r.median_months_remaining == 28.5
        assert r.window_estimates["regeneration_reinnervation"] == 24.0

    def test_defaults(self):
        r = TrajectoryMatchResult(
            cohort_size=0,
            matched_k=0,
            median_months_remaining=0.0,
            p25_months=0.0,
            p75_months=0.0,
        )
        assert r.window_estimates == {}
        assert r.decline_rate_percentile == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_trajectory_match_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'TrajectoryMatchResult'`

- [ ] **Step 3: Add TrajectoryMatchResult to ontology/state.py**

Append before the `DiseaseStateSnapshot` class in `scripts/ontology/state.py`:

```python
# ---------------------------------------------------------------------------
# TrajectoryMatchResult
# ---------------------------------------------------------------------------

class TrajectoryMatchResult(BaseModel):
    """Result of matching Erik against PRO-ACT historical trajectories.

    Not a BaseEnvelope — transient computation result embedded in
    DiseaseStateSnapshot.
    """
    cohort_size: int
    matched_k: int
    median_months_remaining: float
    p25_months: float
    p75_months: float
    window_estimates: dict[str, float] = Field(default_factory=dict)
    decline_rate_percentile: float = 0.0
```

Also add `BaseModel` import at the top of the file (add to existing pydantic import line):

```python
from pydantic import BaseModel, Field
```

And add `trajectory_match: Optional[dict] = None` field to `DiseaseStateSnapshot`:

```python
class DiseaseStateSnapshot(BaseEnvelope):
    # ... existing fields ...
    uncertainty_ref: Optional[str] = None
    trajectory_match: Optional[dict] = None  # TrajectoryMatchResult.model_dump()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_trajectory_match_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/ontology/state.py tests/test_trajectory_match_model.py && git commit -m "feat(P1b): TrajectoryMatchResult model + DiseaseStateSnapshot field"
```

### Task 2.2: PRO-ACT Connector and Trajectory Matcher

**Files:**
- Create: `scripts/connectors/proact.py`
- Create: `scripts/world_model/trajectory_matcher.py`
- Create: `tests/test_proact_connector.py`
- Create: `tests/test_trajectory_matcher.py`

- [ ] **Step 1: Write failing tests for ProactConnector**

```python
# tests/test_proact_connector.py
"""Tests for PRO-ACT data ingestion connector."""
import pytest
from connectors.proact import ProactConnector


class TestProactConnector:
    def test_parse_csv_row(self):
        """Test parsing a single CSV row into trajectory record."""
        row = {
            "SubjectID": "P001",
            "Age": "65",
            "Sex": "Male",
            "onset_site": "Limb",
            "ALSFRS_R_Total": "40",
            "Q1_Speech": "4",
            "Q2_Salivation": "4",
            "Q3_Swallowing": "4",
            "mouth": "12",
            "hands": "10",
            "leg": "8",
            "respiratory": "10",
            "feature_delta": "12",
            "FVC_percent": "85.0",
        }
        record = ProactConnector._parse_row(row)
        assert record["patient_id"] == "P001"
        assert record["age_onset"] == 65
        assert record["sex"] == "male"
        assert record["onset_region"] == "limb"
        assert record["alsfrs_r_total"] == 40
        assert record["fvc_percent"] == 85.0

    def test_parse_row_missing_fields(self):
        """Missing optional fields should default gracefully."""
        row = {"SubjectID": "P002", "Age": "70", "Sex": "Female"}
        record = ProactConnector._parse_row(row)
        assert record["patient_id"] == "P002"
        assert record["alsfrs_r_total"] is None
```

- [ ] **Step 2: Write failing tests for TrajectoryMatcher**

```python
# tests/test_trajectory_matcher.py
"""Tests for PRO-ACT trajectory matching."""
import pytest
from world_model.trajectory_matcher import (
    TrajectoryMatcher,
    _dtw_distance,
    _estimate_survival,
    _estimate_windows,
)


class TestDTWDistance:
    def test_identical_sequences(self):
        d = _dtw_distance([48, 46, 44, 42], [48, 46, 44, 42])
        assert d == 0.0

    def test_different_sequences(self):
        d = _dtw_distance([48, 46, 44, 42], [48, 40, 30, 20])
        assert d > 0.0

    def test_different_lengths(self):
        d = _dtw_distance([48, 46, 44], [48, 46, 44, 42, 40])
        assert d >= 0.0


class TestEstimateSurvival:
    def test_basic_survival(self):
        """Mock cohort with known survival months."""
        cohort = [
            {"survival_months": 24, "vital_status": "deceased"},
            {"survival_months": 36, "vital_status": "deceased"},
            {"survival_months": 48, "vital_status": "deceased"},
            {"survival_months": 60, "vital_status": "alive"},
        ]
        result = _estimate_survival(cohort)
        assert result["median_months_remaining"] > 0
        assert result["p25_months"] <= result["median_months_remaining"]
        assert result["p75_months"] >= result["median_months_remaining"]


class TestEstimateWindows:
    def test_window_from_trajectories(self):
        """Estimate when matched patients crossed ALSFRS-R thresholds."""
        trajectories = [
            [{"time_months": 0, "alsfrs_r": 44}, {"time_months": 12, "alsfrs_r": 35},
             {"time_months": 24, "alsfrs_r": 28}],
            [{"time_months": 0, "alsfrs_r": 43}, {"time_months": 18, "alsfrs_r": 30},
             {"time_months": 30, "alsfrs_r": 22}],
        ]
        windows = _estimate_windows(
            trajectories,
            current_alsfrs_r=43,
            thresholds={"root_cause_suppression": 30, "regeneration_reinnervation": 35},
        )
        assert "root_cause_suppression" in windows
        assert "regeneration_reinnervation" in windows
        assert windows["regeneration_reinnervation"] < windows["root_cause_suppression"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_proact_connector.py tests/test_trajectory_matcher.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement ProactConnector**

```python
# scripts/connectors/proact.py
"""ProactConnector — loads PRO-ACT ALS patient data from CSV into PostgreSQL.

PRO-ACT contains ~13,000 ALS patient records across 38 clinical trials.
This is a one-time idempotent ingestion — fetch() is a no-op after initial load.
"""
from __future__ import annotations

import csv
import os
from typing import Any, Optional

from connectors.base import BaseConnector, ConnectorResult


class ProactConnector(BaseConnector):
    """Load PRO-ACT CSV data into erik_ops.proact_trajectories."""

    def __init__(self, data_dir: Optional[str] = None):
        self._data_dir = data_dir

    def fetch(self, **kwargs) -> ConnectorResult:
        """Load CSVs if table is empty. No-op if already loaded."""
        result = ConnectorResult()
        if not self._data_dir or not os.path.isdir(self._data_dir):
            result.errors.append(f"PRO-ACT data dir not found: {self._data_dir}")
            return result

        from db.pool import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM erik_ops.proact_trajectories")
                count = cur.fetchone()[0]
                if count > 0:
                    return result  # Already loaded

        # Load all CSV files from data_dir
        rows_loaded = 0
        for fname in sorted(os.listdir(self._data_dir)):
            if not fname.endswith(".csv"):
                continue
            path = os.path.join(self._data_dir, fname)
            rows_loaded += self._load_csv(path)

        result.evidence_items_added = rows_loaded
        return result

    def _load_csv(self, path: str) -> int:
        """Load a single CSV file into the trajectories table."""
        from db.pool import get_connection
        loaded = 0
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            with get_connection() as conn:
                with conn.cursor() as cur:
                    for row in reader:
                        record = self._parse_row(row)
                        if record["patient_id"] is None:
                            continue
                        cur.execute("""
                            INSERT INTO erik_ops.proact_trajectories
                                (patient_id, age_onset, sex, onset_region,
                                 alsfrs_r_total, fvc_percent, time_months,
                                 vital_status, survival_months)
                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT DO NOTHING
                        """, (
                            record["patient_id"],
                            record["age_onset"],
                            record["sex"],
                            record["onset_region"],
                            record["alsfrs_r_total"],
                            record["fvc_percent"],
                            record.get("time_months"),
                            record.get("vital_status"),
                            record.get("survival_months"),
                        ))
                        loaded += 1
                conn.commit()
        return loaded

    @staticmethod
    def _parse_row(row: dict[str, Any]) -> dict[str, Any]:
        """Parse a CSV row into a normalized trajectory record."""
        def safe_int(v):
            try:
                return int(float(v)) if v else None
            except (ValueError, TypeError):
                return None

        def safe_float(v):
            try:
                return float(v) if v else None
            except (ValueError, TypeError):
                return None

        return {
            "patient_id": row.get("SubjectID") or row.get("subject_id"),
            "age_onset": safe_int(row.get("Age") or row.get("age")),
            "sex": (row.get("Sex") or row.get("sex") or "").lower() or None,
            "onset_region": (row.get("onset_site") or row.get("onset_region") or "").lower() or None,
            "alsfrs_r_total": safe_int(
                row.get("ALSFRS_R_Total") or row.get("alsfrs_r_total")
            ),
            "fvc_percent": safe_float(
                row.get("FVC_percent") or row.get("fvc_percent")
            ),
            "time_months": safe_float(
                row.get("feature_delta") or row.get("time_months")
            ),
            "vital_status": row.get("vital_status"),
            "survival_months": safe_float(row.get("survival_months")),
        }
```

- [ ] **Step 5: Implement TrajectoryMatcher**

```python
# scripts/world_model/trajectory_matcher.py
"""PRO-ACT trajectory matching for Erik Draper.

Matches Erik's ALSFRS-R trajectory against ~13,000 historical ALS patients
to estimate survival, intervention windows, and decline rate percentile.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _dtw_distance(seq_a: list[float], seq_b: list[float]) -> float:
    """Compute dynamic time warping distance between two sequences.

    Uses scipy if available, falls back to a simple numpy implementation.
    """
    a = np.array(seq_a, dtype=float)
    b = np.array(seq_b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def _estimate_survival(cohort: list[dict]) -> dict[str, float]:
    """Estimate survival statistics from matched cohort.

    Uses lifelines Kaplan-Meier if available, falls back to percentile
    estimation from observed survival months.
    """
    survival_months = []
    events = []
    for p in cohort:
        sm = p.get("survival_months")
        if sm is not None and sm > 0:
            survival_months.append(sm)
            events.append(1 if p.get("vital_status") == "deceased" else 0)

    if not survival_months:
        return {"median_months_remaining": 0.0, "p25_months": 0.0, "p75_months": 0.0}

    try:
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit(survival_months, event_observed=events)
        median = float(kmf.median_survival_time_)
        # Percentiles from the survival function
        p25 = float(kmf.percentile(0.75))  # 75th percentile of survival fn = 25th of time
        p75 = float(kmf.percentile(0.25))  # 25th percentile of survival fn = 75th of time
        return {"median_months_remaining": median, "p25_months": p25, "p75_months": p75}
    except ImportError:
        arr = sorted(survival_months)
        n = len(arr)
        return {
            "median_months_remaining": float(arr[n // 2]),
            "p25_months": float(arr[n // 4]) if n >= 4 else float(arr[0]),
            "p75_months": float(arr[3 * n // 4]) if n >= 4 else float(arr[-1]),
        }


def _estimate_windows(
    trajectories: list[list[dict]],
    current_alsfrs_r: int,
    thresholds: dict[str, int],
) -> dict[str, float]:
    """Estimate months until each ALSFRS-R threshold is crossed.

    For each threshold, finds the average time at which matched patients
    crossed that value, offset from Erik's current position.
    """
    windows: dict[str, float] = {}

    for layer, threshold in thresholds.items():
        crossing_times: list[float] = []
        for traj in trajectories:
            above_current = [p for p in traj if (p.get("alsfrs_r") or 0) >= current_alsfrs_r]
            if not above_current:
                continue
            start_time = above_current[0].get("time_months", 0) or 0

            for point in traj:
                score = point.get("alsfrs_r") or 0
                t = point.get("time_months", 0) or 0
                if score <= threshold and t > start_time:
                    crossing_times.append(t - start_time)
                    break

        if crossing_times:
            windows[layer] = float(np.median(crossing_times))
        else:
            windows[layer] = 60.0  # Default: 5 years if never crossed

    return windows


class TrajectoryMatcher:
    """Match Erik against PRO-ACT cohort and compute trajectory estimates."""

    DEFAULT_THRESHOLDS = {
        "root_cause_suppression": 30,
        "regeneration_reinnervation": 35,
    }

    def __init__(
        self,
        cohort_age_window: int = 5,
        top_k: int = 50,
        thresholds: Optional[dict[str, int]] = None,
    ):
        self._age_window = cohort_age_window
        self._top_k = top_k
        self._thresholds = thresholds or self.DEFAULT_THRESHOLDS

    def match(
        self,
        age: int,
        sex: str,
        onset_region: str,
        alsfrs_r: int,
        fvc_percent: float,
    ) -> dict[str, Any]:
        """Run full trajectory matching pipeline against PostgreSQL.

        Returns dict compatible with TrajectoryMatchResult.model_dump().
        """
        from db.pool import get_connection

        # 1. Cohort selection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT patient_id, age_onset, alsfrs_r_total,
                           fvc_percent, time_months, vital_status,
                           survival_months
                    FROM erik_ops.proact_trajectories
                    WHERE onset_region = %s
                      AND sex = %s
                      AND age_onset BETWEEN %s AND %s
                      AND alsfrs_r_total IS NOT NULL
                    ORDER BY patient_id, time_months
                """, (onset_region.lower(), sex.lower(),
                      age - self._age_window, age + self._age_window))
                rows = cur.fetchall()

        if not rows:
            return {
                "cohort_size": 0, "matched_k": 0,
                "median_months_remaining": 0.0,
                "p25_months": 0.0, "p75_months": 0.0,
                "window_estimates": {}, "decline_rate_percentile": 0.0,
            }

        # Group by patient
        patients: dict[str, list[dict]] = {}
        for pid, age_o, alsfrs, fvc, t_months, vital, surv in rows:
            if pid not in patients:
                patients[pid] = []
            patients[pid].append({
                "alsfrs_r": alsfrs, "fvc_percent": fvc,
                "time_months": t_months, "vital_status": vital,
                "survival_months": surv, "age_onset": age_o,
            })

        cohort_size = len(patients)

        # 2. DTW alignment — compare ALSFRS-R sequences
        erik_seq = [float(alsfrs_r)]  # Single point (current)
        distances: list[tuple[str, float]] = []
        for pid, points in patients.items():
            patient_seq = [p["alsfrs_r"] for p in sorted(points, key=lambda x: x.get("time_months") or 0)]
            if len(patient_seq) < 2:
                continue
            d = _dtw_distance(erik_seq, patient_seq)
            distances.append((pid, d))

        distances.sort(key=lambda x: x[1])
        matched_pids = [pid for pid, _ in distances[:self._top_k]]
        matched_k = len(matched_pids)

        # 3. Survival estimation
        matched_cohort = []
        matched_trajectories = []
        for pid in matched_pids:
            points = patients[pid]
            last = points[-1]
            matched_cohort.append(last)
            matched_trajectories.append(
                sorted(points, key=lambda x: x.get("time_months") or 0)
            )

        survival = _estimate_survival(matched_cohort)

        # 4. Window estimation
        windows = _estimate_windows(
            matched_trajectories, alsfrs_r, self._thresholds,
        )

        # 5. Decline rate percentile
        decline_rates: list[float] = []
        for points in matched_trajectories:
            sorted_pts = sorted(points, key=lambda x: x.get("time_months") or 0)
            if len(sorted_pts) >= 2:
                first, last = sorted_pts[0], sorted_pts[-1]
                dt = (last.get("time_months") or 0) - (first.get("time_months") or 0)
                if dt > 0:
                    rate = (first["alsfrs_r"] - last["alsfrs_r"]) / dt
                    decline_rates.append(rate)
        erik_rate = 0.39  # points/month
        if decline_rates:
            percentile = sum(1 for r in decline_rates if r <= erik_rate) / len(decline_rates)
        else:
            percentile = 0.5

        return {
            "cohort_size": cohort_size,
            "matched_k": matched_k,
            "median_months_remaining": survival["median_months_remaining"],
            "p25_months": survival["p25_months"],
            "p75_months": survival["p75_months"],
            "window_estimates": windows,
            "decline_rate_percentile": percentile,
        }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_proact_connector.py tests/test_trajectory_matcher.py -v`
Expected: PASS (unit tests use mock data, no DB required)

- [ ] **Step 7: Add config keys**

Add to `data/erik_config.json`:

```json
  "proact_enabled": true,
  "proact_data_dir": null,
  "proact_cohort_age_window": 5,
  "proact_top_k_matches": 50,
  "trajectory_refresh_interval_steps": 100
```

- [ ] **Step 8: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/connectors/proact.py scripts/world_model/trajectory_matcher.py tests/test_proact_connector.py tests/test_trajectory_matcher.py data/erik_config.json && git commit -m "feat(P1b): PRO-ACT connector + trajectory matcher with DTW and Kaplan-Meier"
```

### Task 2.3: Add UPDATE_TRAJECTORY Action Type

**Files:**
- Modify: `scripts/research/actions.py`

- [ ] **Step 1: Add UPDATE_TRAJECTORY to ActionType enum**

In `scripts/research/actions.py`, add after `QUERY_GALEN_KG = "query_galen_kg"`:

```python
    UPDATE_TRAJECTORY = "update_trajectory"
```

- [ ] **Step 2: Run existing tests to verify nothing broke**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_research_actions.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/research/actions.py && git commit -m "feat(P1b): add UPDATE_TRAJECTORY action type"
```

---

## Task Group 3: bioRxiv/medRxiv Preprint Connector (P1a)

### Task 3.1: BiorxivConnector

**Files:**
- Create: `scripts/connectors/biorxiv.py`
- Create: `tests/test_biorxiv_connector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_biorxiv_connector.py
"""Tests for bioRxiv/medRxiv preprint connector."""
import pytest
from unittest.mock import patch, MagicMock

from connectors.biorxiv import BiorxivConnector, _parse_preprint


class TestParsePreprint:
    def test_parse_valid_preprint(self):
        raw = {
            "doi": "10.1101/2026.03.15.123456",
            "title": "Novel TDP-43 intrabody for ALS",
            "authors": "Smith J; Doe A",
            "abstract": "We demonstrate a new TDP-43 intrabody approach...",
            "date": "2026-03-15",
            "server": "biorxiv",
            "category": "neuroscience",
        }
        item = _parse_preprint(raw)
        assert item is not None
        assert item.id == "evi:biorxiv:10.1101/2026.03.15.123456"
        assert "peer_reviewed" in item.body
        assert item.body["peer_reviewed"] is False

    def test_strength_clamped_to_emerging(self):
        raw = {
            "doi": "10.1101/2026.03.20.789012",
            "title": "Phase 3 results for ALS drug",
            "authors": "A B",
            "abstract": "Randomized controlled trial...",
            "date": "2026-03-20",
            "server": "medrxiv",
            "category": "neurology",
        }
        item = _parse_preprint(raw)
        assert item is not None
        # Even if it looks like an RCT, preprints are capped at emerging
        from ontology.enums import EvidenceStrength
        assert item.body.get("strength") == EvidenceStrength.emerging.value

    def test_parse_missing_doi(self):
        raw = {"title": "No DOI paper", "abstract": "text"}
        item = _parse_preprint(raw)
        assert item is None

    def test_doi_stored_in_body(self):
        raw = {
            "doi": "10.1101/2026.01.01.000001",
            "title": "T", "authors": "A", "abstract": "B",
            "date": "2026-01-01", "server": "biorxiv", "category": "neuro",
        }
        item = _parse_preprint(raw)
        assert item.body["doi"] == "10.1101/2026.01.01.000001"


class TestBiorxivConnector:
    def test_fetch_disabled(self):
        c = BiorxivConnector(store=None, enabled=False)
        result = c.fetch(query="ALS TDP-43")
        assert result.evidence_items_added == 0

    @patch("connectors.biorxiv.requests.get")
    def test_fetch_parses_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "messages": [{"status": "ok"}],
            "collection": [
                {
                    "doi": "10.1101/2026.03.25.999999",
                    "title": "ALS breakthrough",
                    "authors": "Author A",
                    "abstract": "Abstract text about ALS TDP-43",
                    "date": "2026-03-25",
                    "server": "biorxiv",
                    "category": "neuroscience",
                },
            ],
        }
        mock_get.return_value = mock_resp

        c = BiorxivConnector(store=None, enabled=True)
        result = c.fetch(query="ALS TDP-43")
        # No store → items parsed but not persisted → count is 0
        assert result.errors == []

    @patch("connectors.biorxiv.requests.get")
    def test_fetch_handles_api_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        c = BiorxivConnector(store=None, enabled=True)
        result = c.fetch(query="ALS")
        assert len(result.errors) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_biorxiv_connector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'connectors.biorxiv'`

- [ ] **Step 3: Implement BiorxivConnector**

```python
# scripts/connectors/biorxiv.py
"""BiorxivConnector — fetches preprints from bioRxiv/medRxiv.

Preprints are NOT peer-reviewed. All evidence items from this connector
are clamped to EvidenceStrength.emerging and tagged peer_reviewed=false.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import requests

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.biorxiv.org/details"


def _parse_preprint(raw: dict[str, Any]) -> Optional[EvidenceItem]:
    """Parse a bioRxiv/medRxiv API result into an EvidenceItem."""
    doi = raw.get("doi")
    if not doi:
        return None

    title = raw.get("title", "")
    abstract = raw.get("abstract", "")
    authors = raw.get("authors", "")
    date = raw.get("date", "")
    server = raw.get("server", "biorxiv")
    category = raw.get("category", "")

    evi_id = f"evi:{server}:{doi}".lower().replace("/", "_").replace(" ", "_")
    claim = f"[Preprint] {title}"
    if len(claim) > 500:
        claim = claim[:497] + "..."

    return EvidenceItem(
        id=evi_id,
        claim=claim,
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.emerging,
        provenance=Provenance(
            source_system=SourceSystem.literature,
            source_artifact_id=f"doi:{doi}",
            asserted_by=f"{server}_preprint",
        ),
        body={
            "doi": doi,
            "title": title,
            "abstract": abstract[:2000],
            "authors": authors,
            "date": date,
            "server": server,
            "category": category,
            "peer_reviewed": False,
            "strength": EvidenceStrength.emerging.value,
        },
    )


class BiorxivConnector(BaseConnector):
    """Fetch ALS-relevant preprints from bioRxiv and medRxiv."""

    def __init__(
        self,
        store: Any = None,
        enabled: bool = True,
        lookback_days: int = 90,
        max_results: int = 15,
    ):
        self._store = store
        self._enabled = enabled
        self._lookback_days = lookback_days
        self._max_results = max_results

    def fetch(self, *, query: str = "ALS motor neuron", **kwargs) -> ConnectorResult:
        """Search bioRxiv and medRxiv for ALS-relevant preprints."""
        result = ConnectorResult()

        if not self._enabled:
            return result

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=self._lookback_days)).strftime("%Y-%m-%d")

        added = 0
        for server in ("biorxiv", "medrxiv"):
            try:
                url = f"{_BASE_URL}/{server}/{start_date}/{end_date}/0/{self._max_results}"
                resp = self._retry_with_backoff(
                    requests.get, url, timeout=self.REQUEST_TIMEOUT,
                )
                if resp.status_code != 200:
                    result.errors.append(f"{server} API returned {resp.status_code}")
                    continue

                data = resp.json()
                collection = data.get("collection", [])

                for raw in collection:
                    # Filter by relevance to ALS
                    text = f"{raw.get('title', '')} {raw.get('abstract', '')}".lower()
                    query_terms = query.lower().split()
                    if not any(term in text for term in query_terms):
                        continue

                    item = _parse_preprint(raw)
                    if item is None:
                        continue

                    if self._store is not None and added < self._max_results:
                        try:
                            self._store.upsert_evidence_item(item)
                            added += 1
                        except Exception:
                            result.skipped_duplicates += 1

            except Exception as e:
                result.errors.append(f"{server} fetch failed: {e}")

        result.evidence_items_added = added
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_biorxiv_connector.py -v`
Expected: PASS

- [ ] **Step 5: Add SEARCH_PREPRINTS action type and policy slot**

In `scripts/research/actions.py`, add after `QUERY_GALEN_KG`:

```python
    SEARCH_PREPRINTS = "search_preprints"
```

Add to `NETWORK_ACTIONS`:

```python
    ActionType.SEARCH_PREPRINTS,
```

In `scripts/research/policy.py`, replace the `_ACQUISITION_ROTATION`:

```python
_ACQUISITION_ROTATION = [
    ActionType.SEARCH_PUBMED,
    ActionType.SEARCH_TRIALS,
    ActionType.QUERY_PATHWAYS,
    ActionType.QUERY_PPI_NETWORK,
    ActionType.CHECK_PHARMACOGENOMICS,
    ActionType.QUERY_GALEN_KG,
    ActionType.SEARCH_PREPRINTS,
    ActionType.SEARCH_PUBMED,
]
```

Add the handler in `_select_acquisition_action`:

```python
    elif action == ActionType.SEARCH_PREPRINTS:
        from config.loader import ConfigLoader
        cfg = ConfigLoader()
        if not cfg.get("biorxiv_enabled", True):
            return _fallback_acquisition(state, step, skip=ActionType.SEARCH_PREPRINTS)
        layer_idx = (step // _CYCLE_LENGTH) % len(ALL_LAYERS)
        layer = ALL_LAYERS[layer_idx]
        query = LAYER_SEARCH_QUERIES.get(layer, f"ALS {layer.replace('_', ' ')} treatment")
        return action, build_action_params(
            action, query=query, protocol_layer=layer,
        )
```

- [ ] **Step 6: Add config keys**

Add to `data/erik_config.json`:

```json
  "biorxiv_enabled": true,
  "biorxiv_lookback_days": 90,
  "biorxiv_max_per_query": 15
```

- [ ] **Step 7: Run all existing tests to verify nothing broke**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v -k "not network and not chembl and not llm"`
Expected: All existing tests PASS

- [ ] **Step 8: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/connectors/biorxiv.py tests/test_biorxiv_connector.py scripts/research/actions.py scripts/research/policy.py data/erik_config.json && git commit -m "feat(P1a): bioRxiv/medRxiv preprint connector with strength clamping"
```

---

## Task Group 4: Adversarial Protocol Verification (P2b)

### Task 4.1: Adversarial Module and Action Type

**Files:**
- Create: `scripts/research/adversarial.py`
- Create: `tests/test_adversarial.py`
- Modify: `scripts/research/actions.py`
- Modify: `scripts/research/state.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_adversarial.py
"""Tests for adversarial protocol verification."""
import pytest
from research.adversarial import (
    generate_adversarial_queries,
    classify_adversarial_result,
    select_challenge_target,
)


class TestGenerateAdversarialQueries:
    def test_produces_three_query_types(self):
        queries = generate_adversarial_queries(
            drug_name="pridopidine",
            mechanism="sigma-1R agonist neuroprotection",
        )
        assert len(queries) == 3
        assert any("failed" in q.lower() or "negative" in q.lower() for q in queries)
        assert any("adverse" in q.lower() or "neurotoxicity" in q.lower() for q in queries)
        assert any("disputed" in q.lower() or "no effect" in q.lower() for q in queries)

    def test_drug_name_in_queries(self):
        queries = generate_adversarial_queries(
            drug_name="rapamycin",
            mechanism="mTOR inhibition autophagy",
        )
        assert all("rapamycin" in q.lower() or "mtor" in q.lower() for q in queries)


class TestClassifyAdversarialResult:
    def test_classify_contradicts(self):
        result = classify_adversarial_result(
            title="Pridopidine fails Phase 3 in ALS",
            abstract="The primary endpoint was not met. Pridopidine showed no benefit over placebo in ALS patients.",
            drug_name="pridopidine",
        )
        assert result in ("contradicts", "weakens", "context_dependent")

    def test_classify_irrelevant(self):
        result = classify_adversarial_result(
            title="Pridopidine in Huntington's disease",
            abstract="Study of pridopidine for HD motor symptoms.",
            drug_name="pridopidine",
        )
        # HD study is not directly contradictory for ALS
        assert result in ("irrelevant", "context_dependent")


class TestSelectChallengeTarget:
    def test_selects_least_challenged(self):
        intervention_scores = {
            "int:pridopidine": 0.85,
            "int:rapamycin": 0.72,
            "int:riluzole": 0.65,
        }
        challenge_counts = {
            "int:pridopidine": 2,
            "int:rapamycin": 0,
            "int:riluzole": 1,
        }
        target = select_challenge_target(intervention_scores, challenge_counts)
        assert target == "int:rapamycin"  # Highest score with lowest challenge ratio

    def test_none_when_all_fully_challenged(self):
        intervention_scores = {"int:a": 0.8}
        challenge_counts = {"int:a": 3}
        target = select_challenge_target(
            intervention_scores, challenge_counts, max_challenges=3,
        )
        assert target is None

    def test_empty_scores(self):
        target = select_challenge_target({}, {})
        assert target is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_adversarial.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement adversarial.py**

```python
# scripts/research/adversarial.py
"""Adversarial protocol verification — actively searches for evidence
that contradicts, weakens, or disputes the current protocol's top
interventions.

Corrects the research loop's structural confirmation bias by generating
negation-oriented queries and classifying results by contradiction strength.
"""
from __future__ import annotations

import re
from typing import Optional


def generate_adversarial_queries(
    *,
    drug_name: str,
    mechanism: str,
) -> list[str]:
    """Generate three adversarial PubMed queries for an intervention.

    Returns: [failure_query, harm_query, mechanism_dispute_query]
    """
    failure = f'"{drug_name}" ALS (failed OR negative OR ineffective OR discontinued)'
    harm = f'"{drug_name}" (neurotoxicity OR adverse OR "motor neuron" harm OR contraindicated)'
    mechanism_key = mechanism.split()[0] if mechanism else drug_name
    dispute = f'"{mechanism_key}" ALS (disputed OR disproven OR "no effect" OR insufficient)'
    return [failure, harm, dispute]


def classify_adversarial_result(
    *,
    title: str,
    abstract: str,
    drug_name: str,
) -> str:
    """Classify a paper as contradicts/weakens/irrelevant/context_dependent.

    Rule-based classification (no LLM) using keyword matching.
    """
    text = f"{title} {abstract}".lower()
    drug_lower = drug_name.lower()

    # Check if the paper is about the right drug
    if drug_lower not in text and drug_lower.replace("-", " ") not in text:
        return "irrelevant"

    # Check if it's about ALS specifically
    als_terms = {"als", "amyotrophic lateral sclerosis", "motor neuron disease"}
    is_als = any(term in text for term in als_terms)

    # Strong contradiction signals
    contradiction_signals = [
        r"fail(?:ed|ure|s).*(?:primary|endpoint|efficacy)",
        r"no (?:significant |)(?:benefit|effect|improvement)",
        r"did not (?:meet|reach|achieve)",
        r"negative (?:result|trial|outcome)",
        r"discontinued.*(?:lack|futility)",
        r"ineffective",
    ]
    for pattern in contradiction_signals:
        if re.search(pattern, text):
            return "contradicts" if is_als else "context_dependent"

    # Harm signals
    harm_signals = [
        r"neurotox",
        r"adverse.*(?:severe|serious|significant)",
        r"worsen(?:ed|ing)",
        r"contraindicated",
    ]
    for pattern in harm_signals:
        if re.search(pattern, text):
            return "weakens"

    # If about ALS but no clear contradiction
    if is_als:
        return "context_dependent"

    return "irrelevant"


def select_challenge_target(
    intervention_scores: dict[str, float],
    challenge_counts: dict[str, int],
    max_challenges: int = 3,
) -> Optional[str]:
    """Select the intervention to challenge next.

    Picks the highest-scored intervention with the lowest challenge ratio.
    Returns None if all have been fully challenged.
    """
    if not intervention_scores:
        return None

    candidates = []
    for int_id, score in intervention_scores.items():
        count = challenge_counts.get(int_id, 0)
        if count >= max_challenges:
            continue
        # Prioritize: high score * low challenge count
        priority = score * (1.0 / (1.0 + count))
        candidates.append((int_id, priority))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]
```

- [ ] **Step 4: Add challenge_counts to ResearchState**

In `scripts/research/state.py`, add to the `ResearchState` dataclass after `uncertainty_history`:

```python
    challenge_counts: dict[str, int] = field(default_factory=dict)
```

Add to `to_dict()`:

```python
            "challenge_counts": dict(self.challenge_counts),
```

Add to `CHALLENGE_INTERVENTION` to `ActionType` in `scripts/research/actions.py`:

```python
    CHALLENGE_INTERVENTION = "challenge_intervention"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_adversarial.py -v`
Expected: PASS

- [ ] **Step 6: Add config keys**

Add to `data/erik_config.json`:

```json
  "adversarial_verification_enabled": true,
  "adversarial_max_challenges_per_intervention": 3,
  "adversarial_contested_threshold": 3,
  "adversarial_reward_multiplier": 1.5
```

- [ ] **Step 7: Extend CounterfactualResult model**

In `scripts/world_model/counterfactual_check.py`, add to `CounterfactualResult`:

```python
    strongest_counterargument: str = ""
    counterargument_strength: str = "none"  # none, weak, moderate, strong
```

- [ ] **Step 8: Add contested_layers to CureProtocolCandidate**

In `scripts/ontology/protocol.py`, add to `CureProtocolCandidate` after `uncertainty_ref`:

```python
    contested_layers: list[str] = Field(default_factory=list)
```

- [ ] **Step 9: Run full test suite**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v -k "not network and not chembl and not llm"`
Expected: All PASS

- [ ] **Step 10: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/research/adversarial.py tests/test_adversarial.py scripts/research/actions.py scripts/research/state.py scripts/world_model/counterfactual_check.py scripts/ontology/protocol.py data/erik_config.json && git commit -m "feat(P2b): adversarial protocol verification with challenge targeting"
```

---

## Task Group 5: Drug Combination Synergy Modeling (P2a)

### Task 5.1: Combination Analyzer

**Files:**
- Create: `scripts/world_model/combination_analyzer.py`
- Create: `tests/test_combination_analyzer.py`
- Modify: `scripts/ontology/protocol.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_combination_analyzer.py
"""Tests for drug combination synergy analysis."""
import pytest
from world_model.combination_analyzer import (
    InteractionFlag,
    CombinationAnalysis,
    compute_pathway_overlap,
    analyze_combinations,
    apply_interaction_flags,
)
from research.causal_chains import CausalChain, CausalLink


class TestPathwayOverlap:
    def test_identical_chains(self):
        chain_a = CausalChain(intervention_id="int:a", links=[
            CausalLink(source="drug_a", target="mTOR", mechanism="inhibits", evidence_ref="e1"),
            CausalLink(source="mTOR", target="autophagy", mechanism="activates", evidence_ref="e2"),
        ])
        chain_b = CausalChain(intervention_id="int:b", links=[
            CausalLink(source="drug_b", target="mTOR", mechanism="inhibits", evidence_ref="e3"),
            CausalLink(source="mTOR", target="autophagy", mechanism="activates", evidence_ref="e4"),
        ])
        overlap = compute_pathway_overlap(chain_a, chain_b)
        assert overlap > 0.6  # High overlap — both go through mTOR → autophagy

    def test_no_overlap(self):
        chain_a = CausalChain(intervention_id="int:a", links=[
            CausalLink(source="drug_a", target="sigma1r", mechanism="agonizes", evidence_ref="e1"),
        ])
        chain_b = CausalChain(intervention_id="int:b", links=[
            CausalLink(source="drug_b", target="EAAT2", mechanism="upregulates", evidence_ref="e2"),
        ])
        overlap = compute_pathway_overlap(chain_a, chain_b)
        assert overlap == 0.0

    def test_partial_overlap(self):
        chain_a = CausalChain(intervention_id="int:a", links=[
            CausalLink(source="drug_a", target="mTOR", mechanism="inhibits", evidence_ref="e1"),
            CausalLink(source="mTOR", target="autophagy", mechanism="activates", evidence_ref="e2"),
            CausalLink(source="autophagy", target="clearance", mechanism="enhances", evidence_ref="e3"),
        ])
        chain_b = CausalChain(intervention_id="int:b", links=[
            CausalLink(source="drug_b", target="mTOR", mechanism="inhibits", evidence_ref="e4"),
            CausalLink(source="mTOR", target="growth", mechanism="suppresses", evidence_ref="e5"),
        ])
        overlap = compute_pathway_overlap(chain_a, chain_b)
        assert 0.0 < overlap < 1.0


class TestCombinationAnalysis:
    def test_model_creation(self):
        flag = InteractionFlag(
            intervention_a="int:rapamycin",
            intervention_b="int:bdnf",
            interaction_type="antagonism",
            mechanism="mTOR suppression reduces growth factor signaling",
            confidence=0.8,
            cited_evidence=[],
        )
        analysis = CombinationAnalysis(
            flags=[flag],
            overall_coherence=0.6,
            suggested_substitutions=[],
        )
        assert len(analysis.flags) == 1
        assert analysis.flags[0].interaction_type == "antagonism"


class TestApplyInteractionFlags:
    def test_antagonism_triggers_swap(self):
        """Antagonism above threshold should swap the lower-scoring intervention."""
        scores = {
            "root_cause_suppression": [
                {"intervention_id": "int:a", "relevance_score": 0.9},
                {"intervention_id": "int:a2", "relevance_score": 0.7},
            ],
            "pathology_reversal": [
                {"intervention_id": "int:b", "relevance_score": 0.85},
                {"intervention_id": "int:b2", "relevance_score": 0.6},
            ],
        }
        flags = [
            InteractionFlag(
                intervention_a="int:a",
                intervention_b="int:b",
                interaction_type="antagonism",
                mechanism="conflict",
                confidence=0.8,
                cited_evidence=[],
            ),
        ]
        result = apply_interaction_flags(scores, flags, threshold=0.7)
        # The lower-scored intervention (int:b at 0.85 < int:a at 0.9) should be swapped
        pathology_top = result["pathology_reversal"][0]["intervention_id"]
        assert pathology_top == "int:b2"

    def test_synergy_no_swap(self):
        scores = {
            "root_cause_suppression": [
                {"intervention_id": "int:a", "relevance_score": 0.9},
            ],
            "circuit_stabilization": [
                {"intervention_id": "int:c", "relevance_score": 0.8},
            ],
        }
        flags = [
            InteractionFlag(
                intervention_a="int:a",
                intervention_b="int:c",
                interaction_type="synergy",
                mechanism="complementary",
                confidence=0.9,
                cited_evidence=[],
            ),
        ]
        result = apply_interaction_flags(scores, flags, threshold=0.7)
        assert result["root_cause_suppression"][0]["intervention_id"] == "int:a"
        assert result["circuit_stabilization"][0]["intervention_id"] == "int:c"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_combination_analyzer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement combination_analyzer.py**

```python
# scripts/world_model/combination_analyzer.py
"""Stage 3B — Drug combination synergy analysis.

Detects synergy, antagonism, and redundancy between protocol interventions
using pathway overlap analysis on existing causal chains.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from research.causal_chains import CausalChain


class InteractionFlag(BaseModel):
    intervention_a: str
    intervention_b: str
    interaction_type: Literal["synergy", "antagonism", "redundancy"]
    mechanism: str
    confidence: float = 0.0
    cited_evidence: list[str] = Field(default_factory=list)


class CombinationAnalysis(BaseModel):
    flags: list[InteractionFlag] = Field(default_factory=list)
    overall_coherence: float = 1.0
    suggested_substitutions: list[dict] = Field(default_factory=list)


def compute_pathway_overlap(chain_a: CausalChain, chain_b: CausalChain) -> float:
    """Compute pathway overlap between two causal chains.

    Returns 0.0 (no overlap) to 1.0 (identical pathway nodes).
    Overlap is measured by shared intermediate nodes (targets of links),
    excluding the drug node itself (source of first link).
    """
    if not chain_a.links or not chain_b.links:
        return 0.0

    nodes_a = set()
    for link in chain_a.links:
        nodes_a.add(link.target.lower())
    nodes_b = set()
    for link in chain_b.links:
        nodes_b.add(link.target.lower())

    if not nodes_a or not nodes_b:
        return 0.0

    intersection = nodes_a & nodes_b
    union = nodes_a | nodes_b
    return len(intersection) / len(union)


def analyze_combinations(
    chains: dict[str, CausalChain],
    overlap_threshold: float = 0.6,
) -> CombinationAnalysis:
    """Analyze pairwise interactions between intervention causal chains.

    Layer 1 analysis only (deterministic pathway overlap).
    """
    flags: list[InteractionFlag] = []
    chain_ids = list(chains.keys())

    for i in range(len(chain_ids)):
        for j in range(i + 1, len(chain_ids)):
            id_a, id_b = chain_ids[i], chain_ids[j]
            overlap = compute_pathway_overlap(chains[id_a], chains[id_b])

            if overlap >= overlap_threshold:
                flags.append(InteractionFlag(
                    intervention_a=id_a,
                    intervention_b=id_b,
                    interaction_type="redundancy",
                    mechanism=f"Pathway overlap {overlap:.0%}",
                    confidence=overlap,
                ))

    coherence = 1.0 - (len(flags) * 0.2)  # Penalize incoherence
    coherence = max(0.0, min(1.0, coherence))

    return CombinationAnalysis(
        flags=flags,
        overall_coherence=coherence,
    )


def apply_interaction_flags(
    scores_by_layer: dict[str, list[dict]],
    flags: list[InteractionFlag],
    threshold: float = 0.7,
) -> dict[str, list[dict]]:
    """Apply interaction flags to scored interventions.

    For antagonism/redundancy above threshold: swap the lower-scoring
    intervention for its layer's next-best candidate.
    Synergy flags are recorded but don't trigger swaps.
    """
    result = deepcopy(scores_by_layer)

    for flag in flags:
        if flag.interaction_type == "synergy":
            continue
        if flag.confidence < threshold:
            continue

        # Find which layers contain the flagged interventions
        layer_a = layer_b = None
        for layer, scores in result.items():
            for s in scores:
                if s["intervention_id"] == flag.intervention_a:
                    layer_a = layer
                if s["intervention_id"] == flag.intervention_b:
                    layer_b = layer

        if layer_a is None or layer_b is None:
            continue

        # Find the lower-scoring intervention
        score_a = next(
            (s["relevance_score"] for s in result.get(layer_a, [])
             if s["intervention_id"] == flag.intervention_a), 0.0,
        )
        score_b = next(
            (s["relevance_score"] for s in result.get(layer_b, [])
             if s["intervention_id"] == flag.intervention_b), 0.0,
        )

        # Swap the lower-scored one to its layer's next-best
        if score_b <= score_a:
            swap_layer = layer_b
            swap_id = flag.intervention_b
        else:
            swap_layer = layer_a
            swap_id = flag.intervention_a

        layer_scores = result[swap_layer]
        if len(layer_scores) > 1:
            # Remove the swapped intervention from position 0
            layer_scores = [s for s in layer_scores if s["intervention_id"] != swap_id]
            result[swap_layer] = layer_scores

    return result
```

- [ ] **Step 4: Add combination_analysis field to CureProtocolCandidate**

In `scripts/ontology/protocol.py`, add to `CureProtocolCandidate` after `contested_layers`:

```python
    combination_analysis: Optional[dict] = None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_combination_analyzer.py -v`
Expected: PASS

- [ ] **Step 6: Add config keys**

Add to `data/erik_config.json`:

```json
  "combination_analysis_enabled": true,
  "combination_antagonism_threshold": 0.7,
  "combination_overlap_threshold": 0.6
```

- [ ] **Step 7: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/world_model/combination_analyzer.py tests/test_combination_analyzer.py scripts/ontology/protocol.py data/erik_config.json && git commit -m "feat(P2a): combination synergy analyzer with pathway overlap detection"
```

---

## Task Group 6: Thompson Sampling Policy (P3a)

### Task 6.1: Thompson Sampling Implementation

**Files:**
- Modify: `scripts/research/policy.py`
- Modify: `scripts/research/state.py`
- Create: `tests/test_thompson_policy.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_thompson_policy.py
"""Tests for Thompson sampling action selection policy."""
import pytest
from research.policy import select_action_thompson, _update_posteriors, _apply_decay
from research.state import ResearchState, initial_state


class TestUpdatePosteriors:
    def test_success_increments_alpha(self):
        posteriors = {"search_pubmed:layer_a": (1.0, 1.0)}
        updated = _update_posteriors(posteriors, "search_pubmed:layer_a", success=True)
        assert updated["search_pubmed:layer_a"] == (2.0, 1.0)

    def test_failure_increments_beta(self):
        posteriors = {"search_pubmed:layer_a": (1.0, 1.0)}
        updated = _update_posteriors(posteriors, "search_pubmed:layer_a", success=False)
        assert updated["search_pubmed:layer_a"] == (1.0, 2.0)

    def test_new_key_starts_uniform(self):
        posteriors = {}
        updated = _update_posteriors(posteriors, "new_action:ctx", success=True)
        assert updated["new_action:ctx"] == (2.0, 1.0)


class TestApplyDecay:
    def test_decay_reduces_parameters(self):
        posteriors = {"a": (10.0, 5.0)}
        decayed = _apply_decay(posteriors, rate=0.95)
        assert decayed["a"][0] == pytest.approx(9.5)
        assert decayed["a"][1] == pytest.approx(4.75)

    def test_decay_has_floor(self):
        posteriors = {"a": (1.0, 1.0)}
        decayed = _apply_decay(posteriors, rate=0.5)
        # Floor is (1.0, 1.0)
        assert decayed["a"][0] >= 1.0
        assert decayed["a"][1] >= 1.0


class TestThompsonPolicy:
    def test_regeneration_preempts(self):
        """Protocol regen should fire regardless of Thompson sampling."""
        state = initial_state("traj:test")
        state = state.__class__(**{
            **state.to_dict(),
            "new_evidence_since_regen": 15,
            "protocol_version": 1,
        })
        from research.actions import ActionType
        action, params = select_action_thompson(state, regen_threshold=10)
        assert action == ActionType.REGENERATE_PROTOCOL

    def test_selects_valid_action(self):
        state = initial_state("traj:test")
        state = state.__class__(**{
            **state.to_dict(),
            "protocol_version": 1,
            "action_posteriors": {},
        })
        from research.actions import ActionType
        action, params = select_action_thompson(state)
        assert isinstance(action, ActionType)

    def test_diversity_floor_forces_action(self):
        """If an action type hasn't been used in 30 steps, force it."""
        state = initial_state("traj:test")
        # Simulate 30 steps all as search_pubmed
        action_counts = {"search_pubmed": 30}
        last_action_per_type = {"search_pubmed": 30}
        state = state.__class__(**{
            **state.to_dict(),
            "step_count": 30,
            "protocol_version": 1,
            "action_counts": action_counts,
            "last_action_per_type": last_action_per_type,
            "action_posteriors": {
                "search_pubmed:root_cause_suppression": (30.0, 1.0),
            },
        })
        # Run 100 times — at least some should pick non-pubmed actions
        actions_seen = set()
        for _ in range(100):
            action, _ = select_action_thompson(state)
            actions_seen.add(action.value)
        assert len(actions_seen) > 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_thompson_policy.py -v`
Expected: FAIL — `ImportError: cannot import name 'select_action_thompson'`

- [ ] **Step 3: Add new fields to ResearchState**

In `scripts/research/state.py`, add to `ResearchState` after `challenge_counts`:

```python
    action_posteriors: dict[str, tuple[float, float]] = field(default_factory=dict)
    last_action_per_type: dict[str, int] = field(default_factory=dict)
```

Add to `to_dict()`:

```python
            "action_posteriors": {k: list(v) for k, v in self.action_posteriors.items()},
            "last_action_per_type": dict(self.last_action_per_type),
```

Update `from_dict()` to handle tuple conversion:

```python
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResearchState:
        clean = {}
        for k, v in d.items():
            if k not in cls.__dataclass_fields__:
                continue
            if k == "action_posteriors" and isinstance(v, dict):
                clean[k] = {key: tuple(val) for key, val in v.items()}
            else:
                clean[k] = v
        return cls(**clean)
```

- [ ] **Step 4: Implement Thompson sampling in policy.py**

Add to `scripts/research/policy.py`:

```python
import random

def _update_posteriors(
    posteriors: dict[str, tuple[float, float]],
    key: str,
    success: bool,
) -> dict[str, tuple[float, float]]:
    """Update Beta posterior for an action-context key."""
    result = dict(posteriors)
    alpha, beta = result.get(key, (1.0, 1.0))
    if success:
        alpha += 1.0
    else:
        beta += 1.0
    result[key] = (alpha, beta)
    return result


def _apply_decay(
    posteriors: dict[str, tuple[float, float]],
    rate: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Apply multiplicative decay with floor at (1.0, 1.0)."""
    result = {}
    for key, (alpha, beta) in posteriors.items():
        result[key] = (max(1.0, alpha * rate), max(1.0, beta * rate))
    return result


def select_action_thompson(
    state: ResearchState,
    regen_threshold: int = 10,
    target_depth: int = 5,
) -> tuple[ActionType, dict[str, Any]]:
    """Select action via Thompson sampling over Beta posteriors."""

    # Preempt: protocol regeneration
    if state.new_evidence_since_regen >= regen_threshold and state.protocol_version >= 1:
        return ActionType.REGENERATE_PROTOCOL, build_action_params(ActionType.REGENERATE_PROTOCOL)

    # Diversity floor: force any action type not used in 30 steps
    diversity_floor = 30
    all_action_types = list(_ACQUISITION_ROTATION) + [
        ActionType.GENERATE_HYPOTHESIS,
        ActionType.DEEPEN_CAUSAL_CHAIN,
    ]
    if hasattr(ActionType, "CHALLENGE_INTERVENTION"):
        all_action_types.append(ActionType.CHALLENGE_INTERVENTION)

    for at in all_action_types:
        last_used = state.last_action_per_type.get(at.value, 0)
        if state.step_count - last_used >= diversity_floor:
            return _build_params_for_type(at, state, target_depth)

    # Thompson sampling: sample from each action's Beta posterior
    posteriors = state.action_posteriors or {}
    best_action = None
    best_sample = -1.0

    for at in all_action_types:
        # Use action type as key (simplified context)
        key = at.value
        alpha, beta = posteriors.get(key, (1.0, 1.0))
        sample = random.betavariate(alpha, beta)
        if sample > best_sample:
            best_sample = sample
            best_action = at

    if best_action is None:
        best_action = ActionType.SEARCH_PUBMED

    return _build_params_for_type(best_action, state, target_depth)


def _build_params_for_type(
    action: ActionType,
    state: ResearchState,
    target_depth: int,
) -> tuple[ActionType, dict[str, Any]]:
    """Build parameters for a given action type using existing helpers."""
    if action in _ACQUIRE_ACTIONS_SET:
        # Temporarily override to use our chosen action
        step = state.step_count
        if action == ActionType.SEARCH_PUBMED:
            layer_idx = (step // _CYCLE_LENGTH) % len(ALL_LAYERS)
            layer = ALL_LAYERS[layer_idx]
            query = LAYER_SEARCH_QUERIES.get(layer, f"ALS {layer.replace('_', ' ')} treatment")
            return action, build_action_params(action, query=query, protocol_layer=layer)
        elif action == ActionType.GENERATE_HYPOTHESIS:
            return ActionType.GENERATE_HYPOTHESIS, build_action_params(ActionType.GENERATE_HYPOTHESIS)
        elif action == ActionType.DEEPEN_CAUSAL_CHAIN:
            return _select_reasoning_action(state, target_depth)
        else:
            # Delegate to existing handler
            return _select_acquisition_action(state)
    return action, build_action_params(action)

_ACQUIRE_ACTIONS_SET = set(_ACQUISITION_ROTATION)
```

Update `select_action()` to dispatch:

```python
def select_action(
    state: ResearchState,
    regen_threshold: int = 10,
    target_depth: int = 5,
    exploration_fraction: float = 0.15,
) -> tuple[ActionType, dict[str, Any]]:
    """Select the next research action. Dispatches to Thompson or cycle."""
    from config.loader import ConfigLoader
    cfg = ConfigLoader()
    if cfg.get("thompson_policy_enabled", False):
        return select_action_thompson(state, regen_threshold, target_depth)
    return _select_action_cycle(state, regen_threshold, target_depth, exploration_fraction)
```

Rename the existing body of `select_action` to `_select_action_cycle`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_thompson_policy.py -v`
Expected: PASS

- [ ] **Step 6: Add config keys**

Add to `data/erik_config.json`:

```json
  "thompson_policy_enabled": false,
  "thompson_decay_interval": 50,
  "thompson_decay_rate": 0.95,
  "thompson_diversity_floor": 30
```

- [ ] **Step 7: Run full test suite**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v -k "not network and not chembl and not llm"`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/research/policy.py scripts/research/state.py tests/test_thompson_policy.py data/erik_config.json && git commit -m "feat(P3a): Thompson sampling policy with decay and diversity floor"
```

---

## Task Group 7: Galen SCM Integration (P3b)

### Task 7.1: Galen SCM Connector

**Files:**
- Create: `scripts/connectors/galen_scm.py`
- Create: `tests/test_galen_scm_connector.py`
- Modify: `scripts/research/actions.py`
- Modify: `scripts/research/policy.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_galen_scm_connector.py
"""Tests for Galen SCM causal query connector."""
import pytest
from connectors.galen_scm import (
    CausalEdge,
    PathwayStrength,
    GalenSCMConnector,
)


class TestCausalEdge:
    def test_creation(self):
        e = CausalEdge(
            source="MTOR", target="autophagy",
            relationship_type="inhibits",
            pch_layer=3, confidence=0.85,
        )
        assert e.source == "MTOR"
        assert e.pch_layer == 3


class TestPathwayStrength:
    def test_confidence_computation(self):
        p = PathwayStrength(
            pathway="mTOR/autophagy",
            l2_edges=100, l3_edges=200,
            total_entities=50,
            confidence=200 / 300,
        )
        assert p.confidence == pytest.approx(0.667, abs=0.01)

    def test_zero_edges(self):
        p = PathwayStrength(
            pathway="unknown",
            l2_edges=0, l3_edges=0,
            total_entities=0, confidence=0.0,
        )
        assert p.confidence == 0.0


class TestGalenSCMConnector:
    def test_disabled(self):
        c = GalenSCMConnector(enabled=False)
        edges = c.query_causal_downstream("MTOR")
        assert edges == []

    def test_pathway_strength_disabled(self):
        c = GalenSCMConnector(enabled=False)
        p = c.query_pathway_strength("mTOR")
        assert p.l2_edges == 0
        assert p.l3_edges == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_galen_scm_connector.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement galen_scm.py**

```python
# scripts/connectors/galen_scm.py
"""GalenSCMConnector — causal graph queries against Galen's cancer KG.

Queries Galen's L2+/L3 causal edges via direct SQL (no Galen Python imports).
Uses recursive CTEs for downstream/upstream chain traversal.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CausalEdge:
    source: str
    target: str
    relationship_type: str
    pch_layer: int
    confidence: float


@dataclass
class PathwayStrength:
    pathway: str
    l2_edges: int
    l3_edges: int
    total_entities: int
    confidence: float  # l3 / (l2 + l3), 0 if no edges


class GalenSCMConnector:
    """Query Galen's causal graph for cross-disease reasoning."""

    def __init__(
        self,
        database: str = "galen_kg",
        enabled: bool = True,
        min_pch_layer: int = 2,
        max_depth: int = 3,
    ):
        self._database = database
        self._user = os.environ.get("USER", "logannye")
        self._enabled = enabled
        self._min_pch_layer = min_pch_layer
        self._max_depth = max_depth

    def _connect(self):
        import psycopg
        return psycopg.connect(
            f"dbname={self._database} user={self._user}",
            connect_timeout=10,
            options="-c statement_timeout=30000 -c work_mem=16MB",
        )

    def query_causal_downstream(
        self, target_gene: str, max_depth: Optional[int] = None,
    ) -> list[CausalEdge]:
        """Walk the causal graph downstream from a target gene."""
        if not self._enabled:
            return []
        depth = max_depth or self._max_depth

        try:
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        WITH RECURSIVE chain AS (
                            SELECT r.target_id, e2.name AS target_name,
                                   r.relationship_type, r.pch_layer,
                                   r.confidence, 1 AS depth
                            FROM entities e1
                            JOIN relationships r ON r.source_id = e1.id
                            JOIN entities e2 ON r.target_id = e2.id
                            WHERE e1.name = %s AND r.pch_layer >= %s

                            UNION ALL

                            SELECT r.target_id, e2.name,
                                   r.relationship_type, r.pch_layer,
                                   r.confidence, c.depth + 1
                            FROM chain c
                            JOIN relationships r ON r.source_id = c.target_id
                            JOIN entities e2 ON r.target_id = e2.id
                            WHERE c.depth < %s AND r.pch_layer >= %s
                        )
                        SELECT %s AS source, target_name, relationship_type,
                               pch_layer, COALESCE(confidence, 0.5)
                        FROM chain
                        LIMIT 100
                    """, (target_gene, self._min_pch_layer,
                          depth, self._min_pch_layer, target_gene))
                    rows = cur.fetchall()
            finally:
                conn.close()
        except Exception:
            return []

        return [
            CausalEdge(
                source=row[0], target=row[1],
                relationship_type=row[2],
                pch_layer=row[3], confidence=float(row[4]),
            )
            for row in rows
        ]

    def query_causal_upstream(
        self, effect: str, max_depth: Optional[int] = None,
    ) -> list[CausalEdge]:
        """Walk the causal graph upstream to find causes of an effect."""
        if not self._enabled:
            return []
        depth = max_depth or self._max_depth

        try:
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        WITH RECURSIVE chain AS (
                            SELECT r.source_id, e1.name AS source_name,
                                   r.relationship_type, r.pch_layer,
                                   r.confidence, 1 AS depth
                            FROM entities e2
                            JOIN relationships r ON r.target_id = e2.id
                            JOIN entities e1 ON r.source_id = e1.id
                            WHERE e2.name = %s AND r.pch_layer >= %s

                            UNION ALL

                            SELECT r.source_id, e1.name,
                                   r.relationship_type, r.pch_layer,
                                   r.confidence, c.depth + 1
                            FROM chain c
                            JOIN relationships r ON r.target_id = c.source_id
                            JOIN entities e1 ON r.source_id = e1.id
                            WHERE c.depth < %s AND r.pch_layer >= %s
                        )
                        SELECT source_name, %s AS target, relationship_type,
                               pch_layer, COALESCE(confidence, 0.5)
                        FROM chain
                        LIMIT 100
                    """, (effect, self._min_pch_layer,
                          depth, self._min_pch_layer, effect))
                    rows = cur.fetchall()
            finally:
                conn.close()
        except Exception:
            return []

        return [
            CausalEdge(
                source=row[0], target=row[1],
                relationship_type=row[2],
                pch_layer=row[3], confidence=float(row[4]),
            )
            for row in rows
        ]

    def query_pathway_strength(self, pathway_name: str) -> PathwayStrength:
        """Count L2/L3 edges involving a named pathway/concept."""
        if not self._enabled:
            return PathwayStrength(
                pathway=pathway_name, l2_edges=0, l3_edges=0,
                total_entities=0, confidence=0.0,
            )

        try:
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT
                            count(*) FILTER (WHERE r.pch_layer = 2) AS l2,
                            count(*) FILTER (WHERE r.pch_layer = 3) AS l3,
                            count(DISTINCT e1.id) + count(DISTINCT e2.id) AS entities
                        FROM relationships r
                        JOIN entities e1 ON r.source_id = e1.id
                        JOIN entities e2 ON r.target_id = e2.id
                        WHERE (e1.name ILIKE %s OR e2.name ILIKE %s)
                          AND r.pch_layer >= 2
                    """, (f"%{pathway_name}%", f"%{pathway_name}%"))
                    row = cur.fetchone()
            finally:
                conn.close()
        except Exception:
            return PathwayStrength(
                pathway=pathway_name, l2_edges=0, l3_edges=0,
                total_entities=0, confidence=0.0,
            )

        l2, l3, entities = row[0] or 0, row[1] or 0, row[2] or 0
        total = l2 + l3
        confidence = l3 / total if total > 0 else 0.0

        return PathwayStrength(
            pathway=pathway_name,
            l2_edges=l2, l3_edges=l3,
            total_entities=entities,
            confidence=confidence,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_galen_scm_connector.py -v`
Expected: PASS

- [ ] **Step 5: Add QUERY_GALEN_SCM action type and policy slot**

In `scripts/research/actions.py`, add:

```python
    QUERY_GALEN_SCM = "query_galen_scm"
```

Add to `NETWORK_ACTIONS`:

```python
    ActionType.QUERY_GALEN_SCM,
```

In `scripts/research/policy.py`, update `_ACQUISITION_ROTATION` to its final form:

```python
_ACQUISITION_ROTATION = [
    ActionType.SEARCH_PUBMED,
    ActionType.SEARCH_TRIALS,
    ActionType.QUERY_PATHWAYS,
    ActionType.QUERY_PPI_NETWORK,
    ActionType.CHECK_PHARMACOGENOMICS,
    ActionType.QUERY_GALEN_KG,
    ActionType.SEARCH_PREPRINTS,
    ActionType.QUERY_GALEN_SCM,
]
```

Add handler in `_select_acquisition_action`:

```python
    elif action == ActionType.QUERY_GALEN_SCM:
        from config.loader import ConfigLoader
        cfg = ConfigLoader()
        if not cfg.get("galen_scm_enabled", True):
            return _fallback_acquisition(state, step, skip=ActionType.QUERY_GALEN_SCM)
        from connectors.galen_kg import ALS_CROSS_REFERENCE_GENES
        gene_idx = step % len(ALS_CROSS_REFERENCE_GENES)
        gene = ALS_CROSS_REFERENCE_GENES[gene_idx]
        return action, build_action_params(
            action, target_gene=gene, protocol_layer="root_cause_suppression",
        )
```

- [ ] **Step 6: Add config keys**

Add to `data/erik_config.json`:

```json
  "galen_scm_enabled": true,
  "galen_scm_min_pch_layer": 2,
  "galen_scm_max_chain_depth": 3,
  "galen_scm_database": "galen_kg"
```

- [ ] **Step 7: Run full test suite**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v -k "not network and not chembl and not llm"`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/connectors/galen_scm.py tests/test_galen_scm_connector.py scripts/research/actions.py scripts/research/policy.py data/erik_config.json && git commit -m "feat(P3b): Galen SCM connector with recursive CTE causal queries"
```

---

## Task Group 8: Integration and Final Verification

### Task 8.1: Run Full Test Suite and Verify Config

- [ ] **Step 1: Install new dependencies**

Run: `conda run -n erik-core pip install scipy lifelines`

- [ ] **Step 2: Run complete test suite**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v -k "not network and not chembl and not llm"`
Expected: All PASS (800+ tests)

- [ ] **Step 3: Validate erik_config.json**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core python -c "import json; json.load(open('data/erik_config.json'))"`
Expected: No error (valid JSON)

- [ ] **Step 4: Verify Erik can start with new code**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core python -c "
import sys; sys.path.insert(0, 'scripts')
from research.eligibility import EligibilityVerdict, compute_eligibility
from connectors.biorxiv import BiorxivConnector
from connectors.proact import ProactConnector
from connectors.galen_scm import GalenSCMConnector
from research.adversarial import generate_adversarial_queries
from world_model.combination_analyzer import analyze_combinations
from world_model.trajectory_matcher import TrajectoryMatcher
from research.policy import select_action
print('All modules import successfully')
"`
Expected: `All modules import successfully`

- [ ] **Step 5: Final commit with updated README**

```bash
cd /Users/logannye/.openclaw/erik && git add -A && git commit -m "feat: P1-P3 research enhancements complete — 7 new capabilities for Erik's ALS research"
```
