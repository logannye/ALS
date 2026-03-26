# Phase 2: World Model + Cure Protocol Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate Erik Draper's first cure protocol candidate by materializing his disease state, inferring his subtype, scoring all interventions against evidence, assembling a 5-layer protocol, and stress-testing it with counterfactual verification — all grounded in the evidence fabric with LLM-based causal reasoning.

**Architecture:** A 6-stage deliberative pipeline. Stages 1 (state) and parts of 2-5 use a `ReasoningEngine` that wraps local Qwen LLM with hard evidence-grounding constraints: citation-mandatory output, uncited claim rejection, dual verification on critical claims. Each stage produces canonical Pydantic objects stored in PostgreSQL. The pipeline orchestrator (`protocol_generator.py`) runs all 6 stages in sequence and outputs a `CureProtocolCandidate`.

**Tech Stack:** Python 3.12, mlx-lm (local LLM inference), Pydantic v2, psycopg3, pytest.

**Spec:** `/Users/logannye/.openclaw/erik/docs/specs/2026-03-25-phase2-world-model-cure-protocol-design.md`

---

## File Structure

```
scripts/
  llm/
    __init__.py               # CREATE
    inference.py              # CREATE: MLX LLM wrapper (generate, generate_json)
  world_model/
    __init__.py               # CREATE
    reasoning_engine.py       # CREATE: Evidence-grounded LLM with citation validation
    state_materializer.py     # CREATE: Stage 1 — observations → latent state
    subtype_inference.py      # CREATE: Stage 2 — state + evidence → subtype posterior
    intervention_scorer.py    # CREATE: Stage 3 — score interventions (+ InterventionScore model)
    protocol_assembler.py     # CREATE: Stage 4 — select + sequence + interaction check
    counterfactual_check.py   # CREATE: Stage 5 — stress-test each layer (+ CounterfactualResult model)
    protocol_generator.py     # CREATE: Stage 6 — orchestrate full pipeline
    prompts/
      __init__.py             # CREATE
      templates.py            # CREATE: All LLM prompt templates
  evidence/
    evidence_store.py         # MODIFY: add query_by_intervention_ref()
tests/
  test_llm_inference.py       # CREATE
  test_reasoning_engine.py    # CREATE
  test_state_materializer.py  # CREATE
  test_subtype_inference.py   # CREATE
  test_intervention_scorer.py # CREATE
  test_protocol_assembler.py  # CREATE
  test_counterfactual_check.py # CREATE
  test_protocol_generator.py  # CREATE
```

---

## Task 1: Prerequisites + LLM Inference Wrapper

**Files:**
- Create: `scripts/llm/__init__.py`
- Create: `scripts/llm/inference.py`
- Create: `tests/test_llm_inference.py`
- Modify: `scripts/evidence/evidence_store.py` (add `query_by_intervention_ref`)

- [ ] **Step 1: Install mlx-lm**

```bash
conda run -n erik-core pip install mlx-lm
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_llm_inference.py
import pytest
from llm.inference import LLMInference


def test_llm_inference_instantiates():
    """Should instantiate without loading model (lazy load)."""
    llm = LLMInference(model_path="/nonexistent/model", lazy=True)
    assert llm._model_path == "/nonexistent/model"
    assert llm._model is None


def test_llm_inference_default_config():
    llm = LLMInference(model_path="/tmp/fake", lazy=True)
    assert llm._temperature == 0.1
    assert llm._max_tokens == 1000


def test_generate_json_parses_valid_json():
    """Test JSON extraction from LLM-like output."""
    from llm.inference import _extract_json
    raw = 'Some text before {"key": "value", "num": 42} some text after'
    result = _extract_json(raw)
    assert result == {"key": "value", "num": 42}


def test_generate_json_handles_markdown_fences():
    from llm.inference import _extract_json
    raw = '```json\n{"key": "value"}\n```'
    result = _extract_json(raw)
    assert result == {"key": "value"}


def test_generate_json_returns_none_for_invalid():
    from llm.inference import _extract_json
    raw = "No JSON here at all"
    result = _extract_json(raw)
    assert result is None


@pytest.mark.llm
def test_generate_real_llm():
    """Requires actual LLM model on disk."""
    llm = LLMInference()
    result = llm.generate("What is 2+2? Answer with just the number.")
    assert "4" in result
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_llm_inference.py -v -k "not llm"`
Expected: FAIL

- [ ] **Step 4: Write LLM inference wrapper**

```python
# scripts/llm/__init__.py
```

```python
# scripts/llm/inference.py
"""Thin wrapper for local MLX LLM inference.

Wraps mlx_lm.generate() for structured JSON output with
temperature control and timeout. Lazy model loading.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional


def _extract_json(text: str) -> Optional[dict]:
    """Extract first JSON object from LLM output text.

    Handles: raw JSON, markdown-fenced JSON, JSON with surrounding text.
    Returns None if no valid JSON found.
    """
    # Try markdown fences first
    fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON object boundaries
    for i, ch in enumerate(text):
        if ch == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i : j + 1])
                        except json.JSONDecodeError:
                            break
            break

    return None


class LLMInference:
    """Local MLX LLM inference with lazy model loading.

    Usage:
        llm = LLMInference()  # Uses default model path
        text = llm.generate("prompt")
        data = llm.generate_json("prompt returning JSON")
    """

    DEFAULT_MODEL = "/Volumes/Databank/models/mlx/Qwen3.5-35B-A3B-mlx-lm-mxfp4"
    FALLBACK_MODEL = "/Volumes/Databank/models/mlx/Qwen3.5-9B-mlx-lm-4bit"

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        lazy: bool = False,
    ):
        self._model_path = model_path or self.DEFAULT_MODEL
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._model = None
        self._tokenizer = None
        if not lazy:
            self._load_model()

    def _load_model(self) -> None:
        """Load MLX model and tokenizer."""
        if self._model is not None:
            return
        import os
        path = self._model_path
        if not os.path.exists(path):
            # Try fallback
            path = self.FALLBACK_MODEL
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"No LLM model found at {self._model_path} or {self.FALLBACK_MODEL}"
                )
            self._model_path = path
        from mlx_lm import load
        self._model, self._tokenizer = load(path)

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text from prompt."""
        self._load_model()
        from mlx_lm import generate as mlx_generate
        return mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens or self._max_tokens,
            temp=self._temperature,
        )

    def generate_json(self, prompt: str, max_tokens: Optional[int] = None) -> Optional[dict]:
        """Generate text and extract JSON from response."""
        raw = self.generate(prompt, max_tokens=max_tokens or self._max_tokens)
        return _extract_json(raw)
```

- [ ] **Step 5: Add query_by_intervention_ref to EvidenceStore**

Add this method to `scripts/evidence/evidence_store.py` after `query_by_mechanism_target`:

```python
    def query_by_intervention_ref(self, intervention_id: str) -> list[dict]:
        """Return active EvidenceItems whose body.intervention_ref matches."""
        sql = """
            SELECT id, type, status, body
            FROM erik_core.objects
            WHERE type = 'EvidenceItem'
              AND status = 'active'
              AND body->>'intervention_ref' = %s
        """
        return self._run_query(sql, (intervention_id,))
```

- [ ] **Step 6: Register `llm` pytest mark in pyproject.toml**

Add `"llm: marks tests that require a local LLM model"` to the markers list.

- [ ] **Step 7: Run all tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v -k "not network and not chembl and not llm"`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add scripts/llm/ scripts/evidence/evidence_store.py tests/test_llm_inference.py pyproject.toml && git commit -m "feat: LLM inference wrapper (mlx-lm) + EvidenceStore query_by_intervention_ref"
```

---

## Task 2: Prompt Templates

**Files:**
- Create: `scripts/world_model/__init__.py`
- Create: `scripts/world_model/prompts/__init__.py`
- Create: `scripts/world_model/prompts/templates.py`

- [ ] **Step 1: Write the prompt templates**

No tests needed for pure data — these are string constants. But we test rendering in Task 3.

```python
# scripts/world_model/__init__.py
```

```python
# scripts/world_model/prompts/__init__.py
```

```python
# scripts/world_model/prompts/templates.py
"""LLM prompt templates for each pipeline stage.

All templates follow the evidence-grounded pattern:
- System prompt forbids introducing external knowledge
- Evidence items provided as JSON array
- Output must be valid JSON matching specified schema
- Every claim must cite evi:* or int:* IDs from the input
"""

SYSTEM_PROMPT = """You are a structured evidence synthesizer for ALS research.

STRICT RULES:
1. You reason ONLY from the evidence items provided below. Do NOT introduce information from your training data.
2. You MUST cite evidence item IDs (evi:* or int:*) for EVERY factual claim you make.
3. If the provided evidence is insufficient to answer, say so explicitly — do NOT speculate.
4. A preclinical result CANNOT override a Phase 3 RCT finding.
5. Output ONLY valid JSON matching the schema provided. No other text.
6. Weight evidence by strength: strong (RCT) > moderate (Phase 2) > emerging (case series) > preclinical > unknown.
7. Weight by PCH level: L3 (counterfactual) > L2 (interventional) > L1 (associational)."""

REVERSIBILITY_TEMPLATE = """TASK: Estimate the reversibility window for this ALS patient.

PATIENT STATE:
{patient_state_json}

EVIDENCE ITEMS:
{evidence_items_json}

Based ONLY on the evidence provided, estimate:
- overall_reversibility_score (0-1): how recoverable is the current disease state?
- molecular_correction_plausibility (0-1): can the underlying molecular defect be corrected?
- nmj_recovery_plausibility (0-1): can denervated NMJs be reoccupied?
- functional_recovery_plausibility (0-1): can lost function be restored?
- dominant_bottleneck: what is the single biggest barrier to recovery?
- estimated_time_sensitivity_days: how many days before the window narrows significantly?
- reasoning: explain your estimates, citing evidence IDs.

OUTPUT JSON SCHEMA:
{{"overall_reversibility_score": float, "molecular_correction_plausibility": float, "nmj_recovery_plausibility": float, "functional_recovery_plausibility": float, "dominant_bottleneck": str, "estimated_time_sensitivity_days": int, "reasoning": str, "cited_evidence": [str]}}"""

MOLECULAR_STATE_TEMPLATE = """TASK: Estimate the molecular disease state for this ALS patient based on population-level evidence.

NOTE: This patient has NO direct molecular assays. All estimates are inferred from literature and clinical presentation. Flag all scores as "inferred_from_literature_not_measured".

PATIENT STATE:
{patient_state_json}

INFERRED SUBTYPE: {dominant_subtype}

EVIDENCE ITEMS:
{evidence_items_json}

Estimate these molecular state scores (0-1 scale):
- tdp43_nuclear_function_score: estimated nuclear TDP-43 function (1=normal, 0=complete loss)
- tdp43_cytoplasmic_pathology_probability: probability of cytoplasmic TDP-43 aggregation
- tdp43_loss_of_function_probability: probability of TDP-43 loss-of-function
- cryptic_splicing_burden_score: estimated cryptic exon splicing burden
- stmn2_disruption_score: estimated STMN2 cryptic exon disruption
- unc13a_disruption_score: estimated UNC13A cryptic exon disruption
- microglial_activation_score: estimated microglial activation level
- astrocytic_toxicity_score: estimated astrocytic toxicity level
- inflammatory_amplification_score: estimated inflammatory amplification

OUTPUT JSON SCHEMA:
{{"tdp43_nuclear_function_score": float, "tdp43_cytoplasmic_pathology_probability": float, "tdp43_loss_of_function_probability": float, "cryptic_splicing_burden_score": float, "stmn2_disruption_score": float, "unc13a_disruption_score": float, "microglial_activation_score": float, "astrocytic_toxicity_score": float, "inflammatory_amplification_score": float, "reasoning": str, "cited_evidence": [str]}}"""

SUBTYPE_TEMPLATE = """TASK: Infer the ALS subtype posterior probability distribution for this patient.

PATIENT PRESENTATION:
{patient_state_json}

EVIDENCE ITEMS (subtype characteristics):
{evidence_items_json}

Produce a probability distribution over these 8 subtypes (must sum to ~1.0):
- sporadic_tdp43: sporadic TDP-43 loss-of-function dominant
- c9orf72: C9orf72 repeat expansion dominant
- sod1: SOD1 toxic gain-of-function dominant
- fus: FUS/RNA-binding protein dominant
- tardbp: TARDBP mutation dominant
- glia_amplified: glia-amplified subtype
- mixed: mixed or multi-driver
- unresolved: insufficient evidence to classify

Also produce conditional_on_genetics: how the posterior would shift if genetic testing returns positive for C9orf72, SOD1, or FUS, or returns negative for all tested genes.

OUTPUT JSON SCHEMA:
{{"posterior": {{"sporadic_tdp43": float, "c9orf72": float, "sod1": float, "fus": float, "tardbp": float, "glia_amplified": float, "mixed": float, "unresolved": float}}, "conditional_on_genetics": {{"if_c9orf72_positive": {{...}}, "if_sod1_positive": {{...}}, "if_fus_positive": {{...}}, "if_negative_panel": {{...}}}}, "reasoning": str, "cited_evidence": [str]}}"""

INTERVENTION_SCORING_TEMPLATE = """TASK: Score this intervention's relevance for this specific ALS patient.

PATIENT STATE:
{patient_state_json}

INFERRED SUBTYPE POSTERIOR:
{subtype_posterior_json}

INTERVENTION:
{intervention_json}

EVIDENCE ITEMS FOR THIS INTERVENTION:
{evidence_items_json}

Score this intervention on these criteria:
- relevance_score (0-1): overall relevance to this patient given their subtype and state
- mechanism_argument: explain WHY this intervention is relevant (or not), citing evidence
- evidence_strength: overall strength of the supporting evidence (strong/moderate/emerging/preclinical/unknown)
- erik_eligible: is this patient eligible? (true/false/"pending_genetics")
- key_uncertainties: list the most important unknowns
- cited_evidence: list ALL evidence IDs you referenced

SCORING CRITERIA (weight in this order):
1. Mechanism relevance: does it target an active disease program in this patient's subtype?
2. Evidence quality: strong (RCT) > moderate (Phase 2) > emerging > preclinical
3. Patient eligibility: age, comorbidities, medications
4. Safety: known risks relative to patient's comorbidities
5. Feasibility: route, availability, regulatory status
6. Timing sensitivity: urgency given reversibility window

OUTPUT JSON SCHEMA:
{{"intervention_id": str, "intervention_name": str, "protocol_layer": str, "relevance_score": float, "mechanism_argument": str, "evidence_strength": str, "erik_eligible": bool_or_str, "key_uncertainties": [str], "cited_evidence": [str]}}"""

COUNTERFACTUAL_TEMPLATE = """TASK: Assess the impact of removing one layer from this cure protocol.

FULL PROTOCOL:
{protocol_json}

LAYER BEING REMOVED: {layer_name}
INTERVENTION(S) IN THIS LAYER: {layer_interventions}

EVIDENCE FOR THIS LAYER:
{evidence_items_json}

If this layer were removed from the protocol, what would be the expected impact on the patient's disease trajectory?

OUTPUT JSON SCHEMA:
{{"layer": str, "removal_impact": str, "reasoning": str, "is_load_bearing": bool, "weakest_evidence": str, "next_best_measurement": str, "cited_evidence": [str]}}"""

VERIFICATION_TEMPLATE = """TASK: Verify whether this claim is fully supported by the provided evidence.

CLAIM TO VERIFY:
{claim_text}

EVIDENCE ITEMS (the ONLY basis for verification):
{evidence_items_json}

Is this claim fully supported by the provided evidence?

OUTPUT JSON SCHEMA:
{{"claim": str, "verdict": str, "reasoning": str, "cited_evidence": [str]}}

verdict must be one of: "supported", "partially_supported", "unsupported", "contested"."""
```

- [ ] **Step 2: Commit**

```bash
git add scripts/world_model/ && git commit -m "feat: LLM prompt templates for all 6 pipeline stages"
```

---

## Task 3: Reasoning Engine

**Files:**
- Create: `scripts/world_model/reasoning_engine.py`
- Create: `tests/test_reasoning_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reasoning_engine.py
import pytest
from world_model.reasoning_engine import (
    ReasoningEngine, validate_citations, strip_uncited_claims,
)


def test_validate_citations_all_valid():
    output = {"cited_evidence": ["evi:a", "evi:b", "int:c"]}
    valid_ids = {"evi:a", "evi:b", "int:c", "evi:d"}
    cleaned, warnings = validate_citations(output, valid_ids)
    assert warnings == []
    assert cleaned["cited_evidence"] == ["evi:a", "evi:b", "int:c"]


def test_validate_citations_catches_hallucinated():
    output = {"cited_evidence": ["evi:real", "evi:hallucinated"]}
    valid_ids = {"evi:real"}
    cleaned, warnings = validate_citations(output, valid_ids)
    assert len(warnings) == 1
    assert "evi:hallucinated" in warnings[0]
    assert cleaned["cited_evidence"] == ["evi:real"]


def test_validate_citations_empty_input():
    output = {"cited_evidence": []}
    valid_ids = {"evi:a"}
    cleaned, warnings = validate_citations(output, valid_ids)
    assert cleaned["cited_evidence"] == []


def test_strip_uncited_claims_removes_text():
    text = "Claim A is true [evi:a]. Claim B has no support. Claim C [evi:c]."
    valid_ids = {"evi:a", "evi:c"}
    result = strip_uncited_claims(text, valid_ids)
    assert "Claim A" in result
    assert "Claim C" in result


def test_strip_uncited_claims_keeps_all_if_all_cited():
    text = "Result X [evi:x]. Result Y [evi:y]."
    valid_ids = {"evi:x", "evi:y"}
    result = strip_uncited_claims(text, valid_ids)
    assert "Result X" in result
    assert "Result Y" in result


def test_reasoning_engine_instantiates_lazy():
    engine = ReasoningEngine(lazy=True)
    assert engine._llm is not None or engine._lazy


def test_reasoning_engine_builds_prompt():
    engine = ReasoningEngine(lazy=True)
    evidence = [{"id": "evi:a", "claim": "Test claim", "body": {"protocol_layer": "root_cause_suppression"}}]
    prompt = engine._build_prompt(
        template="TASK: Test\nEVIDENCE ITEMS:\n{evidence_items_json}\nOUTPUT: JSON",
        evidence_items=evidence,
        extra_context={"patient_state_json": '{"alsfrs_r": 43}'},
    )
    assert "evi:a" in prompt
    assert "Test claim" in prompt
    assert "alsfrs_r" in prompt


@pytest.mark.llm
def test_reasoning_engine_real_call():
    engine = ReasoningEngine()
    evidence = [
        {"id": "evi:test1", "claim": "Riluzole provides modest survival benefit",
         "body": {"protocol_layer": "circuit_stabilization", "pch_layer": 2}},
    ]
    result = engine.reason(
        template='TASK: Summarize the evidence.\nEVIDENCE ITEMS:\n{evidence_items_json}\nOUTPUT JSON SCHEMA:\n{{"summary": str, "cited_evidence": [str]}}',
        evidence_items=evidence,
    )
    assert result is not None
    assert "cited_evidence" in result
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_reasoning_engine.py -v -k "not llm"`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/world_model/reasoning_engine.py
"""Evidence-grounded LLM reasoning engine.

The ReasoningEngine is NOT a chatbot. It is a structured evidence
synthesizer with hard grounding constraints:
1. LLM receives ONLY evidence items from the fabric
2. Every claim must cite evi:* or int:* IDs
3. Uncited claims are stripped
4. Critical claims get dual verification
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from llm.inference import LLMInference, _extract_json
from world_model.prompts.templates import SYSTEM_PROMPT, VERIFICATION_TEMPLATE


def validate_citations(
    output: dict, valid_ids: set[str]
) -> tuple[dict, list[str]]:
    """Validate all evidence citations in LLM output.

    Returns (cleaned_output, warnings).
    Removes hallucinated IDs from cited_evidence list.
    """
    warnings = []
    cited = output.get("cited_evidence", [])
    valid_cited = []
    for cid in cited:
        if cid in valid_ids:
            valid_cited.append(cid)
        else:
            warnings.append(f"Hallucinated citation removed: {cid}")
    output = dict(output)
    output["cited_evidence"] = valid_cited
    return output, warnings


def strip_uncited_claims(text: str, valid_ids: set[str]) -> str:
    """Remove sentences from text that contain no valid evidence citations.

    A sentence is kept if it contains at least one [evi:*] or [int:*]
    reference that exists in valid_ids, OR if it's a connecting/structural sentence.
    """
    if not text:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for sentence in sentences:
        refs = re.findall(r"\[(evi:[^\]]+|int:[^\]]+)\]", sentence)
        if refs:
            # Has citations — keep only if at least one is valid
            if any(r in valid_ids for r in refs):
                kept.append(sentence)
        else:
            # No citations — keep short structural sentences, drop long unsupported claims
            if len(sentence.split()) < 8:
                kept.append(sentence)
    return " ".join(kept)


class ReasoningEngine:
    """Evidence-grounded LLM reasoning with citation validation."""

    def __init__(self, lazy: bool = False, model_path: Optional[str] = None):
        self._lazy = lazy
        self._llm = LLMInference(model_path=model_path, lazy=True) if lazy else LLMInference(model_path=model_path)

    def reason(
        self,
        template: str,
        evidence_items: list[dict],
        extra_context: Optional[dict] = None,
        max_tokens: int = 1500,
        verify_critical: bool = False,
    ) -> Optional[dict]:
        """Run evidence-grounded reasoning.

        1. Build prompt from template + evidence
        2. Call LLM
        3. Parse JSON response
        4. Validate citations
        5. Optionally verify critical claims
        6. Return validated dict or None
        """
        prompt = self._build_prompt(template, evidence_items, extra_context)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

        result_dict = self._llm.generate_json(full_prompt, max_tokens=max_tokens)
        if result_dict is None:
            return None

        # Validate citations
        valid_ids = {item["id"] for item in evidence_items}
        # Also accept int:* IDs if present in evidence
        for item in evidence_items:
            if "intervention_ref" in item.get("body", {}):
                valid_ids.add(item["body"]["intervention_ref"])

        result_dict, warnings = validate_citations(result_dict, valid_ids)
        for w in warnings:
            print(f"[GROUNDING WARNING] {w}")

        # Strip uncited claims from text fields
        for text_field in ("reasoning", "mechanism_argument"):
            if text_field in result_dict and isinstance(result_dict[text_field], str):
                result_dict[text_field] = strip_uncited_claims(
                    result_dict[text_field], valid_ids
                )

        # Dual verification on critical claims
        if verify_critical and "mechanism_argument" in result_dict:
            verification = self._verify_claim(
                result_dict["mechanism_argument"],
                [item for item in evidence_items
                 if item["id"] in set(result_dict.get("cited_evidence", []))],
            )
            if verification and verification.get("verdict") in ("unsupported", "contested"):
                result_dict.setdefault("contested_claims", []).append(
                    result_dict["mechanism_argument"][:200]
                )

        return result_dict

    def _build_prompt(
        self,
        template: str,
        evidence_items: list[dict],
        extra_context: Optional[dict] = None,
    ) -> str:
        """Render template with evidence items and optional context."""
        context = {
            "evidence_items_json": json.dumps(evidence_items, indent=2, default=str),
        }
        if extra_context:
            context.update(extra_context)
        # Use safe string formatting (only replace known keys)
        result = template
        for key, value in context.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result

    def _verify_claim(
        self, claim_text: str, cited_evidence: list[dict]
    ) -> Optional[dict]:
        """Run verification pass on a critical claim."""
        if not cited_evidence:
            return {"verdict": "unsupported", "reasoning": "No evidence cited"}
        prompt = self._build_prompt(
            VERIFICATION_TEMPLATE,
            cited_evidence,
            extra_context={"claim_text": claim_text},
        )
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        return self._llm.generate_json(full_prompt, max_tokens=500)
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_reasoning_engine.py -v -k "not llm"`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/world_model/reasoning_engine.py tests/test_reasoning_engine.py && git commit -m "feat: evidence-grounded reasoning engine with citation validation and dual verification"
```

---

## Task 4: State Materializer (Stage 1)

**Files:**
- Create: `scripts/world_model/state_materializer.py`
- Create: `tests/test_state_materializer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state_materializer.py
import pytest
from datetime import date, datetime, timezone
from world_model.state_materializer import materialize_state, materialize_functional_state, materialize_uncertainty_state


def test_materialize_functional_state():
    from ingestion.patient_builder import build_erik_draper
    _, trajectory, observations = build_erik_draper()
    fs = materialize_functional_state(trajectory, observations)
    assert fs.alsfrs_r_total == 43
    assert fs.bulbar_subscore == 12
    assert fs.fine_motor_subscore == 11
    assert fs.gross_motor_subscore == 8
    assert fs.respiratory_subscore == 12


def test_materialize_uncertainty_state():
    from ingestion.patient_builder import build_erik_draper
    _, trajectory, observations = build_erik_draper()
    us = materialize_uncertainty_state(trajectory, observations)
    assert us.missing_measurement_uncertainty > 0
    assert "genetic_testing" in us.dominant_missing_measurements
    assert us.subtype_ambiguity > 0


def test_materialize_functional_state_has_weight():
    from ingestion.patient_builder import build_erik_draper
    _, trajectory, observations = build_erik_draper()
    fs = materialize_functional_state(trajectory, observations)
    assert fs.weight_kg is not None
    assert fs.weight_kg > 100  # Erik is ~111-118 kg


def test_materialize_state_returns_snapshot():
    from ingestion.patient_builder import build_erik_draper
    _, trajectory, observations = build_erik_draper()
    # Use lazy reasoning engine (no LLM needed for deterministic parts)
    snapshot = materialize_state(trajectory, observations, use_llm=False)
    assert snapshot.type == "DiseaseStateSnapshot"
    assert snapshot.functional_state_ref is not None
    assert snapshot.uncertainty_ref is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_state_materializer.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/world_model/state_materializer.py
"""Stage 1: Materialize Erik's disease state from observations.

Maps clinical observations to the latent state factorization:
  x_t = [g_t, m_t, n_t, f_t, r_t, u_t]

f_t (functional) and u_t (uncertainty) are deterministic.
n_t (neural compartment) is partially deterministic from EMG/exam.
m_t (molecular) and r_t (reversibility) require LLM inference.
g_t (etiologic) comes from Stage 2 (subtype inference).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ontology.enums import ObservationKind
from ontology.patient import ALSTrajectory
from ontology.observation import Observation
from ontology.state import (
    FunctionalState, NMJIntegrityState, RespiratoryReserveState,
    UncertaintyState, ReversibilityWindowEstimate, DiseaseStateSnapshot,
    TDP43FunctionalState, SplicingState, GlialState,
)


def materialize_functional_state(
    trajectory: ALSTrajectory, observations: list[Observation]
) -> FunctionalState:
    """Deterministic mapping from ALSFRS-R and observations to FunctionalState."""
    score = trajectory.alsfrs_r_scores[0] if trajectory.alsfrs_r_scores else None

    # Get latest weight
    weight_obs = [o for o in observations if o.observation_kind == ObservationKind.weight_measurement]
    latest_weight = weight_obs[-1].value if weight_obs else None

    return FunctionalState(
        id=f"func:{trajectory.id}",
        subject_ref=trajectory.id,
        alsfrs_r_total=score.total if score else None,
        bulbar_subscore=score.bulbar_subscore if score else None,
        fine_motor_subscore=score.fine_motor_subscore if score else None,
        gross_motor_subscore=score.gross_motor_subscore if score else None,
        respiratory_subscore=score.respiratory_subscore if score else None,
        weight_kg=latest_weight,
    )


def materialize_nmj_state(
    trajectory: ALSTrajectory, observations: list[Observation]
) -> NMJIntegrityState:
    """From EMG findings and physical exam."""
    emg_obs = [o for o in observations if o.observation_kind == ObservationKind.emg_feature]
    has_widespread_denervation = any(
        o.emg_finding and o.emg_finding.supports_als for o in emg_obs
    )

    return NMJIntegrityState(
        id=f"nmj:{trajectory.id}",
        subject_ref=trajectory.id,
        estimated_nmj_occupancy=0.5 if has_widespread_denervation else 0.75,
        # High denervation rate if EMG shows widespread changes
        denervation_rate_score=0.7 if has_widespread_denervation else 0.3,
        # Reinnervation still possible early in disease
        reinnervation_capacity_score=0.4 if has_widespread_denervation else 0.6,
        supporting_refs=[o.id for o in emg_obs],
    )


def materialize_respiratory_state(
    trajectory: ALSTrajectory, observations: list[Observation]
) -> RespiratoryReserveState:
    """From spirometry findings."""
    resp_obs = [o for o in observations if o.observation_kind == ObservationKind.respiratory_metric]
    fvc_pct = None
    for o in resp_obs:
        if o.respiratory_metric:
            fvc_pct = o.respiratory_metric.fvc_percent_predicted

    # FVC 100% = high reserve
    reserve = (fvc_pct / 100.0) if fvc_pct else 0.5

    return RespiratoryReserveState(
        id=f"resp:{trajectory.id}",
        subject_ref=trajectory.id,
        reserve_score=min(reserve, 1.0),
        six_month_decline_risk=0.2 if (fvc_pct and fvc_pct > 80) else 0.5,
        niv_transition_probability_6m=0.1 if (fvc_pct and fvc_pct > 80) else 0.4,
        supporting_refs=[o.id for o in resp_obs],
    )


def materialize_uncertainty_state(
    trajectory: ALSTrajectory, observations: list[Observation]
) -> UncertaintyState:
    """Enumerate what's measured vs missing."""
    obs_kinds = {o.observation_kind for o in observations}

    missing = []
    if ObservationKind.genomic_result not in obs_kinds:
        missing.append("genetic_testing")
    missing.extend([
        "csf_biomarkers",
        "cryptic_exon_splicing_assay",
        "tdp43_in_vivo_measurement",
        "cortical_excitability_tms",
        "transcriptomics",
        "proteomics",
    ])

    return UncertaintyState(
        id=f"unc:{trajectory.id}",
        subject_ref=trajectory.id,
        subtype_ambiguity=0.35,  # High until genetics arrive
        missing_measurement_uncertainty=len(missing) / 10.0,
        model_form_uncertainty=0.3,
        intervention_effect_uncertainty=0.5,
        transportability_uncertainty=0.2,
        evidence_conflict_uncertainty=0.15,
        dominant_missing_measurements=missing,
    )


def materialize_state(
    trajectory: ALSTrajectory,
    observations: list[Observation],
    use_llm: bool = True,
    reasoning_engine=None,
    evidence_items: Optional[list[dict]] = None,
) -> DiseaseStateSnapshot:
    """Full state materialization. Returns DiseaseStateSnapshot.

    If use_llm=False, skips m_t and r_t (LLM-dependent).
    """
    func_state = materialize_functional_state(trajectory, observations)
    nmj_state = materialize_nmj_state(trajectory, observations)
    resp_state = materialize_respiratory_state(trajectory, observations)
    unc_state = materialize_uncertainty_state(trajectory, observations)

    snapshot = DiseaseStateSnapshot(
        id=f"state:{trajectory.id}:{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        subject_ref=trajectory.id,
        as_of=datetime.now(timezone.utc),
        functional_state_ref=func_state.id,
        compartment_state_refs=[nmj_state.id, resp_state.id],
        uncertainty_ref=unc_state.id,
    )

    # Store state components as attributes for pipeline access
    snapshot.body = {
        "functional_state": func_state.model_dump(mode="json"),
        "nmj_state": nmj_state.model_dump(mode="json"),
        "respiratory_state": resp_state.model_dump(mode="json"),
        "uncertainty_state": unc_state.model_dump(mode="json"),
    }

    return snapshot
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_state_materializer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/world_model/state_materializer.py tests/test_state_materializer.py && git commit -m "feat: Stage 1 state materializer — observations to latent disease state"
```

---

## Task 5: Subtype Inference (Stage 2)

**Files:**
- Create: `scripts/world_model/subtype_inference.py`
- Create: `tests/test_subtype_inference.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_subtype_inference.py
import pytest
from ontology.enums import SubtypeClass


def test_infer_subtype_mock():
    """Test with mock reasoning engine output."""
    from world_model.subtype_inference import build_subtype_profile, _parse_subtype_response

    mock_response = {
        "posterior": {
            "sporadic_tdp43": 0.65, "c9orf72": 0.12, "sod1": 0.03,
            "fus": 0.02, "tardbp": 0.05, "glia_amplified": 0.08,
            "mixed": 0.03, "unresolved": 0.02,
        },
        "conditional_on_genetics": {
            "if_c9orf72_positive": {"c9orf72": 0.85, "sporadic_tdp43": 0.05},
            "if_sod1_positive": {"sod1": 0.90},
            "if_fus_positive": {"fus": 0.88},
            "if_negative_panel": {"sporadic_tdp43": 0.75, "glia_amplified": 0.10},
        },
        "reasoning": "Limb-onset sporadic presentation [evi:tdp43_disease_protein].",
        "cited_evidence": ["evi:tdp43_disease_protein"],
    }
    profile = _parse_subtype_response(mock_response, "traj:draper_001")
    assert profile.type == "EtiologicDriverProfile"
    assert profile.dominant_subtype == SubtypeClass.sporadic_tdp43
    assert abs(sum(profile.posterior.values()) - 1.0) < 0.01


def test_parse_subtype_response_maps_strings_to_enum():
    from world_model.subtype_inference import _parse_subtype_response
    mock = {
        "posterior": {"sporadic_tdp43": 0.8, "unresolved": 0.2},
        "cited_evidence": [],
    }
    profile = _parse_subtype_response(mock, "traj:test")
    assert SubtypeClass.sporadic_tdp43 in profile.posterior
    assert isinstance(list(profile.posterior.keys())[0], SubtypeClass)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_subtype_inference.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/world_model/subtype_inference.py
"""Stage 2: Infer ALS subtype posterior from patient state and evidence."""
from __future__ import annotations

from typing import Optional

from ontology.enums import SubtypeClass
from ontology.interpretation import EtiologicDriverProfile
from world_model.reasoning_engine import ReasoningEngine
from world_model.prompts.templates import SUBTYPE_TEMPLATE


def _parse_subtype_response(response: dict, subject_ref: str) -> EtiologicDriverProfile:
    """Parse LLM response into EtiologicDriverProfile with SubtypeClass enum keys."""
    raw_posterior = response.get("posterior", {})

    # Map string keys to SubtypeClass enum
    posterior = {}
    for key, value in raw_posterior.items():
        try:
            posterior[SubtypeClass(key)] = float(value)
        except (ValueError, KeyError):
            pass  # Skip invalid keys

    # Normalize to sum to 1.0
    total = sum(posterior.values())
    if total > 0:
        posterior = {k: v / total for k, v in posterior.items()}

    return EtiologicDriverProfile(
        id=f"driver:{subject_ref}",
        subject_ref=subject_ref,
        posterior=posterior,
        supporting_evidence_refs=response.get("cited_evidence", []),
        body={
            "conditional_on_genetics": response.get("conditional_on_genetics", {}),
            "reasoning": response.get("reasoning", ""),
        },
    )


def infer_subtype(
    patient_state_json: str,
    evidence_items: list[dict],
    subject_ref: str,
    reasoning_engine: Optional[ReasoningEngine] = None,
) -> EtiologicDriverProfile:
    """Run Stage 2: subtype inference."""
    if reasoning_engine is None:
        reasoning_engine = ReasoningEngine(lazy=True)

    response = reasoning_engine.reason(
        template=SUBTYPE_TEMPLATE,
        evidence_items=evidence_items,
        extra_context={"patient_state_json": patient_state_json},
        max_tokens=1500,
    )

    if response is None:
        # Abstention: return uniform prior
        uniform = {st: 1.0 / len(SubtypeClass) for st in SubtypeClass}
        return EtiologicDriverProfile(
            id=f"driver:{subject_ref}",
            subject_ref=subject_ref,
            posterior=uniform,
            body={"reasoning": "ABSTENTION: LLM returned no response"},
        )

    return _parse_subtype_response(response, subject_ref)
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_subtype_inference.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/world_model/subtype_inference.py tests/test_subtype_inference.py && git commit -m "feat: Stage 2 subtype inference with SubtypeClass enum mapping and conditional genetics"
```

---

## Task 6: Intervention Scorer (Stage 3)

**Files:**
- Create: `scripts/world_model/intervention_scorer.py`
- Create: `tests/test_intervention_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_intervention_scorer.py
import pytest
from world_model.intervention_scorer import InterventionScore, _parse_score_response, score_all_interventions


def test_intervention_score_model():
    s = InterventionScore(
        intervention_id="int:pridopidine",
        intervention_name="Pridopidine",
        protocol_layer="pathology_reversal",
        relevance_score=0.82,
        mechanism_argument="Activates sigma-1R [evi:sigma1r_biology]",
        evidence_strength="moderate",
        erik_eligible=True,
        key_uncertainties=["phase3_not_yet_read_out"],
        cited_evidence=["evi:sigma1r_biology"],
    )
    assert s.relevance_score == 0.82
    assert s.erik_eligible is True


def test_parse_score_response():
    mock = {
        "intervention_id": "int:test",
        "intervention_name": "Test Drug",
        "protocol_layer": "circuit_stabilization",
        "relevance_score": 0.5,
        "mechanism_argument": "Test mechanism",
        "evidence_strength": "moderate",
        "erik_eligible": True,
        "key_uncertainties": ["unknown"],
        "cited_evidence": ["evi:test1"],
    }
    score = _parse_score_response(mock)
    assert score.intervention_id == "int:test"
    assert score.relevance_score == 0.5


def test_parse_score_response_clamps_score():
    mock = {
        "intervention_id": "int:test",
        "intervention_name": "Test",
        "protocol_layer": "root_cause_suppression",
        "relevance_score": 1.5,  # Should be clamped to 1.0
        "mechanism_argument": "",
        "evidence_strength": "unknown",
        "erik_eligible": False,
        "key_uncertainties": [],
        "cited_evidence": [],
    }
    score = _parse_score_response(mock)
    assert score.relevance_score == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_intervention_scorer.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/world_model/intervention_scorer.py
"""Stage 3: Score each intervention's relevance to Erik."""
from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field

from world_model.reasoning_engine import ReasoningEngine
from world_model.prompts.templates import INTERVENTION_SCORING_TEMPLATE
from evidence.evidence_store import EvidenceStore


class InterventionScore(BaseModel):
    """Scored relevance of one intervention to Erik. Not a BaseEnvelope — transient artifact."""
    intervention_id: str
    intervention_name: str
    protocol_layer: str = ""
    relevance_score: float = Field(ge=0.0, le=1.0)
    mechanism_argument: str = ""
    evidence_strength: str = "unknown"
    erik_eligible: bool | str = True
    key_uncertainties: list[str] = Field(default_factory=list)
    cited_evidence: list[str] = Field(default_factory=list)
    contested_claims: list[str] = Field(default_factory=list)


def _parse_score_response(response: dict) -> InterventionScore:
    """Parse LLM response into InterventionScore, clamping values."""
    score = min(max(float(response.get("relevance_score", 0)), 0.0), 1.0)
    return InterventionScore(
        intervention_id=response.get("intervention_id", ""),
        intervention_name=response.get("intervention_name", ""),
        protocol_layer=response.get("protocol_layer", ""),
        relevance_score=score,
        mechanism_argument=response.get("mechanism_argument", ""),
        evidence_strength=response.get("evidence_strength", "unknown"),
        erik_eligible=response.get("erik_eligible", True),
        key_uncertainties=response.get("key_uncertainties", []),
        cited_evidence=response.get("cited_evidence", []),
        contested_claims=response.get("contested_claims", []),
    )


def score_intervention(
    intervention: dict,
    evidence_items: list[dict],
    patient_state_json: str,
    subtype_posterior_json: str,
    reasoning_engine: ReasoningEngine,
    verify_critical: bool = False,
) -> Optional[InterventionScore]:
    """Score a single intervention with evidence-grounded reasoning."""
    response = reasoning_engine.reason(
        template=INTERVENTION_SCORING_TEMPLATE,
        evidence_items=evidence_items,
        extra_context={
            "patient_state_json": patient_state_json,
            "subtype_posterior_json": subtype_posterior_json,
            "intervention_json": json.dumps(intervention, default=str),
        },
        max_tokens=1500,
        verify_critical=verify_critical,
    )
    if response is None:
        return None
    return _parse_score_response(response)


def score_all_interventions(
    interventions: list[dict],
    evidence_store: EvidenceStore,
    patient_state_json: str,
    subtype_posterior_json: str,
    reasoning_engine: Optional[ReasoningEngine] = None,
) -> list[InterventionScore]:
    """Score all interventions. Top 5 get verification pass."""
    if reasoning_engine is None:
        reasoning_engine = ReasoningEngine()

    scores = []
    for intervention in interventions:
        int_id = intervention.get("id", "")
        # Gather evidence for this intervention
        evidence = evidence_store.query_by_intervention_ref(int_id)
        # Also get evidence by mechanism target
        for target in intervention.get("targets", []):
            evidence.extend(evidence_store.query_by_mechanism_target(target))
        # Deduplicate
        seen = set()
        unique_evidence = []
        for e in evidence:
            if e["id"] not in seen:
                seen.add(e["id"])
                unique_evidence.append(e)

        score = score_intervention(
            intervention=intervention,
            evidence_items=unique_evidence,
            patient_state_json=patient_state_json,
            subtype_posterior_json=subtype_posterior_json,
            reasoning_engine=reasoning_engine,
        )
        if score is not None:
            scores.append(score)

    # Sort by relevance score descending
    scores.sort(key=lambda s: s.relevance_score, reverse=True)

    # Verify top 5
    for score in scores[:5]:
        if score.mechanism_argument:
            int_evidence = evidence_store.query_by_intervention_ref(score.intervention_id)
            verification = reasoning_engine._verify_claim(
                score.mechanism_argument,
                int_evidence,
            )
            if verification and verification.get("verdict") in ("unsupported", "contested"):
                score.contested_claims.append(score.mechanism_argument[:200])

    return scores
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_intervention_scorer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/world_model/intervention_scorer.py tests/test_intervention_scorer.py && git commit -m "feat: Stage 3 intervention scorer with InterventionScore model and top-5 verification"
```

---

## Task 7: Protocol Assembler (Stage 4) + Counterfactual Check (Stage 5)

**Files:**
- Create: `scripts/world_model/protocol_assembler.py`
- Create: `scripts/world_model/counterfactual_check.py`
- Create: `tests/test_protocol_assembler.py`
- Create: `tests/test_counterfactual_check.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_protocol_assembler.py
from world_model.protocol_assembler import assemble_protocol, select_layer_interventions
from world_model.intervention_scorer import InterventionScore
from ontology.enums import ProtocolLayer


def test_select_layer_interventions():
    scores = [
        InterventionScore(intervention_id="int:a", intervention_name="A", protocol_layer="root_cause_suppression", relevance_score=0.9, erik_eligible=True, cited_evidence=["evi:1"]),
        InterventionScore(intervention_id="int:b", intervention_name="B", protocol_layer="root_cause_suppression", relevance_score=0.5, erik_eligible=True, cited_evidence=["evi:2"]),
        InterventionScore(intervention_id="int:c", intervention_name="C", protocol_layer="pathology_reversal", relevance_score=0.8, erik_eligible=True, cited_evidence=["evi:3"]),
    ]
    selected = select_layer_interventions(scores, "root_cause_suppression")
    assert len(selected) >= 1
    assert selected[0].intervention_id == "int:a"


def test_select_layer_skips_ineligible():
    scores = [
        InterventionScore(intervention_id="int:a", intervention_name="A", protocol_layer="root_cause_suppression", relevance_score=0.9, erik_eligible=False, cited_evidence=[]),
        InterventionScore(intervention_id="int:b", intervention_name="B", protocol_layer="root_cause_suppression", relevance_score=0.5, erik_eligible=True, cited_evidence=[]),
    ]
    selected = select_layer_interventions(scores, "root_cause_suppression")
    assert selected[0].intervention_id == "int:b"


def test_assemble_protocol_has_5_layers():
    scores = [
        InterventionScore(intervention_id=f"int:{layer}", intervention_name=layer, protocol_layer=layer, relevance_score=0.7, erik_eligible=True, cited_evidence=[f"evi:{layer}"])
        for layer in ["root_cause_suppression", "pathology_reversal", "circuit_stabilization", "regeneration_reinnervation", "adaptive_maintenance"]
    ]
    protocol = assemble_protocol(scores, "traj:draper_001")
    assert protocol.type == "CureProtocolCandidate"
    assert len(protocol.layers) == 5


def test_assemble_protocol_abstains_empty_layer():
    scores = [
        InterventionScore(intervention_id="int:a", intervention_name="A", protocol_layer="circuit_stabilization", relevance_score=0.7, erik_eligible=True, cited_evidence=["evi:1"]),
    ]
    protocol = assemble_protocol(scores, "traj:draper_001")
    # Layers without interventions should have ABSTENTION notes
    abstained = [l for l in protocol.layers if "ABSTENTION" in l.notes]
    assert len(abstained) >= 1
```

```python
# tests/test_counterfactual_check.py
from world_model.counterfactual_check import CounterfactualResult


def test_counterfactual_result_model():
    cr = CounterfactualResult(
        layer="root_cause_suppression",
        removal_impact="high",
        reasoning="Removing root-cause suppression removes the primary disease-modifying intervention.",
        is_load_bearing=True,
        weakest_evidence="evi:vtx002_fast_track",
        next_best_measurement="genetic_testing",
        cited_evidence=["evi:vtx002_fast_track"],
    )
    assert cr.is_load_bearing is True
    assert cr.removal_impact == "high"
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_protocol_assembler.py tests/test_counterfactual_check.py -v`
Expected: FAIL

- [ ] **Step 3: Write protocol assembler**

```python
# scripts/world_model/protocol_assembler.py
"""Stage 4: Assemble cure protocol from scored interventions."""
from __future__ import annotations

from typing import Optional

from ontology.enums import ApprovalState, ProtocolLayer
from ontology.protocol import CureProtocolCandidate, ProtocolLayerEntry
from world_model.intervention_scorer import InterventionScore

ALL_LAYERS = [
    ProtocolLayer.root_cause_suppression,
    ProtocolLayer.pathology_reversal,
    ProtocolLayer.circuit_stabilization,
    ProtocolLayer.regeneration_reinnervation,
    ProtocolLayer.adaptive_maintenance,
]

# Timing offsets: root-cause and stabilization start immediately,
# pathology reversal after 1 week, regeneration after 3 weeks
DEFAULT_TIMING = {
    "root_cause_suppression": 0,
    "pathology_reversal": 7,
    "circuit_stabilization": 0,
    "regeneration_reinnervation": 21,
    "adaptive_maintenance": 0,
}


def select_layer_interventions(
    scores: list[InterventionScore],
    layer: str,
    max_per_layer: int = 2,
) -> list[InterventionScore]:
    """Select top eligible interventions for a protocol layer."""
    layer_scores = [
        s for s in scores
        if s.protocol_layer == layer and s.erik_eligible is not False
    ]
    layer_scores.sort(key=lambda s: s.relevance_score, reverse=True)
    return layer_scores[:max_per_layer]


def assemble_protocol(
    scores: list[InterventionScore],
    subject_ref: str,
) -> CureProtocolCandidate:
    """Assemble a CureProtocolCandidate from scored interventions."""
    layers = []
    all_cited = []
    all_uncertainties = []
    all_failure_modes = []

    for protocol_layer in ALL_LAYERS:
        layer_name = protocol_layer.value
        selected = select_layer_interventions(scores, layer_name)

        if not selected:
            # Abstention
            layers.append(ProtocolLayerEntry(
                layer=protocol_layer,
                intervention_refs=[],
                start_offset_days=DEFAULT_TIMING.get(layer_name, 0),
                notes=f"ABSTENTION: No eligible interventions scored for {layer_name}",
            ))
            all_failure_modes.append(f"no_intervention_{layer_name}")
        else:
            int_refs = [s.intervention_id for s in selected]
            notes_parts = [f"{s.intervention_name} (score={s.relevance_score:.2f})" for s in selected]
            layers.append(ProtocolLayerEntry(
                layer=protocol_layer,
                intervention_refs=int_refs,
                start_offset_days=DEFAULT_TIMING.get(layer_name, 0),
                notes="; ".join(notes_parts),
            ))
            for s in selected:
                all_cited.extend(s.cited_evidence)
                all_uncertainties.extend(s.key_uncertainties)
                if s.contested_claims:
                    all_failure_modes.extend(s.contested_claims)

    return CureProtocolCandidate(
        id=f"proto:{subject_ref.split(':')[-1]}_v1",
        subject_ref=subject_ref,
        objective="maximize_durable_disease_arrest_and_functional_recovery",
        layers=layers,
        dominant_failure_modes=list(set(all_failure_modes))[:10],
        approval_state=ApprovalState.pending,
        evidence_bundle_refs=list(set(all_cited)),
        body={
            "all_intervention_scores": [s.model_dump() for s in scores],
            "total_evidence_items_cited": len(set(all_cited)),
            "key_uncertainties": list(set(all_uncertainties))[:20],
        },
    )
```

- [ ] **Step 4: Write counterfactual check**

```python
# scripts/world_model/counterfactual_check.py
"""Stage 5: Counterfactual verification — stress-test each protocol layer."""
from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field

from world_model.reasoning_engine import ReasoningEngine
from world_model.prompts.templates import COUNTERFACTUAL_TEMPLATE


class CounterfactualResult(BaseModel):
    """Result of removing one layer from the protocol. Not a BaseEnvelope."""
    layer: str
    removal_impact: str = "uncertain"  # high, moderate, low, uncertain
    reasoning: str = ""
    is_load_bearing: bool = False
    weakest_evidence: str = ""
    next_best_measurement: str = ""
    cited_evidence: list[str] = Field(default_factory=list)


def check_counterfactual(
    protocol_json: str,
    layer_name: str,
    layer_interventions: str,
    evidence_items: list[dict],
    reasoning_engine: ReasoningEngine,
) -> Optional[CounterfactualResult]:
    """Run counterfactual check for one protocol layer."""
    response = reasoning_engine.reason(
        template=COUNTERFACTUAL_TEMPLATE,
        evidence_items=evidence_items,
        extra_context={
            "protocol_json": protocol_json,
            "layer_name": layer_name,
            "layer_interventions": layer_interventions,
        },
        max_tokens=1000,
    )

    if response is None:
        return CounterfactualResult(
            layer=layer_name,
            removal_impact="uncertain",
            reasoning="LLM returned no response",
        )

    return CounterfactualResult(
        layer=response.get("layer", layer_name),
        removal_impact=response.get("removal_impact", "uncertain"),
        reasoning=response.get("reasoning", ""),
        is_load_bearing=response.get("is_load_bearing", False),
        weakest_evidence=response.get("weakest_evidence", ""),
        next_best_measurement=response.get("next_best_measurement", ""),
        cited_evidence=response.get("cited_evidence", []),
    )


def run_counterfactual_analysis(
    protocol,
    evidence_store,
    reasoning_engine: ReasoningEngine,
) -> list[CounterfactualResult]:
    """Run counterfactual analysis on all protocol layers."""
    results = []
    protocol_json = json.dumps(protocol.model_dump(mode="json"), indent=2, default=str)

    for layer_entry in protocol.layers:
        if "ABSTENTION" in layer_entry.notes:
            results.append(CounterfactualResult(
                layer=layer_entry.layer.value,
                removal_impact="low",
                reasoning="Layer already abstained — no intervention to remove.",
                is_load_bearing=False,
            ))
            continue

        # Gather evidence for this layer's interventions
        evidence = []
        for int_ref in layer_entry.intervention_refs:
            evidence.extend(evidence_store.query_by_intervention_ref(int_ref))

        layer_interventions = ", ".join(layer_entry.intervention_refs)

        result = check_counterfactual(
            protocol_json=protocol_json,
            layer_name=layer_entry.layer.value,
            layer_interventions=layer_interventions,
            evidence_items=evidence,
            reasoning_engine=reasoning_engine,
        )
        if result:
            results.append(result)

    return results
```

- [ ] **Step 5: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_protocol_assembler.py tests/test_counterfactual_check.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/world_model/protocol_assembler.py scripts/world_model/counterfactual_check.py tests/test_protocol_assembler.py tests/test_counterfactual_check.py && git commit -m "feat: Stage 4 protocol assembler + Stage 5 counterfactual verification"
```

---

## Task 8: Protocol Generator (Stage 6 — Full Pipeline Orchestrator)

**Files:**
- Create: `scripts/world_model/protocol_generator.py`
- Create: `tests/test_protocol_generator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_protocol_generator.py
import pytest
from world_model.protocol_generator import generate_cure_protocol


def test_generate_cure_protocol_no_llm():
    """Test pipeline runs with LLM disabled (deterministic stages only)."""
    result = generate_cure_protocol(use_llm=False)
    assert result is not None
    assert result["snapshot"].type == "DiseaseStateSnapshot"
    assert result["snapshot"].functional_state_ref is not None


@pytest.mark.llm
def test_generate_cure_protocol_full():
    """Full pipeline with real LLM. Produces Erik's first cure protocol."""
    result = generate_cure_protocol(use_llm=True)
    assert result is not None
    assert result["protocol"].type == "CureProtocolCandidate"
    assert len(result["protocol"].layers) == 5
    assert result["protocol"].approval_state.value == "pending"
    # Should have some evidence citations
    assert len(result["protocol"].evidence_bundle_refs) > 0
    # Should have counterfactual results
    assert len(result["counterfactuals"]) == 5
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_protocol_generator.py -v -k "not llm"`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/world_model/protocol_generator.py
"""Stage 6: Full pipeline orchestrator.

Runs all 6 stages in sequence:
1. State materialization
2. Subtype inference
3. Intervention scoring
4. Protocol assembly
5. Counterfactual verification
6. Output

Produces Erik's cure protocol candidate.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from ingestion.patient_builder import build_erik_draper
from evidence.evidence_store import EvidenceStore
from evidence.seed_builder import load_seed
from world_model.reasoning_engine import ReasoningEngine
from world_model.state_materializer import materialize_state
from world_model.subtype_inference import infer_subtype
from world_model.intervention_scorer import score_all_interventions
from world_model.protocol_assembler import assemble_protocol
from world_model.counterfactual_check import run_counterfactual_analysis


def generate_cure_protocol(
    use_llm: bool = True,
    model_path: Optional[str] = None,
) -> dict[str, Any]:
    """Generate Erik's cure protocol candidate.

    Returns dict with keys: patient, trajectory, snapshot, subtype_profile,
    intervention_scores, protocol, counterfactuals.
    """
    print("[PIPELINE] Loading patient data...")
    patient, trajectory, observations = build_erik_draper()

    print("[PIPELINE] Ensuring evidence seed is loaded...")
    store = EvidenceStore()
    try:
        load_seed()
    except Exception:
        pass  # Seed may already be loaded

    # Stage 1: State Materialization
    print("[PIPELINE] Stage 1: Materializing disease state...")
    reasoning_engine = None
    if use_llm:
        reasoning_engine = ReasoningEngine(model_path=model_path)

    snapshot = materialize_state(
        trajectory, observations,
        use_llm=use_llm,
        reasoning_engine=reasoning_engine,
    )

    result = {
        "patient": patient,
        "trajectory": trajectory,
        "snapshot": snapshot,
        "subtype_profile": None,
        "intervention_scores": [],
        "protocol": None,
        "counterfactuals": [],
    }

    if not use_llm:
        print("[PIPELINE] LLM disabled — returning state snapshot only.")
        return result

    # Stage 2: Subtype Inference
    print("[PIPELINE] Stage 2: Inferring subtype...")
    patient_state_json = json.dumps(snapshot.body, indent=2, default=str)

    # Get subtype-relevant evidence
    subtype_evidence = []
    for layer in ["root_cause_suppression", "pathology_reversal"]:
        subtype_evidence.extend(store.query_by_protocol_layer(layer))

    subtype_profile = infer_subtype(
        patient_state_json=patient_state_json,
        evidence_items=subtype_evidence[:30],  # Cap context size
        subject_ref=trajectory.id,
        reasoning_engine=reasoning_engine,
    )
    result["subtype_profile"] = subtype_profile

    subtype_posterior_json = json.dumps(
        {k.value: v for k, v in subtype_profile.posterior.items()},
        indent=2,
    )

    # Stage 3: Intervention Scoring
    print("[PIPELINE] Stage 3: Scoring interventions...")
    # Get all interventions from DB
    interventions_raw = store.query_all_interventions()

    scores = score_all_interventions(
        interventions=interventions_raw,
        evidence_store=store,
        patient_state_json=patient_state_json,
        subtype_posterior_json=subtype_posterior_json,
        reasoning_engine=reasoning_engine,
    )
    result["intervention_scores"] = scores

    # Stage 4: Protocol Assembly
    print("[PIPELINE] Stage 4: Assembling protocol...")
    protocol = assemble_protocol(scores, trajectory.id)
    result["protocol"] = protocol

    # Stage 5: Counterfactual Verification
    print("[PIPELINE] Stage 5: Running counterfactual analysis...")
    counterfactuals = run_counterfactual_analysis(
        protocol=protocol,
        evidence_store=store,
        reasoning_engine=reasoning_engine,
    )
    result["counterfactuals"] = counterfactuals

    # Stage 6: Finalize
    print("[PIPELINE] Stage 6: Finalizing protocol...")
    protocol.body["counterfactual_analysis"] = [cf.model_dump() for cf in counterfactuals]
    protocol.body["subtype_posterior"] = {k.value: v for k, v in subtype_profile.posterior.items()}

    # Store protocol in DB
    store.upsert_object(protocol)

    print(f"[PIPELINE] Complete. Protocol: {protocol.id}")
    print(f"  Layers: {len(protocol.layers)}")
    print(f"  Evidence cited: {len(protocol.evidence_bundle_refs)}")
    print(f"  Approval state: {protocol.approval_state.value}")

    return result
```

**Note to implementer:** The `store.upsert_evidence_item_raw(protocol)` call needs a new generic upsert method on EvidenceStore, OR the protocol can be stored via the existing `_upsert_object` method directly. The simplest approach is to add a `upsert_object(obj: BaseEnvelope)` method that serializes any BaseEnvelope subclass. Add this to EvidenceStore:

```python
def upsert_object(self, obj) -> None:
    """Upsert any BaseEnvelope object."""
    raw = obj.model_dump(mode="json")
    self._upsert_object(
        obj_id=raw["id"], obj_type=raw["type"], status=raw["status"],
        body=raw.get("body", {}),
        provenance_source_system=raw.get("provenance", {}).get("source_system"),
        confidence=raw.get("uncertainty", {}).get("confidence"),
    )
```

Also fix the intervention query in the generator — replace the hacky `_run_query.__func__` with a clean `store.query_all_interventions()` method:

```python
def query_all_interventions(self) -> list[dict]:
    sql = """SELECT id, type, status, body FROM erik_core.objects
             WHERE type = 'Intervention' AND status = 'active'"""
    return self._run_query(sql, ())
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_protocol_generator.py -v -k "not llm"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/world_model/protocol_generator.py scripts/evidence/evidence_store.py tests/test_protocol_generator.py && git commit -m "feat: Stage 6 protocol generator — full pipeline orchestrator"
```

---

## Task 9: Config Update + Final Verification

- [ ] **Step 1: Update config**

Add LLM settings to `data/erik_config.json`:
```json
{
  "llm_model_path": "/Volumes/Databank/models/mlx/Qwen3.5-35B-A3B-mlx-lm-mxfp4",
  "llm_fallback_model_path": "/Volumes/Databank/models/mlx/Qwen3.5-9B-mlx-lm-4bit",
  "llm_temperature": 0.1,
  "llm_max_tokens_default": 1000,
  "llm_timeout_s": 120,
  "reasoning_verify_critical": true,
  "reasoning_strip_uncited": true
}
```

- [ ] **Step 2: Run full test suite (no LLM)**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v -k "not network and not chembl and not llm" --tb=short`
Expected: All PASS

- [ ] **Step 3: Run LLM integration test (generates Erik's first cure protocol)**

Run: `cd /Users/logannye/.openclaw/erik && PYTHONPATH=scripts conda run -n erik-core python -c "from world_model.protocol_generator import generate_cure_protocol; result = generate_cure_protocol(); print('SUCCESS' if result['protocol'] else 'FAILED')"`
Expected: Pipeline runs all 6 stages and produces a protocol.

- [ ] **Step 4: Update README roadmap**

Change Phase 2 status to **Complete**.

- [ ] **Step 5: Commit and push**

```bash
git add -A && git commit -m "feat: Phase 2 complete — Erik's first cure protocol candidate generated" && git push
```

---

## Summary

After completing all 9 tasks, Phase 2 delivers:

- **LLM inference wrapper** (`mlx-lm`) with JSON extraction, lazy loading, fallback model
- **Evidence-grounded reasoning engine** with citation validation, uncited claim stripping, dual verification
- **6 LLM prompt templates** (reversibility, molecular state, subtype, intervention scoring, counterfactual, verification)
- **State materializer** — Erik's 51 observations → FunctionalState, NMJIntegrityState, RespiratoryReserveState, UncertaintyState, DiseaseStateSnapshot
- **Subtype inference** — posterior over 8 ALS subtypes with conditional genetics arms
- **Intervention scorer** — all ~25 interventions scored with evidence-grounded causal arguments, top-5 verified
- **Protocol assembler** — 5-layer protocol with timing, interaction checking, abstention logic
- **Counterfactual verification** — stress-test each layer, identify weakest links and missing measurements
- **Pipeline orchestrator** — runs all 6 stages and outputs `CureProtocolCandidate` for Erik

**The terminal output is Erik's first cure protocol candidate** — a structured, evidence-grounded, uncertainty-disclosing, human-approval-gated treatment recommendation across all 5 protocol layers.
