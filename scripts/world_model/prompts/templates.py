"""LLM prompt templates for the Erik pipeline stages.

All templates require strict evidence-grounding: every factual claim must cite
an evidence ID (``evi:*``) or intervention ID (``int:*``) from the provided
context.  The model must NOT use training-data knowledge that is not reflected
in the supplied evidence items.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt (injected as the system/role message for every call)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the Erik ALS Reasoning Engine — a strict evidence-grounded medical AI.

RULES (non-negotiable):
1. Every factual claim MUST be supported by a cited evidence item.
   Cite using its exact ID, e.g. evi:abc123 or int:xyz789.
2. Do NOT use knowledge from your training data that is not present in the
   provided evidence items.  If evidence is absent, say so explicitly.
3. Output ONLY valid JSON matching the schema described in the user prompt.
   No markdown, no prose outside the JSON object.
4. Weight all claims by the evidence strength field and PCH level:
   - L3 (counterfactual/interventional): highest weight
   - L2 (correlational with strong provenance): medium weight
   - L1 (observational/hypothesis): lowest weight
5. If evidence is contradictory, report both sides and flag as "contested".
6. Never fabricate numbers, trial results, or mechanistic claims.
"""

# ---------------------------------------------------------------------------
# Stage 0 — Reversibility window estimation
# ---------------------------------------------------------------------------

REVERSIBILITY_TEMPLATE = """\
TASK: Estimate the reversibility window for the ALS patient described below.

PATIENT STATE:
{patient_state_json}

EVIDENCE ITEMS:
{evidence_items_json}

Return a single JSON object with this exact schema:
{{
  "overall_reversibility_score": <float 0.0–1.0, higher = more reversible>,
  "molecular_correction_plausibility": <float 0.0–1.0>,
  "nmj_recovery_plausibility": <float 0.0–1.0>,
  "functional_recovery_plausibility": <float 0.0–1.0>,
  "dominant_bottleneck": <string — the single most limiting factor>,
  "estimated_time_sensitivity_days": <int — days before window closes>,
  "reasoning": <string — concise mechanistic reasoning citing evidence IDs>,
  "cited_evidence": [<list of evi:* or int:* IDs used>]
}}
"""

# ---------------------------------------------------------------------------
# Stage 1 — Molecular state estimation
# ---------------------------------------------------------------------------

MOLECULAR_STATE_TEMPLATE = """\
TASK: Estimate the current molecular state of the ALS patient from population
evidence, weighted by the dominant subtype.

PATIENT STATE:
{patient_state_json}

DOMINANT SUBTYPE: {dominant_subtype}

EVIDENCE ITEMS:
{evidence_items_json}

Return a single JSON object with this exact schema:
{{
  "tdp43_mislocalization_score": <float 0.0–1.0>,
  "tdp43_aggregation_score": <float 0.0–1.0>,
  "tdp43_nuclear_depletion_score": <float 0.0–1.0>,
  "splicing_dysregulation_score": <float 0.0–1.0>,
  "cryptic_exon_burden_score": <float 0.0–1.0>,
  "astrocyte_reactivity_score": <float 0.0–1.0>,
  "microglia_activation_score": <float 0.0–1.0>,
  "reasoning": <string — concise mechanistic reasoning citing evidence IDs>,
  "cited_evidence": [<list of evi:* or int:* IDs used>]
}}
"""

# ---------------------------------------------------------------------------
# Stage 2 — Subtype inference
# ---------------------------------------------------------------------------

SUBTYPE_TEMPLATE = """\
TASK: Infer the posterior probability distribution over ALS subtypes for the
patient described below.

PATIENT STATE:
{patient_state_json}

EVIDENCE ITEMS:
{evidence_items_json}

The eight subtypes are:
  tdp43_aggregation, fus_proteinopathy, sod1_misfolding, c9orf72_dipeptide,
  ataxin2_stress_granule, hnrnpa1_prion, sporadic_tbi_linked, unknown

Return a single JSON object with this exact schema:
{{
  "posterior": {{
    "tdp43_aggregation": <float>,
    "fus_proteinopathy": <float>,
    "sod1_misfolding": <float>,
    "c9orf72_dipeptide": <float>,
    "ataxin2_stress_granule": <float>,
    "hnrnpa1_prion": <float>,
    "sporadic_tbi_linked": <float>,
    "unknown": <float>
  }},
  "conditional_on_genetics": <string — note any genetic constraints applied>,
  "reasoning": <string — concise reasoning citing evidence IDs>,
  "cited_evidence": [<list of evi:* or int:* IDs used>]
}}

All posterior values must sum to 1.0.
"""

# ---------------------------------------------------------------------------
# Stage 3 — Intervention scoring
# ---------------------------------------------------------------------------

INTERVENTION_SCORING_TEMPLATE = """\
TASK: Score the relevance and predicted efficacy of the intervention below for
the patient described, given the subtype posterior and supporting evidence.

PATIENT STATE:
{patient_state_json}

SUBTYPE POSTERIOR:
{subtype_posterior_json}

INTERVENTION:
{intervention_json}

EVIDENCE ITEMS:
{evidence_items_json}

Return a single JSON object with this exact schema:
{{
  "intervention_id": <string — matches the id field of INTERVENTION>,
  "intervention_name": <string>,
  "protocol_layer": <string>,
  "relevance_score": <float 0.0–1.0>,
  "mechanism_argument": <string — why this intervention addresses the dominant pathology>,
  "evidence_strength": <"strong" | "moderate" | "weak" | "absent">,
  "erik_eligible": <bool — true if relevance_score >= 0.4 and evidence_strength != "absent">,
  "key_uncertainties": [<list of strings>],
  "cited_evidence": [<list of evi:* or int:* IDs used>]
}}
"""

# ---------------------------------------------------------------------------
# Stage 4 — Counterfactual layer assessment
# ---------------------------------------------------------------------------

COUNTERFACTUAL_TEMPLATE = """\
TASK: Assess the impact of removing the named protocol layer from the assembled
treatment protocol.  Determine whether the layer is load-bearing.

PROTOCOL:
{protocol_json}

LAYER TO REMOVE: {layer_name}

INTERVENTIONS IN THIS LAYER:
{layer_interventions}

EVIDENCE ITEMS:
{evidence_items_json}

Return a single JSON object with this exact schema:
{{
  "layer": <string — matches layer_name>,
  "removal_impact": <"critical" | "significant" | "moderate" | "minimal">,
  "reasoning": <string — mechanistic argument citing evidence IDs>,
  "is_load_bearing": <bool>,
  "weakest_evidence": <string — ID of the weakest evidence item cited for this layer>,
  "next_best_measurement": <string — the single most valuable missing piece of evidence>,
  "cited_evidence": [<list of evi:* or int:* IDs used>]
}}
"""

# ---------------------------------------------------------------------------
# Stage 5 — Claim verification
# ---------------------------------------------------------------------------

VERIFICATION_TEMPLATE = """\
TASK: Verify the following claim against the provided evidence items.

CLAIM:
{claim_text}

EVIDENCE ITEMS:
{evidence_items_json}

Return a single JSON object with this exact schema:
{{
  "claim": <string — verbatim copy of the claim>,
  "verdict": <"supported" | "partially_supported" | "unsupported" | "contested">,
  "reasoning": <string — explanation citing specific evidence IDs>,
  "cited_evidence": [<list of evi:* or int:* IDs used>]
}}
"""

# ---------------------------------------------------------------------------
# Physician Report — Clinical summary + recommendations for treating physician
# ---------------------------------------------------------------------------

PHYSICIAN_REPORT_TEMPLATE = """\
TASK: Generate a clinical summary and actionable recommendations for the
treating physician of Erik Draper (67M, limb-onset ALS, diagnosed March 2026).

This report must be evidence-grounded, concise, and actionable. The physician
is Dr. Thakore at Cleveland Clinic. Write for a neurologist who understands
ALS but has not seen this system's analysis before.

PATIENT DISEASE STATE:
{patient_state_json}

CURRENT CURE PROTOCOL (5-layer):
{protocol_json}

TOP EVIDENCE SUMMARY (most relevant findings):
{evidence_items_json}

OPEN CAUSAL GAPS (what we don't yet understand):
{causal_gaps_json}

Return a single JSON object:
{{
  "clinical_summary": <string — 3-4 sentence overview of Erik's current status, trajectory, and the system's assessment>,
  "protocol_explanation": <string — plain-language explanation of the 5-layer treatment strategy and why these interventions were selected>,
  "recommendations": [
    {{
      "recommendation": <string — specific, actionable recommendation>,
      "rationale": <string — why this is recommended, citing evidence IDs>,
      "timeline": <"immediate" | "within_week" | "within_month" | "within_quarter">,
      "type": <"genetic_testing" | "biomarker" | "imaging" | "trial_enrollment" | "intervention_start" | "monitoring" | "referral">,
      "cited_evidence": [<list of evi:* IDs>]
    }}
  ],
  "key_uncertainties": [<string — top 5 things we don't know that affect treatment decisions>],
  "cited_evidence": [<all evi:* and int:* IDs referenced anywhere in this report>]
}}
"""
