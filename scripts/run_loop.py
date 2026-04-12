#!/usr/bin/env python3
"""Erik Research Loop — continuous 24/7 execution entry point.

This is the LaunchAgent target. It runs the research loop continuously:
- Active mode: 15-action research loop seeking convergence
- Monitoring mode: after convergence, slow-poll for new data that
  would re-open research (genetic results, new trial readouts, new
  clinical data for Erik)

State is persisted to PostgreSQL and resumed on restart. The process
can be killed and restarted at any time without losing progress.

Usage:
    # Direct (with live output):
    PYTHONPATH=scripts /opt/homebrew/Caskroom/miniconda/base/envs/erik-core/bin/python scripts/run_loop.py

    # Via LaunchAgent (24/7):
    launchctl load ~/Library/LaunchAgents/ai.erik.researcher.plist
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import threading
from dataclasses import replace

# Force unbuffered stdout/stderr so LaunchAgent logs write immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Ensure scripts/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from config.loader import ConfigLoader
from db.pool import get_connection
from evidence.evidence_store import EvidenceStore
from research.dual_llm import DualLLMManager
from research.loop import research_step, _bootstrap_initial_protocol, _persist_state
from research.state import ResearchState, initial_state
from research.layer_orchestrator import determine_layer
from research.policy import _BASE_LAYER_QUERIES
from daemons.integration_daemon import IntegrationDaemon
from daemons.reasoning_daemon import ReasoningDaemon
from daemons.compound_daemon import CompoundDaemon


# ---------------------------------------------------------------------------
# Cognitive engine daemons
# ---------------------------------------------------------------------------

def _start_cognitive_daemons(claude_api_key: str = "") -> dict[str, tuple]:
    """Start all cognitive engine daemons as background threads.

    Returns a dict mapping daemon name to (daemon_instance, thread) tuples.
    """
    cfg = ConfigLoader()
    daemons: dict[str, tuple] = {}

    # Phase 2: Integration
    if cfg.get("integration_enabled", True):
        integration = IntegrationDaemon()
        t_int = threading.Thread(target=integration.run, name="integration-daemon", daemon=True)
        t_int.start()
        daemons["integration"] = (integration, t_int)
        print("[ERIK] Started IntegrationDaemon")

    # Phase 3: Reasoning (requires Claude API key)
    if cfg.get("reasoning_enabled", True) and claude_api_key:
        reasoning = ReasoningDaemon(claude_api_key=claude_api_key)
        t_rea = threading.Thread(target=reasoning.run, name="reasoning-daemon", daemon=True)
        t_rea.start()
        daemons["reasoning"] = (reasoning, t_rea)
        print("[ERIK] Started ReasoningDaemon")

    # Phase 4: Compound (requires Claude API key)
    if cfg.get("compound_enabled", True) and claude_api_key:
        compound = CompoundDaemon(claude_api_key=claude_api_key)
        t_cmp = threading.Thread(target=compound.run, name="compound-daemon", daemon=True)
        t_cmp.start()
        daemons["compound"] = (compound, t_cmp)
        print("[ERIK] Started CompoundDaemon")

    return daemons


def _stop_cognitive_daemons(daemons: dict[str, tuple]) -> None:
    """Gracefully stop all running daemons."""
    for name, (daemon, thread) in daemons.items():
        daemon.stop()
        thread.join(timeout=5)
        print(f"[ERIK] Stopped {name} daemon")


# ---------------------------------------------------------------------------
# State resume
# ---------------------------------------------------------------------------

def _load_state_from_db(subject_ref: str) -> ResearchState | None:
    """Load the most recent research state from PostgreSQL.

    Returns None if no state exists (first run).
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT state_json FROM erik_ops.research_state WHERE subject_ref = %s",
                    (subject_ref,),
                )
                row = cur.fetchone()
                if row:
                    data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                    return ResearchState.from_dict(data)
    except Exception as e:
        print(f"[ERIK] Could not load state from DB: {e}")
    return None


def _sanitize_resumed_state(state: ResearchState) -> ResearchState:
    """Clean up stale state on restart.

    Fixes accumulated issues that can't self-heal during normal execution:
    1. Flush old-format hypothesis IDs (hyp:...) — replaced by statements in P6
    2. Reset all-zero EMA action values — gives every action a fair chance
    3. Reset stale Thompson posteriors — prevents inherited bias from stall period
    """
    changes: dict[str, object] = {}
    repairs: list[str] = []

    # 1. Flush old-format hypothesis IDs (hyp:...) from active_hypotheses
    old_hyps = state.active_hypotheses
    clean_hyps = [h for h in old_hyps if not h.startswith("hyp:")]
    if len(clean_hyps) < len(old_hyps):
        changes["active_hypotheses"] = clean_hyps
        repairs.append(f"flushed {len(old_hyps) - len(clean_hyps)} old-format hypothesis IDs")

    # 2. Reset all-zero EMA action values — give actions a fair optimistic start
    if state.action_values:
        meaningful = sum(1 for v in state.action_values.values() if v > 0.01)
        if meaningful <= 3:  # Most actions are effectively dead (stall state)
            changes["action_values"] = {}  # Empty = optimistic default (1.0) in policy
            repairs.append(f"reset {len(state.action_values)} stale action EMA values")

    # 3. Reset stale Thompson posteriors to uniform (1, 1) for fresh learning
    if state.action_posteriors:
        changes["action_posteriors"] = {}
        repairs.append(f"reset {len(state.action_posteriors)} Thompson posteriors to uniform")

    if repairs:
        state = replace(state, **changes)
        for r in repairs:
            print(f"[ERIK] Sanitize: {r}")
    else:
        print("[ERIK] Sanitize: state is clean, no repairs needed")

    return state


# ---------------------------------------------------------------------------
# Deep research mode (post-convergence continuous evidence expansion)
# ---------------------------------------------------------------------------

# Systematic queries to run in deep research mode — rotated through
_DEEP_RESEARCH_QUERIES = [
    # PubMed: specific mechanism queries beyond the 5 layer queries
    ("search_pubmed", {"query": "ALS TDP-43 intrabody gene therapy 2025 2026"}),
    ("search_pubmed", {"query": "ALS antisense oligonucleotide STMN2 UNC13A 2025 2026"}),
    ("search_pubmed", {"query": "ALS sigma-1R agonist neuroprotection clinical trial"}),
    ("search_pubmed", {"query": "ALS C9orf72 repeat expansion therapy antisense 2025 2026"}),
    ("search_pubmed", {"query": "ALS riluzole combination therapy augmentation 2025"}),
    ("search_pubmed", {"query": "ALS biomarker neurofilament treatment response prediction"}),
    ("search_pubmed", {"query": "ALS drug combination synergy preclinical motor neuron"}),
    ("search_pubmed", {"query": "ALS rapamycin mTOR autophagy proteostasis clinical"}),
    ("search_pubmed", {"query": "ALS masitinib tyrosine kinase inhibitor neuroinflammation Phase 3"}),
    ("search_pubmed", {"query": "ALS ibudilast phosphodiesterase neuroinflammation clinical"}),
    # Clinical trials
    ("search_trials", {}),
    # STRING PPI for each major target
    ("query_ppi_network", {"gene_symbol": "TARDBP"}),
    ("query_ppi_network", {"gene_symbol": "SIGMAR1"}),
    ("query_ppi_network", {"gene_symbol": "SOD1"}),
    ("query_ppi_network", {"gene_symbol": "FUS"}),
    ("query_ppi_network", {"gene_symbol": "SLC1A2"}),  # EAAT2
    ("query_ppi_network", {"gene_symbol": "MTOR"}),
    # Pathway queries for key targets
    ("query_pathways", {"target_name": "TDP-43"}),
    ("query_pathways", {"target_name": "Sigma-1R"}),
    ("query_pathways", {"target_name": "mTOR"}),
    ("query_pathways", {"target_name": "CSF1R"}),
    # PharmGKB drug safety for protocol drugs
    ("check_pharmacogenomics", {"drug_name": "riluzole"}),
    ("check_pharmacogenomics", {"drug_name": "edaravone"}),
    ("check_pharmacogenomics", {"drug_name": "rapamycin"}),
]


def _get_layer_weighted_query(
    evidence_by_layer: dict[str, int],
    step: int,
) -> tuple[str, str]:
    """Select a query biased toward under-represented evidence layers.

    Uses inverse-proportion weighting so layers with fewer evidence items
    get proportionally more queries.  Selection is deterministic given
    ``step`` (no random state).

    Returns ``(query_string, layer_name)``.
    """
    import datetime as _dt

    # Build inverse-proportion weights for every layer in _BASE_LAYER_QUERIES
    layers = list(_BASE_LAYER_QUERIES.keys())
    weights = [1.0 / (evidence_by_layer.get(layer, 0) + 1) for layer in layers]
    total_weight = sum(weights)

    # Deterministic selection via cumulative scan
    # Use golden-ratio stride for maximal spread across layers
    target = ((step * 61) % 100) / 100.0 * total_weight
    cumulative = 0.0
    chosen_layer = layers[-1]  # fallback
    for layer, w in zip(layers, weights):
        cumulative += w
        if cumulative >= target:
            chosen_layer = layer
            break

    # Pick a query from the chosen layer, rotating by step
    queries = _BASE_LAYER_QUERIES[chosen_layer]
    query = queries[step % len(queries)]
    year = _dt.datetime.now().year
    return (f"{query} {year}", chosen_layer)


def _deep_stagnation_detected(state: ResearchState, window: int = 20) -> bool:
    """Return True if deep mode has produced zero evidence for *window* steps."""
    if state.step_count < window:
        return False
    return (state.step_count - state.last_deep_evidence_step) >= window


def _get_expanded_deep_actions(step: int = 0) -> list[tuple[str, dict]]:
    """Return 30+ action tuples covering all database connectors, LLM actions,
    preprints, and Galen cross-references for use when deep mode is stagnated.

    Each tuple is (action_name_string, params_dict). The caller rotates through
    this list using ``step % len(actions)`` to ensure broad coverage.
    """
    _ALS_GENES = [
        "TARDBP", "SOD1", "FUS", "C9orf72", "SIGMAR1", "MTOR",
        "SLC1A2", "CSF1R", "STMN2", "UNC13A", "ATXN2", "OPTN",
        "TBK1", "NEK1", "VCP", "PFN1",
    ]
    _ALS_DRUGS = [
        "riluzole", "edaravone", "rapamycin", "masitinib",
        "ibudilast", "tofersen", "sodium phenylbutyrate",
    ]
    actions: list[tuple[str, dict]] = []

    # --- Database connector queries (12 distinct action types) ---
    for gene in _ALS_GENES[:6]:
        actions.append(("query_chembl", {"target_name": gene}))
    for gene in _ALS_GENES[:4]:
        actions.append(("query_gwas", {"gene": gene}))
    for gene in _ALS_GENES[:4]:
        actions.append(("query_clinvar", {"gene": gene}))
    for gene in _ALS_GENES[:3]:
        actions.append(("query_bindingdb", {"target_name": gene}))
    for gene in _ALS_GENES[:3]:
        actions.append(("query_hpa", {"gene": gene}))
    for drug in _ALS_DRUGS[:3]:
        actions.append(("query_drugbank", {"drug_name": drug}))
    for gene in _ALS_GENES[:3]:
        actions.append(("query_uniprot", {"gene": gene}))
    for gene in _ALS_GENES[:2]:
        actions.append(("query_alphafold", {"gene": gene}))
    for gene in _ALS_GENES[:4]:
        actions.append(("query_disgenet", {"gene": gene}))
    for gene in _ALS_GENES[:4]:
        actions.append(("query_gnomad", {"gene": gene}))
    for gene in _ALS_GENES[:3]:
        actions.append(("query_geo_als", {"gene": gene}))
    actions.append(("query_reactome_local", {"target_name": "TDP-43"}))
    actions.append(("query_reactome_local", {"target_name": "mTOR signaling"}))
    actions.append(("query_reactome_local", {"target_name": "autophagy"}))

    # --- LLM actions ---
    actions.append(("generate_hypothesis", {"topic": "ALS proteostasis failure mechanisms", "uncertainty": "TDP-43 aggregation triggers"}))
    actions.append(("generate_hypothesis", {"topic": "ALS neuroinflammation therapeutic targets", "uncertainty": "CSF1R inhibition safety"}))
    actions.append(("generate_hypothesis", {"topic": "ALS RNA metabolism dysfunction", "uncertainty": "STMN2 cryptic exon as biomarker"}))
    actions.append(("deepen_causal_chain", {"intervention_id": "TDP-43 proteostasis"}))
    actions.append(("deepen_causal_chain", {"intervention_id": "SOD1 aggregation"}))

    # --- Preprints ---
    actions.append(("search_preprints", {"query": "ALS motor neuron degeneration mechanism 2025 2026"}))
    actions.append(("search_preprints", {"query": "ALS gene therapy clinical 2025 2026"}))
    actions.append(("search_preprints", {"query": "ALS biomarker neurofilament 2025 2026"}))

    # --- Galen cross-references ---
    actions.append(("query_galen_kg", {"entity_type": "gene", "entity_name": "TARDBP"}))
    actions.append(("query_galen_kg", {"entity_type": "gene", "entity_name": "SOD1"}))
    actions.append(("query_galen_kg", {"entity_type": "drug", "entity_name": "riluzole"}))

    # --- Original 5 action types for completeness ---
    actions.append(("search_pubmed", {"query": "ALS novel therapeutic target 2025 2026"}))
    actions.append(("search_pubmed", {"query": "ALS stem cell therapy motor neuron 2025"}))
    actions.append(("search_trials", {}))
    for gene in ["TARDBP", "NEK1", "OPTN"]:
        actions.append(("query_ppi_network", {"gene_symbol": gene}))
    actions.append(("query_pathways", {"target_name": "autophagy"}))
    actions.append(("query_pathways", {"target_name": "ubiquitin proteasome"}))
    actions.append(("check_pharmacogenomics", {"drug_name": "tofersen"}))
    actions.append(("check_pharmacogenomics", {"drug_name": "masitinib"}))

    return actions


def _select_deep_action(
    state: ResearchState,
    is_stagnated: bool,
) -> tuple[str, dict]:
    """Choose an action for deep mode, prioritizing hypothesis lifecycle.

    Returns (action_name, params).  An empty action_name signals the caller
    to fall through to the existing gap-driven / rotation logic.

    Rules:
    - If stagnated: return an expanded action from the full action set.
    - Every 10th step (step_count % 10 == 0) AND not stagnated:
        - If active_hypotheses is non-empty: validate the first one.
        - Otherwise: generate a new hypothesis.
    - All other cases: return ("", {}) so the caller uses normal logic.
    """
    if is_stagnated:
        expanded = _get_expanded_deep_actions(state.step_count)
        idx = state.step_count % len(expanded)
        return expanded[idx]

    if state.step_count % 10 == 0:
        if state.active_hypotheses:
            return ("validate_hypothesis", {"hypothesis_id": state.active_hypotheses[0]})
        else:
            return ("generate_hypothesis", {"topic": "ALS mechanism", "uncertainty": ""})

    return ("", {})


def _deep_research_step(
    state: ResearchState,
    evidence_store: EvidenceStore,
    llm_manager: DualLLMManager,
) -> ResearchState:
    """Execute one deep research step -- intelligent evidence expansion
    that continues even after protocol convergence.

    When stagnated (zero new evidence for ``deep_stagnation_window`` steps),
    switches to the expanded action set covering all database connectors,
    LLM actions, preprints, and Galen cross-references.

    Otherwise alternates between:
    - Gap-driven research: analyze protocol gaps, search for what matters most
    - Systematic rotation: cycle through hardcoded queries for broad coverage

    Every 3rd step is gap-driven (uses protocol gap analysis to pick the
    most impactful query). Other steps use the hardcoded rotation.
    """
    from research.actions import ActionType, ActionResult
    from research.loop import _execute_action, _persist_state
    from research.rewards import compute_reward

    cfg = ConfigLoader()
    stagnation_window = cfg.get("deep_stagnation_window", 50)
    stagnated = _deep_stagnation_detected(state, window=stagnation_window)

    # Build action_name -> ActionType map from the full enum
    _action_type_map = {at.value: at for at in ActionType}

    # --- Priority 1: hypothesis lifecycle / stagnation breaker ---
    action_name, params = _select_deep_action(state, is_stagnated=stagnated)

    if action_name:
        if stagnated:
            print(
                f"[ERIK-DEEP] STAGNATION DETECTED ({state.step_count - state.last_deep_evidence_step} steps dry) "
                f"— using expanded action: {action_name}"
            )
        else:
            print(f"[ERIK-DEEP] Hypothesis step: {action_name}")
    else:
        # --- Priority 2: gap-driven + hardcoded rotation ---
        use_gap_analysis = (state.step_count % 3 == 0)

        if use_gap_analysis:
            try:
                from research.intelligence import analyze_protocol_gaps, filter_actionable_gaps
                gaps = analyze_protocol_gaps(state, evidence_store)
                # Log clinical recommendations periodically
                clinical_gaps = [g for g in gaps if g.get("resolvability") == "clinical_required"]
                if clinical_gaps and state.step_count % 50 == 0:
                    for cg in clinical_gaps[:3]:
                        print(f"[ERIK-CLINICAL] Recommendation: {cg['description']} (requires clinical action)")
                actionable_gaps = filter_actionable_gaps(gaps)
                if actionable_gaps:
                    gap = actionable_gaps[0]
                    queries = gap.get("search_queries", [])
                    if queries:
                        # Use the top gap's first search query
                        action_name = "search_pubmed"
                        params = {"query": queries[0], "max_results": 20}
                        print(f"[ERIK-DEEP] Gap-driven: {gap['gap_type']} — {gap['description'][:60]}...")
                    else:
                        use_gap_analysis = False
                else:
                    use_gap_analysis = False
            except Exception:
                use_gap_analysis = False

        if not use_gap_analysis:
            # Alternate between layer-weighted (even) and hardcoded rotation (odd)
            if state.step_count % 2 == 0:
                query, layer = _get_layer_weighted_query(state.evidence_by_layer, state.step_count)
                action_name = "search_pubmed"
                params = {"query": query, "max_results": 20}
                print(f"[ERIK-DEEP] Layer-balanced: targeting {layer}")
            else:
                deep_step = state.step_count % len(_DEEP_RESEARCH_QUERIES)
                action_name, params = _DEEP_RESEARCH_QUERIES[deep_step]

    # Map string to ActionType using full enum map (supports all 37+ action types)
    action = _action_type_map.get(action_name, ActionType.SEARCH_PUBMED)
    params["action"] = action

    # Execute (with evidence deduplication)
    try:
        _db_count_before = evidence_store.count_by_type("EvidenceItem")
    except Exception:
        _db_count_before = None

    result = _execute_action(action, params, state, evidence_store, llm_manager)

    # Use DB delta as the TRUE evidence count (deduplicates upserts)
    if _db_count_before is not None and result.evidence_items_added > 0:
        try:
            _db_count_after = evidence_store.count_by_type("EvidenceItem")
            _true_new = max(0, _db_count_after - _db_count_before)
            result.evidence_items_added = _true_new
        except Exception:
            pass

    # Compute reward — pass through hypothesis_resolved and causal_depth_added
    reward = compute_reward(
        evidence_items_added=result.evidence_items_added,
        uncertainty_before=0.3,
        uncertainty_after=0.3,
        protocol_score_delta=0.0,
        hypothesis_resolved=result.hypothesis_resolved,
        causal_depth_added=result.causal_depth_added,
        interaction_safe=result.interaction_safe,
        eligibility_confirmed=result.eligibility_confirmed,
        protocol_stable=False,
    )

    # Update state
    new_evidence = state.total_evidence_items + result.evidence_items_added
    new_since_regen = state.new_evidence_since_regen + result.evidence_items_added

    # Update action counts
    action_counts = dict(state.action_counts)
    action_counts[action.value] = action_counts.get(action.value, 0) + 1

    # Track last_deep_evidence_step: update when evidence or causal depth was added
    new_last_deep_evidence_step = state.last_deep_evidence_step
    if result.evidence_items_added > 0 or result.causal_depth_added > 0:
        new_last_deep_evidence_step = state.step_count + 1  # +1 because step_count increments below

    state = replace(
        state,
        step_count=state.step_count + 1,
        total_evidence_items=new_evidence,
        new_evidence_since_regen=new_since_regen,
        last_action=f"deep:{action.value}",
        last_reward=reward.total(),
        action_counts=action_counts,
        last_deep_evidence_step=new_last_deep_evidence_step,
    )

    print(
        f"[ERIK-DEEP] Step {state.step_count}: {action.value} | "
        f"evidence={result.evidence_items_added} | "
        f"total={new_evidence} | "
        f"since_regen={new_since_regen}"
        + (f" | STAGNATED (last_evidence@{new_last_deep_evidence_step})" if stagnated else "")
    )

    return state


def _monitoring_cycle(
    state: ResearchState,
    evidence_store: EvidenceStore,
    llm_manager: DualLLMManager,
    regen_threshold: int = 15,
) -> ResearchState:
    """One monitoring cycle — runs deep research AND checks for triggers.

    Instead of passively sleeping, the monitoring cycle:
    1. Checks for genetic results (immediate reactivation trigger)
    2. Runs one deep research step (systematic evidence expansion)
    3. If enough new evidence accumulated, triggers full re-convergence
    """
    cfg = ConfigLoader()

    # One-time force reconverge (set via config, auto-clears)
    if cfg.get("force_active_research", False):
        _min_active = cfg.get("min_active_steps_after_transition", 200)
        print(f"[ERIK-MONITOR] force_active_research=true — {_min_active} active steps required")
        state = replace(state, converged=False, protocol_stable_cycles=0, min_active_steps_remaining=_min_active)
        # Auto-disable so it only fires once
        try:
            import json
            from pathlib import Path
            _cfg_path = Path(__file__).resolve().parent.parent / "data" / "erik_config.json"
            _cfg_data = json.loads(_cfg_path.read_text())
            _cfg_data["force_active_research"] = False
            _cfg_path.write_text(json.dumps(_cfg_data, indent=2) + "\n")
        except Exception:
            pass  # Config write failed; will re-trigger next cycle but that's fine

    # Check if genetic results have arrived
    if cfg.get("genetics_received", False) and "genetics_processed" not in (state.top_uncertainties or []):
        print("[ERIK-MONITOR] Genetic results detected! Re-entering active research.")
        state = replace(
            state,
            converged=False,
            protocol_stable_cycles=0,
            top_uncertainties=["genetics_received_need_interpretation"],
        )
        return state

    # Run a deep research step (evidence expansion continues post-convergence)
    state = _deep_research_step(state, evidence_store, llm_manager)

    # Update research layer
    new_layer = determine_layer(
        evidence_count=state.total_evidence_items,
        genetic_profile=state.genetic_profile,
        validated_targets=sum(1 for d in state.causal_chains.values() if d >= 3),
        provisional_genetics_enabled=cfg.get("provisional_genetics_enabled", False),
        provisional_genetics_min_evidence=cfg.get("provisional_genetics_min_evidence", 500),
    )
    if new_layer.value != state.research_layer:
        _min_active = cfg.get("min_active_steps_after_transition", 200)
        print(f"[ERIK-MONITOR] ★ LAYER TRANSITION: {state.research_layer} → {new_layer.value}")
        print(f"[ERIK-MONITOR] Layer transition invalidates convergence — {_min_active} active steps required")
        state = replace(
            state,
            research_layer=new_layer.value,
            converged=False,
            protocol_stable_cycles=0,
            min_active_steps_remaining=_min_active,
        )

    # Uncertainty-aware re-convergence trigger
    from research.convergence import compute_uncertainty_score
    current_score = compute_uncertainty_score(state)
    history = list(getattr(state, 'uncertainty_history', []))
    history.append(current_score)
    if len(history) > 50:
        history = history[-50:]
    state = replace(state, uncertainty_score=current_score, uncertainty_history=history)

    # Re-enter active research only if uncertainty dropped meaningfully (>5%)
    if len(history) >= 5:
        recent = sum(history[-5:]) / 5
        older = sum(history[-10:-5]) / 5 if len(history) >= 10 else history[0]
        drop = older - recent
        if drop > 0.05:
            print(f"[ERIK-MONITOR] Uncertainty dropped {drop:.3f}. Re-entering active research.")
            state = replace(state, converged=False, protocol_stable_cycles=0)
        elif state.new_evidence_since_regen >= regen_threshold * 2:
            print(f"[ERIK-MONITOR] {state.new_evidence_since_regen} evidence items but uncertainty flat. Regenerating protocol.")
            state = replace(state, converged=False, protocol_stable_cycles=0)

    return state


# ---------------------------------------------------------------------------
# Main continuous loop
# ---------------------------------------------------------------------------

def main():
    subject_ref = "traj:draper_001"
    cfg = ConfigLoader()

    print("[ERIK] ================================================")
    print("[ERIK] CONTINUOUS RESEARCH LOOP — 24/7 MODE")
    print(f"[ERIK] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("[ERIK] ================================================")

    # Try to resume from DB
    state = _load_state_from_db(subject_ref)
    if state is not None:
        print(f"[ERIK] Resumed from DB: step={state.step_count}, "
              f"protocol_v={state.protocol_version}, "
              f"evidence={state.total_evidence_items}, "
              f"converged={state.converged}, "
              f"layer={state.research_layer}")
    else:
        print("[ERIK] No saved state — starting fresh.")
        state = initial_state(subject_ref=subject_ref)

    evidence_store = EvidenceStore()
    llm_manager = DualLLMManager()

    # Initialize evidence count from DB (DB is the source of truth)
    try:
        db_count = evidence_store.count_by_type("EvidenceItem")
        state = replace(state, total_evidence_items=db_count)
        print(f"[ERIK] Evidence in DB: {db_count}")
    except Exception:
        pass

    # Sanitize resumed state: flush stale values that accumulate during stalls
    state = _sanitize_resumed_state(state)

    # Bootstrap if no protocol exists yet
    if state.protocol_version == 0:
        print("[ERIK] No protocol — bootstrapping...")
        state = _bootstrap_initial_protocol(state, evidence_store, llm_manager)
        _persist_state(state, evidence_store)

    # Backfill KG entities from existing evidence (idempotent, runs once on unextracted items)
    if cfg.get("kg_extraction_enabled", True):
        try:
            from knowledge_quality.entity_extractor import extract_kg_from_evidence
            total_entities = 0
            total_rels = 0
            while True:
                stats = extract_kg_from_evidence(batch_size=100)
                if stats["items_processed"] == 0:
                    break
                total_entities += stats["entities_created"]
                total_rels += stats["relationships_created"]
            if total_entities > 0:
                print(f"[ERIK] KG backfill: {total_entities} entities, {total_rels} relationships")
            else:
                print("[ERIK] KG: all evidence already extracted")
        except Exception as e:
            print(f"[ERIK] KG backfill skipped: {e}")

    # Initialize causal gap tracker
    if cfg.get("causal_gap_tracking_enabled", True):
        try:
            from research.causal_gaps import _ensure_gaps_table, seed_gaps_from_targets, count_gaps_by_status
            _ensure_gaps_table()
            new_gaps = seed_gaps_from_targets()
            gap_counts = count_gaps_by_status()
            if new_gaps > 0:
                print(f"[ERIK] Causal gaps: seeded {new_gaps} new gaps")
            print(f"[ERIK] Causal gaps: {gap_counts}")
        except Exception as e:
            print(f"[ERIK] Causal gap init skipped: {e}")

    # Seed TCG scaffold if not already done
    try:
        from tcg.graph import TCGraph
        from tcg.seed_scaffold import seed_scaffold
        _tcg = TCGraph()
        _summary = _tcg.summary()
        if _summary["node_count"] == 0:
            print("[ERIK] Seeding TCG scaffold...")
            stats = seed_scaffold(_tcg)
            print(f"[ERIK] TCG scaffold: {stats['nodes_created']} nodes, {stats['edges_created']} edges")
        else:
            print(f"[ERIK] TCG: {_summary['node_count']} nodes, {_summary['edge_count']} edges, "
                  f"mean confidence {_summary['mean_confidence']:.3f}")
    except Exception as e:
        print(f"[ERIK] TCG scaffold skipped: {e}")

    # Start cognitive engine daemons
    _claude_key = os.environ.get("ANTHROPIC_API_KEY", cfg.get("anthropic_api_key", ""))
    cognitive_daemons = _start_cognitive_daemons(claude_api_key=_claude_key)

    # Config
    regen_threshold = cfg.get("research_protocol_regen_threshold", 15)
    active_pause = cfg.get("research_inter_step_pause_s", 1.0)
    monitoring_interval = 30  # 30 seconds between deep research steps (active evidence expansion)

    print("[ERIK] Entering main loop...")

    while True:
        try:
            if state.converged:
                # Deep research mode: actively expand evidence even while converged
                state = _monitoring_cycle(state, evidence_store, llm_manager, regen_threshold)
                _persist_state(state, evidence_store)

                if state.converged:
                    # Still converged — pause then do another deep research step
                    time.sleep(monitoring_interval)
                    continue
                else:
                    # Re-convergence triggered — fall through to active research
                    print("[ERIK] Re-entering active research mode for re-convergence.")

            # Active research mode — hot-reload config
            cfg.reload_if_changed()
            regen_threshold = cfg.get("research_protocol_regen_threshold", regen_threshold)
            target_depth = cfg.get("research_causal_chain_target_depth", 5)

            state = research_step(
                state=state,
                evidence_store=evidence_store,
                llm_manager=llm_manager,
                dry_run=False,
                regen_threshold=regen_threshold,
                target_depth=target_depth,
            )

            # Update research layer based on current state
            from config.loader import ConfigLoader as _CfgLoader
            _layer_cfg = _CfgLoader()
            new_layer = determine_layer(
                evidence_count=state.total_evidence_items,
                genetic_profile=state.genetic_profile,
                validated_targets=sum(1 for d in state.causal_chains.values() if d >= 3),
                provisional_genetics_enabled=_layer_cfg.get("provisional_genetics_enabled", False),
                provisional_genetics_min_evidence=_layer_cfg.get("provisional_genetics_min_evidence", 500),
            )
            if new_layer.value != state.research_layer:
                print(f"[ERIK] ★ LAYER TRANSITION: {state.research_layer} → {new_layer.value}")
                state = replace(state, research_layer=new_layer.value)

            _persist_state(state, evidence_store)

            # Check convergence: protocol must be stable AND uncertainty low
            # (prevents trivial convergence from recycled evidence)
            _unc = state.uncertainty_score
            _stable = state.protocol_stable_cycles >= 5
            _quality = _unc < 0.3
            _min_remaining = getattr(state, "min_active_steps_remaining", 0)
            if _min_remaining > 0:
                # Convergence guard: must complete minimum active steps first
                state = replace(state, min_active_steps_remaining=_min_remaining - 1)
                if _min_remaining % 50 == 0:
                    print(f"[ERIK] Convergence guard: {_min_remaining} active steps remaining")
            elif (_stable and _quality) and not state.converged:
                state = replace(state, converged=True)
                _persist_state(state, evidence_store)
                print(f"[ERIK] ★ CONVERGED at step {state.step_count}. "
                      f"Entering monitoring mode.")

            # Pause between active steps
            time.sleep(active_pause)

            # Periodic memory cleanup
            if state.step_count % 50 == 0:
                gc.collect()

        except KeyboardInterrupt:
            print(f"\n[ERIK] Interrupted at step {state.step_count}. State saved.")
            _persist_state(state, evidence_store)
            _stop_cognitive_daemons(cognitive_daemons)
            break
        except Exception as e:
            print(f"[ERIK] Error at step {state.step_count}: {e}")
            _persist_state(state, evidence_store)
            time.sleep(10)  # Back off on error, then retry


if __name__ == "__main__":
    main()
