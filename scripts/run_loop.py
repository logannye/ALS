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


def _deep_research_step(
    state: ResearchState,
    evidence_store: EvidenceStore,
    llm_manager: DualLLMManager,
) -> ResearchState:
    """Execute one deep research step — intelligent evidence expansion
    that continues even after protocol convergence.

    Alternates between:
    - Gap-driven research: analyze protocol gaps, search for what matters most
    - Systematic rotation: cycle through hardcoded queries for broad coverage

    Every 3rd step is gap-driven (uses protocol gap analysis to pick the
    most impactful query). Other steps use the hardcoded rotation.
    """
    from research.actions import ActionType, ActionResult
    from research.loop import _execute_action, _persist_state
    from research.rewards import compute_reward

    # Every 3rd step: gap-driven research
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
        # Systematic rotation through hardcoded queries
        deep_step = state.step_count % len(_DEEP_RESEARCH_QUERIES)
        action_name, params = _DEEP_RESEARCH_QUERIES[deep_step]

    # Map string to ActionType
    action_map = {
        "search_pubmed": ActionType.SEARCH_PUBMED,
        "search_trials": ActionType.SEARCH_TRIALS,
        "query_ppi_network": ActionType.QUERY_PPI_NETWORK,
        "query_pathways": ActionType.QUERY_PATHWAYS,
        "check_pharmacogenomics": ActionType.CHECK_PHARMACOGENOMICS,
    }
    action = action_map.get(action_name, ActionType.SEARCH_PUBMED)
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

    # Compute simple reward
    reward = compute_reward(
        evidence_items_added=result.evidence_items_added,
        uncertainty_before=0.3,
        uncertainty_after=0.3,
        protocol_score_delta=0.0,
        hypothesis_resolved=False,
        causal_depth_added=0,
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

    state = replace(
        state,
        step_count=state.step_count + 1,
        total_evidence_items=new_evidence,
        new_evidence_since_regen=new_since_regen,
        last_action=f"deep:{action.value}",
        last_reward=reward.total(),
        action_counts=action_counts,
    )

    print(
        f"[ERIK-DEEP] Step {state.step_count}: {action.value} | "
        f"evidence={result.evidence_items_added} | "
        f"total={new_evidence} | "
        f"since_regen={new_since_regen}"
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
        print(f"[ERIK-MONITOR] ★ LAYER TRANSITION: {state.research_layer} → {new_layer.value}")
        print(f"[ERIK-MONITOR] Layer transition invalidates convergence — re-entering active research")
        state = replace(
            state,
            research_layer=new_layer.value,
            converged=False,
            protocol_stable_cycles=0,
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
            if (_stable and _quality) and not state.converged:
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
            break
        except Exception as e:
            print(f"[ERIK] Error at step {state.step_count}: {e}")
            _persist_state(state, evidence_store)
            time.sleep(10)  # Back off on error, then retry


if __name__ == "__main__":
    main()
