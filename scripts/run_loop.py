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
    """Execute one deep research step — systematic evidence expansion
    that continues even after protocol convergence.

    Rotates through _DEEP_RESEARCH_QUERIES, executing one per call.
    When new evidence accumulates past the regen threshold, triggers
    protocol regeneration and re-convergence.
    """
    from research.actions import ActionType, ActionResult
    from research.loop import _execute_action, _persist_state
    from research.rewards import compute_reward

    # Pick the next query in the rotation
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

    # Execute
    result = _execute_action(action, params, state, evidence_store, llm_manager)

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

    # If enough new evidence since last protocol, trigger re-convergence
    if state.new_evidence_since_regen >= regen_threshold:
        print(f"[ERIK-MONITOR] {state.new_evidence_since_regen} new evidence items since last protocol. Re-entering active research for re-convergence.")
        state = replace(
            state,
            converged=False,
            protocol_stable_cycles=0,
        )

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
              f"converged={state.converged}")
    else:
        print("[ERIK] No saved state — starting fresh.")
        state = initial_state(subject_ref=subject_ref)

    evidence_store = EvidenceStore()
    llm_manager = DualLLMManager()

    # Initialize evidence count from DB
    try:
        db_count = evidence_store.count_by_type("EvidenceItem")
        state = replace(state, total_evidence_items=max(state.total_evidence_items, db_count))
        print(f"[ERIK] Evidence in DB: {db_count}")
    except Exception:
        pass

    # Bootstrap if no protocol exists yet
    if state.protocol_version == 0:
        print("[ERIK] No protocol — bootstrapping...")
        state = _bootstrap_initial_protocol(state, evidence_store, llm_manager)
        _persist_state(state, evidence_store)

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

            state = research_step(
                state=state,
                evidence_store=evidence_store,
                llm_manager=llm_manager,
                dry_run=False,
                regen_threshold=regen_threshold,
            )
            _persist_state(state, evidence_store)

            # Check convergence
            if state.protocol_stable_cycles >= 3 and not state.converged:
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
