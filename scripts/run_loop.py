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
# Monitoring mode (post-convergence)
# ---------------------------------------------------------------------------

def _monitoring_cycle(
    state: ResearchState,
    evidence_store: EvidenceStore,
    llm_manager: DualLLMManager,
) -> ResearchState:
    """One monitoring cycle — check for new data that would re-open research.

    Checks:
    1. genetics_received flag in config → triggers INTERPRET_VARIANT
    2. New evidence in DB since last check → re-enters active research
    3. Config changes (hot-reload)
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

    # Check if evidence has grown (e.g. from external ingestion)
    current_evidence = evidence_store.count_by_type("EvidenceItem")
    if current_evidence > state.total_evidence_items + 20:
        print(f"[ERIK-MONITOR] Significant new evidence detected ({current_evidence} vs {state.total_evidence_items}). Re-entering active research.")
        state = replace(
            state,
            converged=False,
            protocol_stable_cycles=0,
            total_evidence_items=current_evidence,
        )
        return state

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
    monitoring_interval = 300  # 5 minutes between monitoring checks

    print("[ERIK] Entering main loop...")

    while True:
        try:
            if state.converged:
                # Monitoring mode: slow poll for new data
                print(f"[ERIK-MONITOR] Protocol converged. Checking for new data... "
                      f"({time.strftime('%H:%M:%S')})")
                state = _monitoring_cycle(state, evidence_store, llm_manager)

                if state.converged:
                    # Still converged — sleep and check again
                    _persist_state(state, evidence_store)
                    time.sleep(monitoring_interval)
                    continue
                else:
                    # New data detected — fall through to active research
                    print("[ERIK] Re-entering active research mode.")

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
