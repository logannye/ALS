#!/usr/bin/env python3
"""Erik Research Monitor — real-time terminal display of loop progress.

Usage:
    conda run -n erik-core python scripts/monitor.py
    conda run -n erik-core python scripts/monitor.py --interval 2

Reads from erik_ops.research_state (written by the research loop every step)
and displays a live dashboard in the terminal.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Add scripts/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from db.pool import get_connection


# ── ANSI ──────────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
MAGENTA = "\033[35m"
WHITE = "\033[37m"
CLEAR_SCREEN = "\033[2J\033[H"


# ── DB QUERY ──────────────────────────────────────────────────────────────

def fetch_state() -> dict | None:
    """Read the latest research state from PostgreSQL."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT state_json, updated_at
                    FROM erik_ops.research_state
                    ORDER BY updated_at DESC
                    LIMIT 1
                """)
                row = cur.fetchone()
                if row:
                    state_json, updated_at = row
                    if isinstance(state_json, str):
                        state_json = json.loads(state_json)
                    state_json["_updated_at"] = str(updated_at)
                    return state_json
    except Exception as e:
        return {"_error": str(e)}
    return None


def fetch_evidence_count() -> int:
    """Count total evidence items in the DB."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM erik_core.objects WHERE type = 'EvidenceItem' AND status = 'active'")
                row = cur.fetchone()
                return row[0] if row else 0
    except Exception:
        return 0


def fetch_episode_count() -> int:
    """Count total learning episodes."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM erik_core.objects WHERE type = 'LearningEpisode'")
                row = cur.fetchone()
                return row[0] if row else 0
    except Exception:
        return 0


# ── DISPLAY ───────────────────────────────────────────────────────────────

def render(state: dict | None, db_evidence: int, episode_count: int) -> str:
    """Render the state as a terminal dashboard string."""
    lines: list[str] = []

    lines.append(f"{BOLD}{CYAN}{'=' * 70}{RESET}")
    lines.append(f"{BOLD}{CYAN}  ERIK ALS RESEARCH MONITOR{RESET}")
    lines.append(f"{BOLD}{CYAN}{'=' * 70}{RESET}")

    if state is None:
        lines.append(f"\n  {DIM}No research state found. Is the loop running?{RESET}")
        lines.append(f"  {DIM}Start with: PYTHONPATH=scripts conda run -n erik-core python -c \"...\" {RESET}")
        return "\n".join(lines)

    if "_error" in state:
        lines.append(f"\n  {RED}Error reading state: {state['_error']}{RESET}")
        return "\n".join(lines)

    updated = state.get("_updated_at", "?")
    step = state.get("step_count", 0)
    converged = state.get("converged", False)

    # ── Status bar ──
    if converged:
        status = f"{GREEN}{BOLD}CONVERGED{RESET}"
    elif step > 0:
        status = f"{YELLOW}{BOLD}RUNNING{RESET}"
    else:
        status = f"{DIM}IDLE{RESET}"

    lines.append(f"\n  Status: {status}     Last update: {DIM}{updated}{RESET}")

    # ── Progress ──
    lines.append(f"\n{BOLD}  PROGRESS{RESET}")
    lines.append(f"  {'─' * 50}")
    lines.append(f"  Step:              {BOLD}{step}{RESET}")
    lines.append(f"  Protocol version:  {BOLD}{state.get('protocol_version', 0)}{RESET}")
    lines.append(f"  Protocol ID:       {state.get('current_protocol_id', 'none')}")
    lines.append(f"  Evidence (loop):   {state.get('total_evidence_items', 0)}")
    lines.append(f"  Evidence (DB):     {BOLD}{db_evidence}{RESET}")
    lines.append(f"  Since last regen:  {state.get('new_evidence_since_regen', 0)}")
    lines.append(f"  Episodes logged:   {episode_count}")

    # ── Last action ──
    last_action = state.get("last_action", "")
    last_reward = state.get("last_reward", 0.0)
    reward_color = GREEN if last_reward > 0 else DIM
    lines.append(f"\n{BOLD}  LAST ACTION{RESET}")
    lines.append(f"  {'─' * 50}")
    lines.append(f"  Action:  {MAGENTA}{last_action}{RESET}")
    lines.append(f"  Reward:  {reward_color}{last_reward:.2f}{RESET}")

    # ── Action counts ──
    action_counts = state.get("action_counts", {})
    if action_counts:
        lines.append(f"\n{BOLD}  ACTION DISTRIBUTION{RESET}")
        lines.append(f"  {'─' * 50}")
        total_actions = sum(action_counts.values())
        for action_name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            pct = (count / total_actions * 100) if total_actions > 0 else 0
            bar_len = int(pct / 3)
            bar = "█" * bar_len
            lines.append(f"  {action_name:<30} {count:>4}  ({pct:4.1f}%) {CYAN}{bar}{RESET}")

    # ── Action values ──
    action_values = state.get("action_values", {})
    if action_values:
        lines.append(f"\n{BOLD}  ACTION VALUES (EMA){RESET}")
        lines.append(f"  {'─' * 50}")
        for action_name, value in sorted(action_values.items(), key=lambda x: -x[1]):
            val_color = GREEN if value > 1.0 else YELLOW if value > 0 else DIM
            lines.append(f"  {action_name:<30} {val_color}{value:>7.2f}{RESET}")

    # ── Causal chains ──
    chains = state.get("causal_chains", {})
    if chains:
        lines.append(f"\n{BOLD}  CAUSAL CHAINS{RESET}")
        lines.append(f"  {'─' * 50}")
        for int_id, depth in sorted(chains.items(), key=lambda x: -x[1]):
            depth_bar = "●" * depth + "○" * max(0, 5 - depth)
            depth_color = GREEN if depth >= 5 else YELLOW if depth >= 3 else RED
            lines.append(f"  {int_id:<25} {depth_color}{depth_bar} ({depth}/5){RESET}")

    # ── Hypotheses ──
    active_hyps = state.get("active_hypotheses", [])
    resolved = state.get("resolved_hypotheses", 0)
    lines.append(f"\n{BOLD}  HYPOTHESES{RESET}")
    lines.append(f"  {'─' * 50}")
    lines.append(f"  Active:    {len(active_hyps)}")
    lines.append(f"  Resolved:  {resolved}")
    for hyp_id in active_hyps[:5]:
        lines.append(f"    {DIM}→ {hyp_id}{RESET}")

    # ── Top uncertainties ──
    uncertainties = state.get("top_uncertainties", [])
    if uncertainties:
        lines.append(f"\n{BOLD}  TOP UNCERTAINTIES{RESET}")
        lines.append(f"  {'─' * 50}")
        for i, unc in enumerate(uncertainties[:5], 1):
            # Truncate long uncertainties
            display = unc[:65] + "..." if len(unc) > 65 else unc
            lines.append(f"  {i}. {YELLOW}{display}{RESET}")

    # ── Convergence ──
    stable_cycles = state.get("protocol_stable_cycles", 0)
    lines.append(f"\n{BOLD}  CONVERGENCE{RESET}")
    lines.append(f"  {'─' * 50}")
    conv_bar = "■" * stable_cycles + "□" * max(0, 3 - stable_cycles)
    conv_color = GREEN if stable_cycles >= 3 else YELLOW if stable_cycles >= 1 else DIM
    lines.append(f"  Stable cycles: {conv_color}{conv_bar} ({stable_cycles}/3){RESET}")
    if converged:
        lines.append(f"  {GREEN}{BOLD}★ PROTOCOL CONVERGED — ready for review{RESET}")

    lines.append(f"\n{DIM}  Press Ctrl+C to exit monitor{RESET}")
    lines.append(f"{CYAN}{'=' * 70}{RESET}")

    return "\n".join(lines)


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Erik Research Monitor")
    parser.add_argument("--interval", type=float, default=3.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    print(f"{BOLD}Erik Research Monitor starting (refresh every {args.interval}s)...{RESET}")

    try:
        while True:
            state = fetch_state()
            db_evidence = fetch_evidence_count()
            episode_count = fetch_episode_count()
            output = render(state, db_evidence, episode_count)
            print(CLEAR_SCREEN + output, flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n{DIM}Monitor stopped.{RESET}")


if __name__ == "__main__":
    main()
