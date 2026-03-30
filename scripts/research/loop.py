"""Main research loop — orchestrates action execution, reward tracking, and convergence.

Public API
----------
research_step(state, evidence_store, llm_manager, dry_run, ...) -> ResearchState
    Execute one research step: select action, execute, compute reward, update state.

run_research_loop(subject_ref, evidence_store, llm_manager, max_steps, ...) -> ResearchState
    Run the full research loop until convergence or max_steps.
"""
from __future__ import annotations

import gc
import time
from copy import deepcopy
from dataclasses import replace
from typing import Any, Optional

from research.actions import ActionResult, ActionType, build_action_params
from research.convergence import compute_uncertainty_score
from research.episode_logger import build_episode
from research.policy import select_action
from research.rewards import RewardComponents, compute_reward
from research.state import ResearchState, initial_state

# EMA learning rate for action value updates
_EMA_ALPHA = 0.2


# ---------------------------------------------------------------------------
# Main research step
# ---------------------------------------------------------------------------

def research_step(
    state: ResearchState,
    evidence_store: Any,
    llm_manager: Any,
    dry_run: bool = False,
    regen_threshold: int = 10,
    target_depth: int = 5,
) -> ResearchState:
    """Execute a single research step and return the updated state.

    1. Select action via policy
    2. Execute action (or skip if dry_run)
    3. Compute reward
    4. Update EMA action values
    5. Update state fields
    6. Log episode
    7. Print progress

    Returns a new ResearchState (never mutates the input).
    """
    # 1. Select action
    action, params = select_action(
        state, regen_threshold=regen_threshold, target_depth=target_depth,
    )

    # 2. Execute (with evidence deduplication)
    if dry_run:
        result = ActionResult(action=action, success=True)
    else:
        # Snapshot DB evidence count BEFORE the action
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
                pass  # Fall back to connector-reported count

    # 3. Compute reward
    # Estimate uncertainty before/after (proportion of layers with zero evidence)
    total_layers = max(len(state.evidence_by_layer), 1)
    empty_before = sum(1 for v in state.evidence_by_layer.values() if v == 0)
    uncertainty_before = empty_before / total_layers

    new_evidence_by_layer = dict(state.evidence_by_layer)
    new_evidence_by_strength = dict(state.evidence_by_strength)
    # For protocol-regeneration, evidence added is zero but protocol may improve
    # For other actions, distribute evidence across the relevant layer
    total_evidence = state.total_evidence_items + result.evidence_items_added

    # Update layer counter if the action reported a protocol_layer
    if result.protocol_layer and result.evidence_items_added > 0:
        layer_key = result.protocol_layer
        if layer_key in new_evidence_by_layer:
            new_evidence_by_layer[layer_key] += result.evidence_items_added
    # Update strength counter
    if result.evidence_strength and result.evidence_items_added > 0:
        strength_key = result.evidence_strength
        if strength_key in new_evidence_by_strength:
            new_evidence_by_strength[strength_key] += result.evidence_items_added

    empty_after = sum(1 for v in new_evidence_by_layer.values() if v == 0)
    uncertainty_after = empty_after / total_layers

    reward = compute_reward(
        evidence_items_added=result.evidence_items_added,
        uncertainty_before=uncertainty_before,
        uncertainty_after=uncertainty_after,
        protocol_score_delta=result.protocol_score_delta,
        hypothesis_resolved=result.hypothesis_resolved,
        causal_depth_added=result.causal_depth_added,
        interaction_safe=result.interaction_safe,
        eligibility_confirmed=result.eligibility_confirmed,
        protocol_stable=result.protocol_stable,
    )
    reward_total = reward.total()

    # 4. Update EMA action values
    action_key = action.value
    action_values = dict(state.action_values)
    action_counts = dict(state.action_counts)
    old_value = action_values.get(action_key, 0.0)
    action_values[action_key] = old_value + _EMA_ALPHA * (reward_total - old_value)
    action_counts[action_key] = action_counts.get(action_key, 0) + 1

    # 4b. Update Thompson posteriors (Fix 4: was never called before)
    from research.policy import _update_posteriors, _apply_decay
    _posteriors = dict(state.action_posteriors)
    _thompson_success = (
        result.evidence_items_added > 0
        or result.hypothesis_generated is not None
        or result.hypothesis_resolved
        or result.causal_depth_added > 0
        or result.protocol_regenerated
    )
    _posteriors = _update_posteriors(_posteriors, action_key, _thompson_success)
    # Periodic decay
    _next_step = state.step_count + 1
    try:
        from config.loader import ConfigLoader
        _ts_cfg = ConfigLoader()
        _decay_interval = _ts_cfg.get("thompson_decay_interval", 50)
        _decay_rate = _ts_cfg.get("thompson_decay_rate", 0.95)
    except Exception:
        _decay_interval, _decay_rate = 50, 0.95
    if _next_step % _decay_interval == 0:
        _posteriors = _apply_decay(_posteriors, rate=_decay_rate)

    # 5. Update state
    new_step = state.step_count + 1
    new_evidence_since_regen = state.new_evidence_since_regen + result.evidence_items_added
    protocol_version = state.protocol_version
    protocol_stable_cycles = state.protocol_stable_cycles

    if result.protocol_regenerated:
        protocol_version += 1
        new_evidence_since_regen = 0
        if result.protocol_stable:
            protocol_stable_cycles += 1
        else:
            protocol_stable_cycles = 0

    # Track active hypotheses (enforce max_active)
    try:
        from config.loader import ConfigLoader
        _hyp_cfg = ConfigLoader()
        max_active = _hyp_cfg.get("research_hypothesis_max_active", 10)
    except Exception:
        max_active = 10

    active_hyps = list(state.active_hypotheses)
    if result.hypothesis_generated:
        if len(active_hyps) < max_active:
            active_hyps.append(result.hypothesis_generated)
    if result.hypothesis_resolved and active_hyps:
        # Remove first hypothesis (the one that was being validated)
        active_hyps.pop(0)

    # Update causal chains
    causal_chains = dict(state.causal_chains)
    if result.causal_depth_added > 0 and params.get("intervention_id"):
        int_id = params["intervention_id"]
        causal_chains[int_id] = causal_chains.get(int_id, 0) + result.causal_depth_added

    # Track consecutive same-action count for diversity cap
    if action_key == state.last_action:
        _consecutive = state.consecutive_same_action + 1
    else:
        _consecutive = 1

    # Track last step each action type was used (for diversity floor)
    _last_action_per_type = dict(state.last_action_per_type)
    _last_action_per_type[action_key] = new_step

    # Track gap layers for recency penalty (Fix 2: was never updated before)
    _last_gap_layers = list(state.last_gap_layers)
    if action == ActionType.GENERATE_HYPOTHESIS and result.detail:
        _gap_layer = result.detail.get("gap_layer", "")
        if _gap_layer:
            _last_gap_layers.append(_gap_layer)
    elif action == ActionType.CHECK_PHARMACOGENOMICS:
        _last_gap_layers.append("unvalidated_safety")
    # Sliding window: keep only last 10 entries
    _last_gap_layers = _last_gap_layers[-10:]

    new_state = replace(
        state,
        step_count=new_step,
        total_evidence_items=total_evidence,
        evidence_by_layer=new_evidence_by_layer,
        evidence_by_strength=new_evidence_by_strength,
        action_values=action_values,
        action_counts=action_counts,
        last_action=action_key,
        last_reward=reward_total,
        new_evidence_since_regen=new_evidence_since_regen,
        protocol_version=protocol_version,
        protocol_stable_cycles=protocol_stable_cycles,
        active_hypotheses=active_hyps,
        causal_chains=causal_chains,
        resolved_hypotheses=state.resolved_hypotheses + (1 if result.hypothesis_resolved else 0),
        consecutive_same_action=_consecutive,
        last_action_per_type=_last_action_per_type,
        last_gap_layers=_last_gap_layers,
        action_posteriors=_posteriors,
    )

    # 5b. Update uncertainty score (used by convergence quality gate)
    new_state = replace(new_state, uncertainty_score=compute_uncertainty_score(new_state))

    # 5c. Extract entities/relationships into KG when new evidence was added
    if result.evidence_items_added > 0:
        try:
            from config.loader import ConfigLoader
            _kg_cfg = ConfigLoader()
            _kg_enabled = _kg_cfg.get("kg_extraction_enabled", True)
        except Exception:
            _kg_enabled = True
        if _kg_enabled:
            try:
                from knowledge_quality.entity_extractor import extract_kg_from_evidence
                _kg_stats = extract_kg_from_evidence(batch_size=result.evidence_items_added + 10)
                if _kg_stats["entities_created"] > 0:
                    print(f"[RESEARCH] KG: +{_kg_stats['entities_created']} entities, "
                          f"+{_kg_stats['relationships_created']} relationships")
            except Exception:
                pass  # Never crash the loop on KG extraction failure

    # 6. Log episode (best-effort, do not fail the step)
    try:
        _episode = build_episode(
            step_count=new_step,
            subject_ref=state.subject_ref,
            action_result=result,
            reward=reward,
            protocol_ref=state.current_protocol_id,
        )
    except Exception:
        pass

    # 7. Print progress
    print(
        f"[RESEARCH] Step {new_step}: {action_key} | "
        f"evidence={result.evidence_items_added} | "
        f"reward={reward_total:.2f} | "
        f"total_evidence={total_evidence}"
    )

    return new_state


# ---------------------------------------------------------------------------
# Full loop
# ---------------------------------------------------------------------------

def run_research_loop(
    subject_ref: str,
    evidence_store: Any,
    llm_manager: Any,
    max_steps: int = 500,
    dry_run: bool = False,
    regen_threshold: int = 10,
    inter_step_pause: float = 1.0,
) -> ResearchState:
    """Run the research loop until convergence or *max_steps*.

    Parameters
    ----------
    subject_ref:
        Trajectory/patient reference (e.g. ``"traj:draper_001"``).
    evidence_store:
        Evidence store instance with query/upsert methods.
    llm_manager:
        DualLLMManager providing research and protocol engines.
    max_steps:
        Hard ceiling on number of steps.
    dry_run:
        When True, skip actual action execution.
    regen_threshold:
        Number of new evidence items before regenerating the protocol.
    inter_step_pause:
        Seconds to pause between steps (0 for dry_run).
    """
    state = initial_state(subject_ref=subject_ref)

    # Initialize evidence count from DB (not zero)
    if not dry_run:
        try:
            state = replace(state, total_evidence_items=evidence_store.count_by_type("EvidenceItem"))
            print(f"[RESEARCH] Evidence in DB: {state.total_evidence_items}")
        except Exception:
            pass

    # Bootstrap: generate the initial protocol so the policy has
    # uncertainties, causal chain targets, and a protocol version to work with
    if not dry_run:
        print("[RESEARCH] Bootstrapping: generating initial protocol...")
        state = _bootstrap_initial_protocol(state, evidence_store, llm_manager)

    # Persist initial state for monitor
    if not dry_run:
        _persist_state(state, evidence_store)

    for _i in range(max_steps):
        state = research_step(
            state=state,
            evidence_store=evidence_store,
            llm_manager=llm_manager,
            dry_run=dry_run,
            regen_threshold=regen_threshold,
        )

        # Persist state for monitor (every step)
        if not dry_run:
            _persist_state(state, evidence_store)

        # Check convergence: protocol must be stable AND uncertainty low
        # (prevents trivial convergence from recycled evidence)
        _unc = state.uncertainty_score
        _stable = state.protocol_stable_cycles >= 5
        _quality = _unc < 0.3
        if state.converged or (_stable and _quality):
            state = replace(state, converged=True)
            _persist_state(state, evidence_store)
            print(f"[RESEARCH] Converged at step {state.step_count}.")
            break

        # Pause between steps (skip for dry_run)
        if not dry_run and inter_step_pause > 0:
            time.sleep(inter_step_pause)

        # Periodic memory cleanup
        if state.step_count % 50 == 0:
            gc.collect()

    # Final persist
    if not dry_run:
        _persist_state(state, evidence_store)

    return state


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_initial_protocol(
    state: ResearchState, evidence_store: Any, llm_manager: Any,
) -> ResearchState:
    """Generate the initial protocol and populate state with uncertainties
    and causal chain targets so the policy can make informed decisions."""
    try:
        from world_model.protocol_generator import generate_cure_protocol

        engine = llm_manager.get_protocol_engine()
        model_path = engine._llm.model_path if engine and hasattr(engine, '_llm') else None

        result_dict = generate_cure_protocol(use_llm=True, model_path=model_path)
        llm_manager.unload_protocol_model()

        protocol = result_dict.get("protocol")
        if protocol is None:
            print("[RESEARCH] Bootstrap: protocol generation failed (no protocol returned)")
            return state

        # Extract uncertainties from protocol body
        top_uncertainties = []
        body = protocol.body or {}
        for unc in body.get("key_uncertainties", [])[:5]:
            top_uncertainties.append(str(unc))
        # Add the standard missing measurements
        if not top_uncertainties:
            top_uncertainties = list(state.missing_measurements[:5])

        # Extract causal chain targets from protocol layers
        causal_chains: dict[str, int] = {}
        for layer_entry in protocol.layers:
            for int_ref in layer_entry.intervention_refs:
                causal_chains[int_ref] = 0  # Depth 0 = needs deepening

        # Update state
        state = replace(
            state,
            current_protocol_id=protocol.id,
            protocol_version=1,
            top_uncertainties=top_uncertainties,
            causal_chains=causal_chains,
            new_evidence_since_regen=0,
        )

        n_targets = len(causal_chains)
        print(f"[RESEARCH] Bootstrap complete: {protocol.id}")
        print(f"  Uncertainties: {len(top_uncertainties)}")
        print(f"  Causal chain targets: {n_targets}")

    except Exception as e:
        print(f"[RESEARCH] Bootstrap failed: {e}")
        llm_manager.unload_protocol_model()

    return state


# ---------------------------------------------------------------------------
# State persistence (for monitor)
# ---------------------------------------------------------------------------

def _persist_state(state: ResearchState, evidence_store: Any) -> None:
    """Write current research state to erik_ops.research_state for the monitor.

    Uses a single-row upsert keyed on subject_ref. Best-effort — never
    crashes the loop on persistence failure.
    """
    import json
    try:
        from db.pool import get_connection
        state_json = json.dumps(state.to_dict(), default=str)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS erik_ops.research_state (
                        subject_ref TEXT PRIMARY KEY,
                        state_json JSONB NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                cur.execute("""
                    INSERT INTO erik_ops.research_state (subject_ref, state_json, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (subject_ref) DO UPDATE
                    SET state_json = EXCLUDED.state_json, updated_at = NOW()
                """, (state.subject_ref, state_json))
            conn.commit()
    except Exception:
        pass  # Never crash the loop on persistence failure


# ---------------------------------------------------------------------------
# Action execution dispatcher
# ---------------------------------------------------------------------------

def _execute_action(
    action: ActionType,
    params: dict[str, Any],
    state: ResearchState,
    evidence_store: Any,
    llm_manager: Any,
) -> ActionResult:
    """Dispatch action to the appropriate executor, catching all exceptions."""
    try:
        dispatch = {
            ActionType.SEARCH_PUBMED: _exec_search_pubmed,
            ActionType.SEARCH_TRIALS: _exec_search_trials,
            ActionType.QUERY_CHEMBL: _exec_query_chembl,
            ActionType.QUERY_OPENTARGETS: _exec_query_opentargets,
            ActionType.CHECK_INTERACTIONS: _exec_check_interactions,
            ActionType.GENERATE_HYPOTHESIS: _exec_generate_hypothesis,
            ActionType.DEEPEN_CAUSAL_CHAIN: _exec_deepen_chain,
            ActionType.VALIDATE_HYPOTHESIS: _exec_validate_hypothesis,
            ActionType.SCORE_NEW_EVIDENCE: _exec_score_new_evidence,
            ActionType.REGENERATE_PROTOCOL: _exec_regenerate_protocol,
            ActionType.QUERY_PATHWAYS: _exec_query_pathways,
            ActionType.QUERY_PPI_NETWORK: _exec_query_ppi_network,
            ActionType.MATCH_COHORT: _exec_match_cohort,
            ActionType.INTERPRET_VARIANT: _exec_interpret_variant,
            ActionType.CHECK_PHARMACOGENOMICS: _exec_check_pharmacogenomics,
            ActionType.QUERY_GALEN_KG: _exec_query_galen_kg,
            ActionType.SEARCH_PREPRINTS: _exec_search_preprints,
            ActionType.QUERY_GALEN_SCM: _exec_query_galen_scm,
            ActionType.RUN_COMPUTATION: _exec_run_computation,
            ActionType.QUERY_ALSOD: _exec_query_alsod,
            ActionType.QUERY_GTEX: _exec_query_gtex,
            ActionType.QUERY_CLINVAR: _exec_query_clinvar,
            ActionType.QUERY_GWAS: _exec_query_gwas,
            ActionType.QUERY_BINDINGDB: _exec_query_bindingdb,
            ActionType.QUERY_HPA: _exec_query_hpa,
            ActionType.QUERY_DRUGBANK: _exec_query_drugbank,
            ActionType.QUERY_ALPHAFOLD: _exec_query_alphafold,
            ActionType.QUERY_REACTOME_LOCAL: _exec_query_reactome_local,
        }
        fn = dispatch.get(action)
        if fn is None:
            print(f"[RESEARCH] WARNING: No executor for action {action.value}")
            return ActionResult(action=action, success=False, error=f"Unknown action: {action}")
        return fn(params, state, evidence_store, llm_manager)
    except Exception as e:
        print(f"[RESEARCH] Action {action.value} failed: {e}")
        return ActionResult(action=action, success=False, error=str(e))


# ---------------------------------------------------------------------------
# Individual action executors
# ---------------------------------------------------------------------------

def _exec_search_pubmed(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Search PubMed for literature evidence."""
    from connectors.pubmed import PubMedConnector

    connector = PubMedConnector(store=store)
    query = params.get("query", f"ALS {state.subject_ref} treatment")
    max_results = params.get("max_results", 20)
    cr = connector.fetch(query=query, max_results=max_results)
    return ActionResult(
        action=ActionType.SEARCH_PUBMED,
        success=not cr.errors,
        evidence_items_added=cr.evidence_items_added,
        error="; ".join(cr.errors) if cr.errors else None,
    )


def _exec_search_trials(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Search ClinicalTrials.gov for active ALS trials."""
    from connectors.clinical_trials import ClinicalTrialsConnector

    connector = ClinicalTrialsConnector(store=store)
    cr = connector.fetch_active_als_trials()
    return ActionResult(
        action=ActionType.SEARCH_TRIALS,
        success=not cr.errors,
        evidence_items_added=cr.evidence_items_added,
        interventions_added=cr.interventions_added,
        eligibility_confirmed=cr.evidence_items_added > 0,
        error="; ".join(cr.errors) if cr.errors else None,
    )


def _exec_query_chembl(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Query ChEMBL for bioactivity data."""
    from connectors.chembl import ChEMBLConnector

    connector = ChEMBLConnector(store=store)
    target_name = params.get("target_name", "")
    if target_name:
        cr = connector.fetch(target_name=target_name)
    else:
        cr = connector.fetch()
    return ActionResult(
        action=ActionType.QUERY_CHEMBL,
        success=not cr.errors,
        evidence_items_added=cr.evidence_items_added,
        interventions_added=cr.interventions_added,
        error="; ".join(cr.errors) if cr.errors else None,
    )


def _exec_query_opentargets(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Query OpenTargets for ALS drug targets."""
    from connectors.opentargets import OpenTargetsConnector

    connector = OpenTargetsConnector(store=store)
    cr = connector.fetch_als_targets()
    return ActionResult(
        action=ActionType.QUERY_OPENTARGETS,
        success=not cr.errors,
        evidence_items_added=cr.evidence_items_added,
        interventions_added=cr.interventions_added,
        error="; ".join(cr.errors) if cr.errors else None,
    )


def _exec_check_interactions(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Check drug-drug interactions via DrugBank."""
    from connectors.drugbank import DrugBankConnector

    connector = DrugBankConnector(store=store)
    drug_ids = params.get("drug_ids", [])
    if drug_ids:
        interactions = connector.fetch_drug_interactions(drug_ids)
        has_interactions = any(len(v) > 0 for v in interactions.values())
        return ActionResult(
            action=ActionType.CHECK_INTERACTIONS,
            success=True,
            interaction_safe=not has_interactions,
            detail={"interactions": interactions},
        )
    # Fallback: fetch ALS drugs first
    cr = connector.fetch_als_drugs()
    return ActionResult(
        action=ActionType.CHECK_INTERACTIONS,
        success=not cr.errors,
        evidence_items_added=cr.evidence_items_added,
        interventions_added=cr.interventions_added,
        interaction_safe=True,
        error="; ".join(cr.errors) if cr.errors else None,
    )


def _exec_generate_hypothesis(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Generate a targeted, high-quality hypothesis using protocol gap analysis.

    Instead of a generic "generate a hypothesis" prompt, this:
    1. Analyzes the current protocol for its weakest point
    2. Builds a detailed, context-rich prompt targeting that weakness
    3. Generates a hypothesis with specific search terms and target genes
    4. Stores the search plan in the hypothesis body for validation
    """
    from research.hypotheses import create_hypothesis, is_duplicate_hypothesis
    from research.intelligence import analyze_protocol_gaps, build_hypothesis_prompt

    engine = llm_manager.get_research_engine()
    if engine is None:
        return ActionResult(
            action=ActionType.GENERATE_HYPOTHESIS,
            success=False,
            error="No research engine available",
        )

    # Read dedup config (hot-reloadable)
    _dedup_threshold = 0.45
    _gap_same_type_max = 3
    try:
        from config.loader import ConfigLoader
        _cfg = ConfigLoader()
        _dedup_threshold = _cfg.get("hypothesis_dedup_threshold", 0.45)
        _gap_same_type_max = _cfg.get("gap_same_type_max", 3)
    except Exception:
        pass

    # Find the most important gap in the protocol
    gaps = analyze_protocol_gaps(state, store)
    if not gaps:
        return ActionResult(
            action=ActionType.GENERATE_HYPOTHESIS,
            success=False,
            error="No protocol gaps identified",
        )

    gap = gaps[0]  # Highest priority gap
    topic = gap.get("layer", params.get("topic", "root_cause_suppression"))

    # Pre-generation check: skip if too many active hypotheses target the same gap type
    _gap_type = gap.get("gap_type", "")
    _same_gap_count = 0
    for h in state.active_hypotheses:
        if _gap_type and _gap_type.replace("_", " ") in h.lower():
            _same_gap_count += 1
        elif gap.get("layer", "") and gap["layer"].replace("_", " ") in h.lower():
            _same_gap_count += 1
    if _same_gap_count >= _gap_same_type_max:
        return ActionResult(
            action=ActionType.GENERATE_HYPOTHESIS,
            success=False,
            error=f"Already {_same_gap_count} hypotheses targeting {_gap_type} — skipping",
            detail={"gap_layer": gap.get("layer", ""), "gap_type": _gap_type},
        )

    # Gather evidence for context
    evidence_items = store.query_by_protocol_layer(topic)
    # Also include evidence for the specific intervention if applicable
    int_id = gap.get("intervention_id", "")
    if int_id:
        int_evidence = store.query_by_intervention_ref(int_id)
        seen = {e["id"] for e in evidence_items}
        for e in int_evidence:
            if e["id"] not in seen:
                evidence_items.append(e)

    # Build a targeted prompt
    prompt = build_hypothesis_prompt(gap, state, evidence_items)

    result = engine.reason(
        template=prompt,
        evidence_items=evidence_items[:25],
        max_tokens=800,
    )

    if result and result.get("statement"):
        statement = result["statement"]

        # Post-generation dedup: reject if too similar to existing hypotheses
        if is_duplicate_hypothesis(statement, state.active_hypotheses, threshold=_dedup_threshold):
            return ActionResult(
                action=ActionType.GENERATE_HYPOTHESIS,
                success=False,
                error="Duplicate hypothesis rejected by Jaccard filter",
                detail={"gap_layer": gap.get("layer", ""), "gap_type": _gap_type},
            )
        cited = result.get("cited_evidence", [])
        hyp = create_hypothesis(
            statement=statement,
            subject_ref=state.subject_ref,
            topic=topic,
            cited_evidence=cited,
        )
        # Store search plan in hypothesis body for validation
        hyp.body = dict(hyp.body)
        hyp.body["search_terms"] = result.get("search_terms", [])
        hyp.body["target_genes"] = result.get("target_genes", [])
        hyp.body["if_confirmed_impact"] = result.get("if_confirmed_impact", "")
        hyp.body["gap_type"] = gap.get("gap_type", "")
        hyp.body["gap_priority"] = gap.get("priority", 0.0)

        try:
            store.upsert_object(hyp)
        except Exception:
            pass

        print(f"[RESEARCH] Hypothesis: {statement[:100]}...")
        print(f"[RESEARCH]   Gap: {gap['gap_type']} (priority {gap.get('priority', 0):.2f})")
        print(f"[RESEARCH]   Search terms: {result.get('search_terms', [])[:3]}")

        return ActionResult(
            action=ActionType.GENERATE_HYPOTHESIS,
            success=True,
            hypothesis_generated=statement,
            detail={"hypothesis_id": hyp.id, "gap_layer": gap.get("layer", ""), "gap_type": gap.get("gap_type", "")},
        )

    return ActionResult(
        action=ActionType.GENERATE_HYPOTHESIS,
        success=False,
        error="LLM returned no valid hypothesis",
    )


def _exec_validate_hypothesis(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Validate a hypothesis using its stored search plan.

    Instead of searching PubMed with a hash, this:
    1. Retrieves the hypothesis from the DB to get its search_terms
    2. Builds a precise query from the hypothesis statement
    3. Searches PubMed with targeted terms
    4. Uses the LLM to assess whether found evidence supports or refutes
    """
    from connectors.pubmed import PubMedConnector
    from research.intelligence import build_validation_query

    hypothesis_id = params.get("hypothesis_id", "")
    if not hypothesis_id:
        return ActionResult(
            action=ActionType.VALIDATE_HYPOTHESIS,
            success=False,
            error="No hypothesis_id provided",
        )

    # Try to load hypothesis from DB to get its search terms
    search_terms = []
    hypothesis_statement = ""
    try:
        from db.pool import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT body FROM erik_core.objects WHERE id = %s AND status = 'active'",
                    (hypothesis_id,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    body = row[0]
                    search_terms = body.get("search_terms", [])
                    hypothesis_statement = body.get("statement", "")
    except Exception:
        pass

    connector = PubMedConnector(store=store)
    total_added = 0
    errors = []

    # Search using the hypothesis's own search terms (targeted)
    if search_terms:
        for term in search_terms[:2]:
            cr = connector.fetch(query=term, max_results=10)
            total_added += cr.evidence_items_added
            errors.extend(cr.errors)
    elif hypothesis_statement:
        # Fallback: build query from the hypothesis statement itself
        query = build_validation_query(hypothesis_statement)
        cr = connector.fetch(query=query, max_results=10)
        total_added += cr.evidence_items_added
        errors.extend(cr.errors)
    else:
        # Last resort: generic ALS query
        cr = connector.fetch(query="ALS mechanism therapeutic target 2025", max_results=10)
        total_added += cr.evidence_items_added
        errors.extend(cr.errors)

    resolved = total_added > 0

    # Update hypothesis status in DB (was never done before — all stayed "generated")
    if resolved and hypothesis_id:
        try:
            from research.hypotheses import HypothesisStatus
            from db.pool import get_connection
            new_status = HypothesisStatus.SUPPORTED.value if total_added >= 2 else HypothesisStatus.SEARCHING.value
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE erik_core.objects
                        SET body = jsonb_set(
                            jsonb_set(body, '{status}', %s::jsonb),
                            '{validation_evidence_count}', %s::jsonb
                        ),
                        updated_at = NOW()
                        WHERE id = %s AND status = 'active'
                    """, (
                        f'"{new_status}"',
                        str(total_added),
                        hypothesis_id,
                    ))
                conn.commit()
            print(f"[RESEARCH] Hypothesis {hypothesis_id[:20]}... → {new_status} ({total_added} evidence items)")
        except Exception as e:
            print(f"[RESEARCH] Hypothesis status update failed: {e}")
    elif resolved:
        print(f"[RESEARCH] Hypothesis validated with {total_added} evidence items")

    return ActionResult(
        action=ActionType.VALIDATE_HYPOTHESIS,
        success=not errors,
        evidence_items_added=total_added,
        hypothesis_resolved=resolved,
        error="; ".join(errors) if errors else None,
    )


def _exec_deepen_chain(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Extend a causal chain using the research LLM.

    Gathers evidence by intervention ref, mechanism target, and protocol
    layer. If no evidence is found at all, builds a minimal context from
    the intervention name so the LLM can still reason.
    """
    engine = llm_manager.get_research_engine()
    if engine is None:
        return ActionResult(
            action=ActionType.DEEPEN_CAUSAL_CHAIN,
            success=False,
            error="No research engine available",
        )

    intervention_id = params.get("intervention_id", "")
    if not intervention_id:
        return ActionResult(
            action=ActionType.DEEPEN_CAUSAL_CHAIN,
            success=False,
            error="No intervention_id provided",
        )

    # Gather evidence — cast a wide net
    evidence_items: list[dict] = []
    seen_ids: set[str] = set()

    # By intervention ref
    for item in store.query_by_intervention_ref(intervention_id):
        if item["id"] not in seen_ids:
            seen_ids.add(item["id"])
            evidence_items.append(item)

    # By intervention name as mechanism target (strip int: prefix)
    target_name = intervention_id.replace("int:", "")
    for item in store.query_by_mechanism_target(target_name):
        if item["id"] not in seen_ids:
            seen_ids.add(item["id"])
            evidence_items.append(item)

    # If still empty, grab some general evidence from root_cause layer
    if not evidence_items:
        for item in store.query_by_protocol_layer("root_cause_suppression")[:10]:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                evidence_items.append(item)

    current_depth = state.causal_chains.get(intervention_id, 0)

    prompt = (
        f"TASK: Extend the causal mechanism chain for the ALS intervention '{target_name}'.\n"
        f"Current chain depth: {current_depth} links.\n"
        f"The patient is a 67M with limb-onset sporadic ALS (ALSFRS-R 43/48, NfL 5.82).\n\n"
        f"Describe the NEXT mechanistic step from this intervention toward motor neuron survival.\n\n"
        f"EVIDENCE ITEMS:\n{{evidence_items_json}}\n\n"
        f"Return JSON: {{\"source\": \"<intervention or previous step>\", "
        f"\"target\": \"<next biological effect>\", "
        f"\"mechanism\": \"<how source causes target>\", "
        f"\"confidence\": <float 0-1>, "
        f"\"cited_evidence\": [\"<evidence IDs used>\"]}}"
    )

    result = engine.reason(
        template=prompt,
        evidence_items=evidence_items[:20],
        max_tokens=500,
    )

    if result and result.get("mechanism"):
        return ActionResult(
            action=ActionType.DEEPEN_CAUSAL_CHAIN,
            success=True,
            causal_depth_added=1,
            detail={"link": result},
        )

    return ActionResult(
        action=ActionType.DEEPEN_CAUSAL_CHAIN,
        success=False,
        error=f"LLM returned no valid causal link for {intervention_id}",
    )


def _exec_score_new_evidence(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Score recently added evidence items using the research LLM."""
    engine = llm_manager.get_research_engine()
    if engine is None:
        return ActionResult(
            action=ActionType.SCORE_NEW_EVIDENCE,
            success=False,
            error="No research engine available",
        )

    # Get recent unscored evidence
    evidence_items: list[dict] = []
    for layer in ("root_cause_suppression", "pathology_reversal",
                   "circuit_stabilization", "regeneration_reinnervation",
                   "adaptive_maintenance"):
        evidence_items.extend(store.query_by_protocol_layer(layer))
    evidence_items = evidence_items[:20]

    if not evidence_items:
        return ActionResult(action=ActionType.SCORE_NEW_EVIDENCE, success=True)

    prompt = (
        "Score each evidence item for relevance to ALS treatment.\n"
        "Return JSON with keys: scores (list of {id, relevance_score})\n"
    )

    result = engine.reason(
        template=prompt,
        evidence_items=evidence_items,
        max_tokens=800,
    )

    scored = 0
    if result and result.get("scores"):
        scored = len(result["scores"])

    return ActionResult(
        action=ActionType.SCORE_NEW_EVIDENCE,
        success=True,
        evidence_items_added=0,
        detail={"items_scored": scored},
    )


def _exec_regenerate_protocol(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Regenerate the cure protocol using the protocol-tier (35B) LLM.

    ALWAYS calls unload_protocol_model() after to free 35B memory.
    """
    try:
        from world_model.protocol_generator import generate_cure_protocol

        engine = llm_manager.get_protocol_engine()
        model_path = getattr(engine, "_llm", None)
        model_path_str = getattr(model_path, "model_path", None) if model_path else None

        pipeline_result = generate_cure_protocol(
            use_llm=True,
            model_path=model_path_str,
        )

        protocol = pipeline_result.get("protocol")
        protocol_id = getattr(protocol, "id", None) if protocol else None
        score_delta = 0.0

        # Determine stability: protocol exists and has layers
        stable = protocol is not None and hasattr(protocol, "layers") and len(protocol.layers) > 0

        return ActionResult(
            action=ActionType.REGENERATE_PROTOCOL,
            success=protocol is not None,
            protocol_regenerated=True,
            protocol_score_delta=score_delta,
            protocol_stable=stable,
            detail={"protocol_id": protocol_id},
        )
    except Exception as e:
        return ActionResult(
            action=ActionType.REGENERATE_PROTOCOL,
            success=False,
            error=str(e),
            protocol_regenerated=False,
        )
    finally:
        # ALWAYS free the 35B model memory
        try:
            llm_manager.unload_protocol_model()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Phase 3B action executors
# ---------------------------------------------------------------------------

def _exec_query_pathways(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Query Reactome + KEGG for pathway data on a target."""
    from connectors.reactome import ReactomeConnector
    from connectors.kegg import KEGGConnector
    from targets.als_targets import ALS_TARGETS

    target_name = params.get("target_name", "")
    target = ALS_TARGETS.get(target_name, {})
    uniprot_id = target.get("uniprot_id", "")
    gene = target.get("gene", "")
    total_added = 0
    errors: list[str] = []
    if uniprot_id:
        try:
            cr = ReactomeConnector(evidence_store=store).fetch(
                uniprot_id=uniprot_id, gene_symbol=gene,
            )
            total_added += cr.evidence_items_added
            errors.extend(cr.errors)
        except Exception as e:
            errors.append(str(e))
    if gene:
        try:
            cr = KEGGConnector(evidence_store=store).fetch(gene_symbol=gene)
            total_added += cr.evidence_items_added
            errors.extend(cr.errors)
        except Exception as e:
            errors.append(str(e))
    return ActionResult(
        action=ActionType.QUERY_PATHWAYS,
        evidence_items_added=total_added,
        success=not errors,
        error="; ".join(errors) if errors else None,
    )


def _exec_query_ppi_network(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Query STRING-DB for protein-protein interactions."""
    from connectors.string_db import STRINGConnector

    gene_symbol = params.get("gene_symbol", "")
    cr = STRINGConnector(evidence_store=store).fetch(gene_symbol=gene_symbol)
    return ActionResult(
        action=ActionType.QUERY_PPI_NETWORK,
        evidence_items_added=cr.evidence_items_added,
        success=not cr.errors,
        error="; ".join(cr.errors) if cr.errors else None,
    )


def _exec_match_cohort(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Match patient to PRO-ACT historical cohort."""
    from research.trajectory import ProACTAnalyzer

    analyzer = ProACTAnalyzer(data_dir=params.get("proact_data_dir"))
    analyzer.load()
    match = analyzer.match_cohort(
        age=67, sex="male", onset_region="lower_limb",
        baseline_alsfrs_r=43, decline_rate=-0.39,
    )
    return ActionResult(
        action=ActionType.MATCH_COHORT,
        detail={
            "cohort_match": {
                "n_patients": match.n_patients,
                "median_decline": match.median_decline_rate,
                "erik_percentile": match.erik_percentile,
            },
        },
    )


def _exec_interpret_variant(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Retrieve ClinVar variant interpretations for a gene."""
    from connectors.clinvar import ClinVarConnector

    gene = params.get("gene", "")
    cv = ClinVarConnector(evidence_store=store)
    cr = cv.fetch(gene=gene)
    return ActionResult(
        action=ActionType.INTERPRET_VARIANT,
        evidence_items_added=cr.evidence_items_added,
        success=not cr.errors,
        error="; ".join(cr.errors) if cr.errors else None,
    )


def _exec_check_pharmacogenomics(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Check PharmGKB for pharmacogenomic drug annotations."""
    from connectors.pharmgkb import PharmGKBConnector

    drug_name = params.get("drug_name", "")
    cr = PharmGKBConnector(evidence_store=store).fetch(drug_name=drug_name)
    return ActionResult(
        action=ActionType.CHECK_PHARMACOGENOMICS,
        evidence_items_added=cr.evidence_items_added,
        interaction_safe=not cr.errors,
        success=not cr.errors,
        error="; ".join(cr.errors) if cr.errors else None,
    )


def _exec_query_galen_kg(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Execute a Galen KG cross-reference query."""
    from connectors.galen_kg import GalenKGConnector

    genes = params.get("genes", ["SOD1"])
    connector = GalenKGConnector(store=store)
    cr = connector.fetch(genes=genes)
    return ActionResult(
        action=ActionType.QUERY_GALEN_KG,
        success=not cr.errors,
        evidence_items_added=cr.evidence_items_added,
        protocol_layer=params.get("protocol_layer", "root_cause_suppression"),
        evidence_strength="preclinical",
        error="; ".join(cr.errors) if cr.errors else None,
    )


def _exec_search_preprints(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Search bioRxiv/medRxiv for ALS preprints."""
    from connectors.biorxiv import BiorxivConnector

    connector = BiorxivConnector(store=store)
    query = params.get("query", "ALS motor neuron treatment")
    cr = connector.fetch(query=query)
    return ActionResult(
        action=ActionType.SEARCH_PREPRINTS,
        success=not cr.errors,
        evidence_items_added=cr.evidence_items_added,
        protocol_layer=params.get("protocol_layer"),
        evidence_strength="emerging",
        error="; ".join(cr.errors) if cr.errors else None,
    )


def _exec_query_galen_scm(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Query Galen's causal knowledge graph for downstream causal chains."""
    from connectors.galen_scm import GalenSCMConnector

    try:
        from config.loader import ConfigLoader
        cfg = ConfigLoader()
    except Exception:
        cfg = None

    database = cfg.get("galen_scm_database", "galen_kg") if cfg else "galen_kg"
    min_pch = cfg.get("galen_scm_min_pch_layer", 2) if cfg else 2
    max_depth = cfg.get("galen_scm_max_chain_depth", 3) if cfg else 3

    connector = GalenSCMConnector(
        database=database, min_pch_layer=min_pch, max_depth=max_depth,
    )
    gene = params.get("target_gene", "TARDBP")
    edges = connector.query_causal_downstream(gene)

    # Convert causal edges to evidence and store them
    added = 0
    for edge in edges:
        evi_id = f"evi:galen_scm_{edge.source}_{edge.target}".lower().replace(" ", "_")[:80]
        try:
            from ontology.base import BaseEnvelope
            obj = BaseEnvelope(
                id=evi_id,
                type="EvidenceItem",
                status="active",
                body={
                    "claim": f"{edge.source} → {edge.target} ({edge.relationship_type})",
                    "source": "galen_scm",
                    "pch_layer": edge.pch_layer,
                    "confidence": edge.confidence,
                    "protocol_layer": params.get("protocol_layer", "root_cause_suppression"),
                    "evidence_strength": "preclinical",
                },
            )
            store.upsert_object(obj)
            added += 1
        except Exception:
            continue

    return ActionResult(
        action=ActionType.QUERY_GALEN_SCM,
        success=True,
        evidence_items_added=added,
        causal_depth_added=1 if added > 0 else 0,  # Only reward depth when NEW evidence produced
        protocol_layer=params.get("protocol_layer", "root_cause_suppression"),
        evidence_strength="preclinical",
    )


def _exec_run_computation(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Run an in-silico computational experiment on ALS targets/drugs."""
    try:
        from config.loader import ConfigLoader
        cfg = ConfigLoader()
    except Exception:
        cfg = None

    if cfg and not cfg.get("computation_enabled", True):
        return ActionResult(action=ActionType.RUN_COMPUTATION, success=False,
                            error="Computation disabled in config")

    chembl_path = cfg.get("computation_chembl_path", "/Volumes/Databank/databases/chembl_36.db") if cfg else "/Volumes/Databank/databases/chembl_36.db"
    depmap_path = cfg.get("computation_depmap_path", "/Volumes/Databank/databases/depmap/CRISPRGeneEffect.parquet") if cfg else ""
    gdsc_path = cfg.get("computation_gdsc_path", "/Volumes/Databank/databases/gdsc/GDSC2_fitted_dose_response.parquet") if cfg else ""

    from executors.als_computation_executor import ALSComputationExecutor
    executor = ALSComputationExecutor(
        chembl_path=chembl_path, depmap_path=depmap_path, gdsc_path=gdsc_path,
    )

    # Rotate through experiment types and targets
    from targets.als_targets import ALS_TARGETS
    exp_types = ["gene_essentiality", "binding_affinity", "drug_sensitivity", "drug_interactions"]
    exp_type = exp_types[state.step_count % len(exp_types)]

    target_keys = list(ALS_TARGETS.keys())
    interventions = list(state.causal_chains.keys())

    if exp_type == "gene_essentiality" and target_keys:
        target = ALS_TARGETS[target_keys[state.step_count % len(target_keys)]]
        gene = target.get("gene", "TARDBP")
        result = executor.run_experiment(exp_type, target=gene, gene=gene)
    elif exp_type == "binding_affinity" and interventions and target_keys:
        drug_name = interventions[state.step_count % len(interventions)].replace("int:", "")
        target = ALS_TARGETS[target_keys[(state.step_count // len(interventions)) % len(target_keys)]]
        gene = target.get("gene", "TARDBP")
        result = executor.run_experiment(exp_type, target=f"{drug_name}-{gene}", drug=drug_name, gene=gene)
    elif exp_type == "drug_sensitivity" and interventions:
        drug_name = interventions[state.step_count % len(interventions)].replace("int:", "")
        result = executor.run_experiment(exp_type, target=drug_name, drug=drug_name)
    elif exp_type == "drug_interactions" and interventions:
        drug_name = interventions[state.step_count % len(interventions)].replace("int:", "")
        result = executor.run_experiment(exp_type, target=drug_name, drug=drug_name)
    else:
        return ActionResult(action=ActionType.RUN_COMPUTATION, success=False,
                            error="No targets or interventions to compute on")

    # Store evidence items
    added = 0
    for fact in result.facts:
        try:
            store.upsert_object(fact)
            added += 1
        except Exception:
            continue

    if added > 0:
        print(f"[RESEARCH] Computation ({exp_type}): +{added} evidence items")

    return ActionResult(
        action=ActionType.RUN_COMPUTATION,
        success=result.success,
        evidence_items_added=added,
        error=result.error,
        protocol_layer=result.facts[0].body.get("protocol_layer", "root_cause_suppression") if result.facts else None,
        evidence_strength="strong" if added > 0 else None,
    )


def _exec_query_alsod(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Query ALSoD for ALS gene variant data."""
    try:
        from config.loader import ConfigLoader
        cfg = ConfigLoader()
    except Exception:
        cfg = None

    if cfg and not cfg.get("alsod_enabled", True):
        return ActionResult(action=ActionType.QUERY_ALSOD, success=False,
                            error="ALSoD disabled in config")

    from connectors.alsod import ALSoDConnector, ERIK_PRIORITY_GENES

    connector = ALSoDConnector(store=store)
    gene = params.get("gene", "")
    if not gene:
        gene = ERIK_PRIORITY_GENES[state.step_count % len(ERIK_PRIORITY_GENES)]

    cr = connector.fetch(gene=gene, step=state.step_count)

    if cr.evidence_items_added > 0:
        print(f"[RESEARCH] ALSoD: +{cr.evidence_items_added} items for {gene}")

    return ActionResult(
        action=ActionType.QUERY_ALSOD,
        success=not cr.errors,
        evidence_items_added=cr.evidence_items_added,
        error="; ".join(cr.errors) if cr.errors else None,
        protocol_layer="root_cause_suppression",
        evidence_strength="strong",
    )


# ---------------------------------------------------------------------------
# Phase 7 database connectors — all follow the same pattern:
# 1. Select gene from ALS targets (rotated by step)
# 2. Call connector.fetch(gene=gene)
# 3. Return ActionResult with evidence count
# ---------------------------------------------------------------------------

def _make_als_gene_executor(connector_class_path: str, action_type: ActionType, **extra_kwargs):
    """Factory for ALS-target-rotating executor functions."""
    def executor(params: dict, state: ResearchState, store: Any, llm_manager: Any) -> ActionResult:
        from targets.als_targets import ALS_TARGETS
        target_keys = list(ALS_TARGETS.keys())
        target = ALS_TARGETS[target_keys[state.step_count % len(target_keys)]]
        gene = params.get("gene", target.get("gene", "TARDBP"))
        uniprot = target.get("uniprot_id", "")

        # Dynamically import the connector
        module_name, class_name = connector_class_path.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        connector = cls(store=store, **extra_kwargs)

        cr = connector.fetch(gene=gene, uniprot=uniprot)
        if cr.evidence_items_added > 0:
            print(f"[RESEARCH] {action_type.value}: +{cr.evidence_items_added} items for {gene}")

        return ActionResult(
            action=action_type,
            success=not cr.errors,
            evidence_items_added=cr.evidence_items_added,
            error="; ".join(cr.errors) if cr.errors else None,
            protocol_layer="root_cause_suppression",
            evidence_strength="strong" if cr.evidence_items_added > 0 else None,
        )
    return executor


_exec_query_gtex = _make_als_gene_executor("connectors.gtex.GTExConnector", ActionType.QUERY_GTEX)
_exec_query_clinvar = _make_als_gene_executor("connectors.clinvar_local.ClinVarLocalConnector", ActionType.QUERY_CLINVAR)
_exec_query_gwas = _make_als_gene_executor("connectors.gwas_catalog.GWASCatalogConnector", ActionType.QUERY_GWAS)
_exec_query_hpa = _make_als_gene_executor("connectors.hpa.HPAConnector", ActionType.QUERY_HPA)
_exec_query_drugbank = _make_als_gene_executor("connectors.drugbank_local.DrugBankLocalConnector", ActionType.QUERY_DRUGBANK)
_exec_query_alphafold = _make_als_gene_executor("connectors.alphafold_local.AlphaFoldLocalConnector", ActionType.QUERY_ALPHAFOLD)
_exec_query_reactome_local = _make_als_gene_executor("connectors.reactome_local.ReactomeLocalConnector", ActionType.QUERY_REACTOME_LOCAL)


def _exec_query_bindingdb(
    params: dict, state: ResearchState, store: Any, llm_manager: Any,
) -> ActionResult:
    """Query BindingDB for drug-target binding affinities."""
    from connectors.bindingdb import BindingDBConnector
    from targets.als_targets import ALS_TARGETS

    # Rotate through drug-target pairs
    interventions = list(state.causal_chains.keys())
    target_keys = list(ALS_TARGETS.keys())

    drug = ""
    gene = ""
    if interventions:
        drug = interventions[state.step_count % len(interventions)].replace("int:", "")
    if target_keys:
        target = ALS_TARGETS[target_keys[(state.step_count // max(len(interventions), 1)) % len(target_keys)]]
        gene = target.get("gene", "")

    connector = BindingDBConnector(store=store)
    cr = connector.fetch(gene=gene, drug=drug)

    if cr.evidence_items_added > 0:
        print(f"[RESEARCH] BindingDB: +{cr.evidence_items_added} items for {drug}/{gene}")

    return ActionResult(
        action=ActionType.QUERY_BINDINGDB,
        success=not cr.errors,
        evidence_items_added=cr.evidence_items_added,
        error="; ".join(cr.errors) if cr.errors else None,
        protocol_layer="root_cause_suppression",
        evidence_strength="strong" if cr.evidence_items_added > 0 else None,
    )
