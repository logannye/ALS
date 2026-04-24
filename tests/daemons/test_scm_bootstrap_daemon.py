"""Integration tests for SCMBootstrapDaemon.

These tests seed real rows in ``erik_core.relationships`` + ``erik_core.entities``,
run one ``process_one_batch`` call, and assert the expected mutations in
``erik_ops.scm_*`` tables. Skipped if PG is unreachable.
"""
from __future__ import annotations

import os
import uuid

import pytest

from daemons.scm_bootstrap_daemon import (
    SCMBootstrapDaemon,
    _EVIDENCE_TYPE_DENYLIST,
    _EVIDENCE_TYPE_TO_ALGORITHM,
    _infer_als_role,
)
from world_model.scm_writer import (
    SCMWriterLockContention,
    get_scm_writer,
    shutdown_scm_writer,
)


# ─── Pure-function tests (no DB) ─────────────────────────────────────────────


def test_denylist_excludes_inferred_types():
    assert 'inferred_from_evidence' in _EVIDENCE_TYPE_DENYLIST
    assert 'inferred_chain' in _EVIDENCE_TYPE_DENYLIST
    # Legit types must not be denied.
    assert 'clinical_trial' not in _EVIDENCE_TYPE_DENYLIST
    assert 'pubmed' not in _EVIDENCE_TYPE_DENYLIST


def test_algorithm_map_covers_als_gold_standards():
    """Bootstrap must be able to promote rows from the ALS gold-standard sources."""
    for ev in ['clinical_trial', 'ipsc_motor_neuron', 'als_mouse_model', 'galen_scm']:
        assert ev in _EVIDENCE_TYPE_TO_ALGORITHM


def test_infer_als_role_matches_compound_names():
    assert _infer_als_role("riluzole", None) == 'compound'
    assert _infer_als_role("TARDBP", None) == 'gene'
    assert _infer_als_role("protein aggregation", None) == 'mechanism'
    assert _infer_als_role("ALSFRS-R", None) == 'clinical_endpoint'


def test_infer_als_role_uses_explicit_entity_type_first():
    """explicit entity_type beats keyword inference."""
    assert _infer_als_role("TARDBP", "drug") == 'compound'
    assert _infer_als_role("riluzole", "pathway") == 'pathway'


def test_infer_als_role_falls_back_to_other_when_no_match():
    assert _infer_als_role("unknown_entity", None) == 'other'


# ─── PG-gated integration tests ──────────────────────────────────────────────


def _can_connect() -> bool:
    import psycopg
    user = os.environ.get("USER", "logannye")
    try:
        c = psycopg.connect(f"dbname=erik_kg user={user}", connect_timeout=3)
        c.close()
        return True
    except Exception:
        return False


pg = pytest.mark.skipif(not _can_connect(), reason="erik_kg PG not reachable")


@pytest.fixture
def conn():
    import psycopg
    user = os.environ.get("USER", "logannye")
    c = psycopg.connect(f"dbname=erik_kg user={user}")
    yield c
    c.close()


@pytest.fixture
def running_scm_writer():
    """Start a fresh SCMWriter, yield it, shut down at end."""
    # Reset module-level singleton so parallel tests can't share a lock.
    import world_model.scm_writer as m
    m._singleton = None
    writer = get_scm_writer()
    try:
        writer.start()
    except SCMWriterLockContention:
        pytest.skip("SCMWriter lock contended — another process holds it")
    yield writer
    shutdown_scm_writer(timeout=5.0)


def _seed_entity(conn, entity_id: str, entity_type: str, name: str,
                 druggability: float | None = None) -> None:
    import json
    with conn.cursor() as cur:
        props = {}
        if druggability is not None:
            props['druggability_prior'] = druggability
        cur.execute("""
            INSERT INTO erik_core.entities(id, entity_type, name, properties, confidence)
            VALUES (%s, %s, %s, %s::jsonb, 0.9)
            ON CONFLICT (id) DO NOTHING
        """, (entity_id, entity_type, name, json.dumps(props)))
        conn.commit()


def _seed_relationship(conn, rel_id: str, src: str, tgt: str,
                       evidence_type: str, confidence: float,
                       sources: list[str] | None = None) -> None:
    import json
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO erik_core.relationships
                (id, source_id, target_id, relationship_type,
                 confidence, sources, evidence_type)
            VALUES (%s, %s, %s, 'causal', %s, %s::jsonb, %s)
            ON CONFLICT (id) DO NOTHING
        """, (rel_id, src, tgt, confidence, json.dumps(sources or []), evidence_type))
        conn.commit()


@pg
def test_bootstrap_promotes_high_confidence_clinical_trial_to_scm_edge(
    conn, running_scm_writer,
):
    """End-to-end: clinical_trial evidence + high confidence → scm_edge + write_log row."""
    import time
    tag = uuid.uuid4().hex[:8]
    src = f"test:compound:riluzole_{tag}"
    tgt = f"test:mechanism:glutamate_excitotoxicity_{tag}"
    rel = f"test:rel:{tag}"
    _seed_entity(conn, src, 'drug', 'riluzole', druggability=0.9)
    _seed_entity(conn, tgt, 'mechanism', 'glutamate excitotoxicity', druggability=0.1)
    _seed_relationship(conn, rel, src, tgt, 'clinical_trial', 0.92,
                       sources=['pubmed:111', 'pubmed:222'])

    daemon = SCMBootstrapDaemon()
    # Process at least once — may require multiple picks to land on our evidence type.
    did_work = False
    for _ in range(5):
        if daemon.process_one_batch():
            did_work = True
            break
    assert did_work

    # Give the writer thread a moment to drain.
    for _ in range(20):
        with conn.cursor() as cur:
            cur.execute("SELECT scm_edge_id FROM erik_core.relationships WHERE id=%s", (rel,))
            row = cur.fetchone()
        if row and row[0] is not None:
            break
        time.sleep(0.1)

    with conn.cursor() as cur:
        cur.execute("SELECT scm_edge_id FROM erik_core.relationships WHERE id=%s", (rel,))
        scm_edge_id = cur.fetchone()[0]
        assert scm_edge_id is not None

        cur.execute("""
            SELECT identification_algorithm, is_intervention_candidate
              FROM erik_ops.scm_edges WHERE id = %s
        """, (scm_edge_id,))
        algo, is_candidate = cur.fetchone()
        assert algo == 'rct'
        assert is_candidate is True   # druggable compound → mechanism target

        cur.execute("""
            SELECT COUNT(*) FROM erik_ops.scm_write_log
             WHERE operation = 'edge_created' AND target_id = %s
        """, (scm_edge_id,))
        assert cur.fetchone()[0] >= 1


@pg
def test_bootstrap_skips_denylist_evidence_types(conn, running_scm_writer):
    """Inferred-from-evidence rows must never auto-promote."""
    tag = uuid.uuid4().hex[:8]
    src = f"test:compound:x_{tag}"
    tgt = f"test:mechanism:y_{tag}"
    rel = f"test:rel_denied:{tag}"
    _seed_entity(conn, src, 'drug', 'x')
    _seed_entity(conn, tgt, 'mechanism', 'y')
    _seed_relationship(conn, rel, src, tgt, 'inferred_from_evidence', 0.95)

    daemon = SCMBootstrapDaemon()
    # Run several picks — 'inferred_from_evidence' must never be selected.
    for _ in range(5):
        daemon.process_one_batch()

    with conn.cursor() as cur:
        cur.execute("SELECT scm_edge_id FROM erik_core.relationships WHERE id=%s", (rel,))
        assert cur.fetchone()[0] is None


@pg
def test_bootstrap_skips_low_confidence_rows(conn, running_scm_writer):
    """Rows below scm_bootstrap_min_confidence must not promote."""
    tag = uuid.uuid4().hex[:8]
    src = f"test:compound:lc_{tag}"
    tgt = f"test:mechanism:lc_{tag}"
    rel = f"test:rel_lowconf:{tag}"
    _seed_entity(conn, src, 'drug', 'lowconf')
    _seed_entity(conn, tgt, 'mechanism', 'lowconf')
    _seed_relationship(conn, rel, src, tgt, 'clinical_trial', 0.50)  # below 0.85 floor

    daemon = SCMBootstrapDaemon()
    for _ in range(5):
        daemon.process_one_batch()
    with conn.cursor() as cur:
        cur.execute("SELECT scm_edge_id FROM erik_core.relationships WHERE id=%s", (rel,))
        assert cur.fetchone()[0] is None


@pg
def test_bootstrap_progress_persists_across_batches(conn, running_scm_writer):
    """After one batch for an evidence type, progress row exists with status != 'pending'."""
    tag = uuid.uuid4().hex[:8]
    src = f"test:compound:p_{tag}"
    tgt = f"test:mechanism:p_{tag}"
    _seed_entity(conn, src, 'drug', 'progress-drug', druggability=0.8)
    _seed_entity(conn, tgt, 'mechanism', 'progress-mech')
    _seed_relationship(conn, f"test:rel_p:{tag}", src, tgt, 'clinical_trial', 0.90,
                       sources=['pubmed:1'])

    daemon = SCMBootstrapDaemon()
    for _ in range(5):
        if daemon.process_one_batch():
            break

    with conn.cursor() as cur:
        cur.execute("""
            SELECT status, processed_count FROM erik_ops.scm_bootstrap_progress
             WHERE evidence_type = 'clinical_trial'
        """)
        row = cur.fetchone()
    assert row is not None
    assert row[0] in ('running', 'complete')
    assert row[1] >= 1
