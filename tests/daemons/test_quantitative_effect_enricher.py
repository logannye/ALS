"""End-to-end test for QuantitativeEffectEnricher.

Validates the full flow:
  1. Seed an scm_edge with effect_mean=NULL and compound/gene als_roles.
  2. Run one enricher cycle.
  3. Assert effect_mean is populated (via SCMWriter's effect_updated path).

Gated behind ChEMBL + UniProt availability in addition to PG.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pytest

from daemons.quantitative_effect_enricher import QuantitativeEffectEnricher, _entity_id_tail
from world_model.chembl_effect_mapper import DEFAULT_CHEMBL_DB, DEFAULT_UNIPROT_TSV
from world_model.scm_writer import (
    SCMWriterLockContention,
    get_scm_writer,
    shutdown_scm_writer,
)


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
chembl = pytest.mark.skipif(
    not Path(DEFAULT_CHEMBL_DB).exists(),
    reason="chembl db not mounted",
)
uniprot = pytest.mark.skipif(
    not Path(DEFAULT_UNIPROT_TSV).exists(),
    reason="uniprot tsv not mounted",
)


# ─── Pure-function tests ─────────────────────────────────────────────────────


def test_entity_id_tail_parses_colon_suffix():
    assert _entity_id_tail('compound:riluzole') == 'riluzole'
    assert _entity_id_tail('gene:SOD1') == 'SOD1'
    assert _entity_id_tail('compound:drug:riluzole') == 'riluzole'     # rsplit


def test_entity_id_tail_handles_missing_prefix():
    assert _entity_id_tail('riluzole') == 'riluzole'


# ─── Integration tests ──────────────────────────────────────────────────────


@pytest.fixture
def conn():
    import psycopg
    user = os.environ.get("USER", "logannye")
    c = psycopg.connect(f"dbname=erik_kg user={user}")
    yield c
    c.close()


@pytest.fixture
def running_writer():
    import world_model.scm_writer as m
    m._singleton = None
    writer = get_scm_writer()
    try:
        writer.start()
    except SCMWriterLockContention:
        pytest.skip("SCMWriter lock contended")
    yield writer
    shutdown_scm_writer(timeout=5.0)


def _seed_compound_gene_edge(
    conn,
    compound_name: str,
    gene_symbol: str,
) -> int:
    """Seed an scm_edge with NULL effect + compound→gene als_roles.

    entity_id uses the ``<prefix>:<name>`` convention so
    ``_entity_id_tail`` extracts the name.
    """
    tag = uuid.uuid4().hex[:6]
    src_entity = f"test:compound:{compound_name}_{tag}"
    tgt_entity = f"test:gene:{gene_symbol}"
    # Note: target entity_id has no tag suffix so _entity_id_tail gives
    # back 'SOD1' verbatim for the gene resolver.
    # But we need uniqueness → embed tag differently:
    tgt_entity = f"test:gene_{tag}:{gene_symbol}"
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO erik_ops.scm_nodes(entity_id, node_class, als_role)
                       VALUES (%s, 'treatment', 'compound') RETURNING id""", (src_entity,))
        src = cur.fetchone()[0]
        cur.execute("""INSERT INTO erik_ops.scm_nodes(entity_id, node_class, als_role)
                       VALUES (%s, 'covariate', 'gene') RETURNING id""", (tgt_entity,))
        tgt = cur.fetchone()[0]
        cur.execute("""
            INSERT INTO erik_ops.scm_edges
                (source_node_id, target_node_id, edge_kind,
                 identification_algorithm, identification_confidence, status)
            VALUES (%s, %s, 'causal', 'rct', 0.9, 'active')
            RETURNING id
        """, (src, tgt))
        eid = int(cur.fetchone()[0])
        conn.commit()
    return eid


def _wait_for_effect(conn, edge_id: int, timeout: float = 5.0) -> tuple | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT effect_mean, effect_std, effect_scale FROM erik_ops.scm_edges WHERE id = %s",
                (edge_id,),
            )
            r = cur.fetchone()
        if r and r[0] is not None:
            return r
        time.sleep(0.1)
    return None


@pg
@chembl
@uniprot
def test_enricher_populates_effect_for_known_pair(conn, running_writer):
    """End-to-end: seed riluzole→SLC6A2, run enricher, assert effect_mean is set."""
    # Before seeding the test edge, reset the entity-tail parser's expectation:
    # _entity_id_tail returns the part after the last ':'. So entity ID
    # 'test:compound:riluzole_XXXX' gives 'riluzole_XXXX' — which won't
    # resolve. The enricher uses the tail directly, so we must encode the
    # name as the last colon-segment.
    tag = uuid.uuid4().hex[:6]
    src_entity = f"test_{tag}:riluzole"
    tgt_entity = f"test_{tag}:SLC6A2"
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO erik_ops.scm_nodes(entity_id, node_class, als_role)
                       VALUES (%s, 'treatment', 'compound') RETURNING id""", (src_entity,))
        src = cur.fetchone()[0]
        cur.execute("""INSERT INTO erik_ops.scm_nodes(entity_id, node_class, als_role)
                       VALUES (%s, 'covariate', 'gene') RETURNING id""", (tgt_entity,))
        tgt = cur.fetchone()[0]
        cur.execute("""
            INSERT INTO erik_ops.scm_edges
                (source_node_id, target_node_id, edge_kind,
                 identification_algorithm, identification_confidence, status)
            VALUES (%s, %s, 'causal', 'rct', 0.9, 'active')
            RETURNING id
        """, (src, tgt))
        edge_id = int(cur.fetchone()[0])
        conn.commit()

    enricher = QuantitativeEffectEnricher()
    stats = enricher.run_once()
    assert stats['considered'] >= 1
    assert stats['resolved'] >= 1
    assert stats['updated'] >= 1

    row = _wait_for_effect(conn, edge_id, timeout=10.0)
    assert row is not None
    assert row[0] is not None  # effect_mean populated
    assert row[1] is not None and float(row[1]) > 0  # nonzero std invariant
    assert row[2] in ('ic50_log_nm', 'ec50_log_nm')


@pg
@chembl
@uniprot
def test_enricher_skips_non_compound_gene_pairs(conn, running_writer):
    """compound→pathway or gene→compound edges must not be touched."""
    tag = uuid.uuid4().hex[:6]
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO erik_ops.scm_nodes(entity_id, node_class, als_role)
                       VALUES (%s, 'treatment', 'compound') RETURNING id""",
                    (f"t_{tag}:riluzole",))
        src = cur.fetchone()[0]
        cur.execute("""INSERT INTO erik_ops.scm_nodes(entity_id, node_class, als_role)
                       VALUES (%s, 'covariate', 'pathway') RETURNING id""",
                    (f"t_{tag}:autophagy",))
        tgt = cur.fetchone()[0]
        cur.execute("""
            INSERT INTO erik_ops.scm_edges
                (source_node_id, target_node_id, edge_kind,
                 identification_algorithm, identification_confidence, status)
            VALUES (%s, %s, 'causal', 'rct', 0.9, 'active')
            RETURNING id
        """, (src, tgt))
        edge_id = int(cur.fetchone()[0])
        conn.commit()

    enricher = QuantitativeEffectEnricher()
    stats = enricher.run_once()
    # This edge should not be fetched because target als_role='pathway'.
    # The candidate-fetch SQL filters by (compound, gene), so 'considered'
    # only counts rows the daemon actually saw.
    # Assert the edge is still NULL.
    with conn.cursor() as cur:
        cur.execute("SELECT effect_mean FROM erik_ops.scm_edges WHERE id = %s", (edge_id,))
        assert cur.fetchone()[0] is None


@pg
@chembl
@uniprot
def test_enricher_skips_unresolvable_names(conn, running_writer):
    """A compound/gene pair that doesn't resolve must not write spurious data."""
    tag = uuid.uuid4().hex[:6]
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO erik_ops.scm_nodes(entity_id, node_class, als_role)
                       VALUES (%s, 'treatment', 'compound') RETURNING id""",
                    (f"t_{tag}:notarealdrug",))
        src = cur.fetchone()[0]
        cur.execute("""INSERT INTO erik_ops.scm_nodes(entity_id, node_class, als_role)
                       VALUES (%s, 'covariate', 'gene') RETURNING id""",
                    (f"t_{tag}:NOTAGENE",))
        tgt = cur.fetchone()[0]
        cur.execute("""
            INSERT INTO erik_ops.scm_edges
                (source_node_id, target_node_id, edge_kind,
                 identification_algorithm, identification_confidence, status)
            VALUES (%s, %s, 'causal', 'rct', 0.9, 'active')
            RETURNING id
        """, (src, tgt))
        edge_id = int(cur.fetchone()[0])
        conn.commit()

    enricher = QuantitativeEffectEnricher()
    stats = enricher.run_once()
    assert stats['skipped_no_mapping'] >= 1
    assert stats['updated'] == 0

    with conn.cursor() as cur:
        cur.execute("SELECT effect_mean FROM erik_ops.scm_edges WHERE id = %s", (edge_id,))
        assert cur.fetchone()[0] is None
