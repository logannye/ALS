"""Seed druggability_prior on canonical ALS drugs + targets.

Prerequisite for SCMBootstrapDaemon's is_intervention_candidate logic
to fire. Without this, bootstrap promotes edges but never flags any
as intervention candidates, and CompoundDaemon / CPTS have nothing to
simulate.

Curated list reflects ALS-field consensus as of 2026-04:
  * Approved / in-trial drugs: riluzole, edaravone, tofersen, masitinib,
    ibudilast, rapamycin, phenylbutyrate, sodesta, biib078, pridopidine.
  * Targets Erik's protocol watches: TARDBP, SOD1, FUS, C9orf72, SIGMAR1,
    MTOR, SLC1A2 (EAAT2), CSF1R, STMN2, UNC13A.

Druggability priors:
  * Approved drugs: 0.95 (proven drug-like)
  * In-trial drugs: 0.85 (safety + PK/PD validated)
  * Druggable targets: 0.70 (clear binding pocket, known modulators)
  * Hard targets: 0.40 (RNA-binding proteins, aggregation-prone —
    harder to drug directly but not impossible).

Idempotent: upserts ``properties->'druggability_prior'`` without
disturbing other property keys. Safe to re-run any time.

Usage:
    PYTHONPATH=scripts python -m ops.seed_druggability
    PYTHONPATH=scripts python -m ops.seed_druggability --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable

from db.pool import get_connection


# -----------------------------------------------------------------------------
# Curated priors.
# -----------------------------------------------------------------------------

_DRUGS_APPROVED_OR_TRIAL: dict[str, float] = {
    # Approved for ALS
    'riluzole': 0.95,
    'edaravone': 0.95,
    'tofersen': 0.95,
    # Late-stage clinical / approved abroad
    'masitinib': 0.85,
    'ibudilast': 0.85,
    'sodium phenylbutyrate': 0.85,
    'phenylbutyrate': 0.85,
    'taurursodiol': 0.85,
    'relyvrio': 0.85,
    # Repurposing candidates / active ALS trials
    'rapamycin': 0.80,
    'sirolimus': 0.80,
    'pridopidine': 0.85,
    'sodesta': 0.80,
    'biib078': 0.75,
    # Mechanism-of-interest anchors for Erik's protocol
    'rilmenidine': 0.70,
    'lithium carbonate': 0.70,
    'deferiprone': 0.70,
}

_TARGETS_DRUGGABLE: dict[str, float] = {
    # Validated binding pockets / approved-drug targets
    'SIGMAR1': 0.80,        # pridopidine, riluzole (secondary)
    'MTOR': 0.85,           # rapamycin class
    'SLC1A2': 0.70,         # EAAT2 — glutamate transporter
    'CSF1R': 0.80,          # kinase; masitinib-adjacent
    # Clinical relevance but harder to drug
    'TARDBP': 0.45,         # TDP-43 — RNA-binding, intrinsically disordered
    'SOD1': 0.50,           # tofersen (ASO) proves druggability via RNA
    'FUS': 0.45,            # RNA-binding, similar challenge to TDP-43
    'C9orf72': 0.50,        # repeat expansion — ASO-accessible
    'STMN2': 0.55,          # cryptic exon target; ASO-accessible
    'UNC13A': 0.55,
    # Other ALS-context kinases
    'GSK3B': 0.70,
    'TBK1': 0.65,
    'NEK1': 0.55,
    'VCP': 0.60,
    'ATXN2': 0.50,
}


def _name_variants(canonical: str) -> list[str]:
    """Return plausible entity_id / name forms for a canonical drug/target name.

    Erik's entity IDs follow the convention ``f"{type}:{name}".lower().replace(" ", "_")``
    so we try both the raw name and the space→underscore variant.
    """
    base = canonical.strip()
    out = [base, base.lower(), base.upper()]
    out.append(base.lower().replace(' ', '_'))
    out.append(base.upper().replace(' ', '_'))
    return list(dict.fromkeys(out))  # dedupe, preserve order


def _iter_seed_rows() -> Iterable[tuple[str, str, float]]:
    """Yield (kind, canonical_name, druggability) tuples.

    kind ∈ {'drug','target'} — used only for logging.
    """
    for name, prior in _DRUGS_APPROVED_OR_TRIAL.items():
        yield ('drug', name, prior)
    for name, prior in _TARGETS_DRUGGABLE.items():
        yield ('target', name, prior)


def seed(dry_run: bool = False, verbose: bool = True) -> dict[str, int]:
    """Upsert druggability_prior onto matching entities.

    Matching strategy: for each canonical name, find entities whose
    lowercased name matches *any* of the name variants. This handles the
    real diversity of how entities get registered (drug:riluzole,
    compound:riluzole, protein:SOD1, etc.) without the script needing to
    know which entity_type prefix a given curator used.

    Returns stats: {'considered', 'entities_matched', 'entities_updated', 'skipped_dry_run'}.
    """
    stats = {'considered': 0, 'entities_matched': 0, 'entities_updated': 0,
             'skipped_dry_run': 0}
    with get_connection() as conn:
        with conn.cursor() as cur:
            for kind, canonical, prior in _iter_seed_rows():
                stats['considered'] += 1
                variants = _name_variants(canonical)
                # Match against either entities.name (CITEXT) or the tail
                # segment of entities.id (split on ':').
                cur.execute("""
                    SELECT id, name, entity_type, properties
                      FROM erik_core.entities
                     WHERE LOWER(name::text) = ANY(%s::text[])
                        OR LOWER(SPLIT_PART(id, ':', -1)) = ANY(%s::text[])
                """, ([v.lower() for v in variants], [v.lower() for v in variants]))
                rows = cur.fetchall()
                if not rows:
                    continue
                stats['entities_matched'] += len(rows)
                for entity_id, name, entity_type, properties in rows:
                    props = dict(properties) if isinstance(properties, dict) else {}
                    existing = props.get('druggability_prior')
                    if existing is not None and float(existing) >= prior:
                        # Already seeded with at-least-as-high a value — skip.
                        continue
                    props['druggability_prior'] = prior
                    if verbose:
                        print(f"  [{kind:6}] {entity_id:40}  {canonical:25}  "
                              f"prior {existing} → {prior}")
                    if dry_run:
                        stats['skipped_dry_run'] += 1
                        continue
                    cur.execute("""
                        UPDATE erik_core.entities
                           SET properties = %s::jsonb,
                               updated_at = NOW()
                         WHERE id = %s
                    """, (json.dumps(props), entity_id))
                    stats['entities_updated'] += 1
            if not dry_run:
                conn.commit()
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog='seed_druggability')
    p.add_argument('--dry-run', action='store_true',
                   help='Print proposed updates without writing.')
    p.add_argument('--quiet', action='store_true', help='Suppress per-entity output.')
    args = p.parse_args(argv)
    stats = seed(dry_run=args.dry_run, verbose=not args.quiet)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
