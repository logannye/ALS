"""Propagation rollback CLI.

Restore before-snapshots for previously-applied R4 propagation events.
This is the operator escape hatch when a refutation was itself refuted
(e.g. a meta-analysis was retracted) and Erik's protocol should be
un-contested without waiting for the next live write-log event to
propose the reversal.

Usage::

    # Show recently-applied propagation events:
    PYTHONPATH=scripts python -m ops.propagation_rollback list --limit 20

    # Show one event's full diff (before-snapshot + affected objects):
    PYTHONPATH=scripts python -m ops.propagation_rollback show <event_id>

    # Roll back a single event:
    PYTHONPATH=scripts python -m ops.propagation_rollback rollback <event_id>

    # Roll back everything triggered by a specific scm_write_log event:
    PYTHONPATH=scripts python -m ops.propagation_rollback rollback-by-source <write_log_id>

Safety:
  * Confirms destructive actions via a stdin prompt unless ``--yes`` is set.
  * Never deletes propagation_events rows (the schema trigger forbids it
    anyway); rollback writes a new event linked via rollback_lineage_id
    so the audit trail is preserved.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from db.pool import get_connection
from world_model.propagation_rules import rollback_event


def _print_event(row: tuple) -> None:
    (ev_id, rule_kind, status, refuted_edge, src_log_id,
     affected_ids, affected_types, reason, proposed_at, applied_at) = row
    print(f"  #{ev_id}  [{status:>11}]  rule={rule_kind}")
    print(f"    refuted_scm_edge_id = {refuted_edge}")
    print(f"    source_write_log_id = {src_log_id}")
    print(f"    affected           = {len(affected_ids or [])} objects "
          f"({', '.join(sorted(set(affected_types or [])))})")
    print(f"    reason             = {reason}")
    print(f"    proposed_at        = {proposed_at}")
    if applied_at:
        print(f"    applied_at         = {applied_at}")


def cmd_list(args: argparse.Namespace) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, rule_kind, status, refuted_scm_edge_id,
                       source_write_log_id, affected_object_ids,
                       affected_object_types, reason, proposed_at, applied_at
                  FROM erik_ops.propagation_events
                 ORDER BY id DESC
                 LIMIT %s
            """, (args.limit,))
            rows = cur.fetchall()
    if not rows:
        print("(no propagation_events)")
        return 0
    print(f"most recent {len(rows)} propagation_events:")
    for row in rows:
        _print_event(row)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, rule_kind, status, refuted_scm_edge_id,
                       source_write_log_id, affected_object_ids,
                       affected_object_types, applied_change, reason,
                       proposed_at, applied_at, rolled_back_at
                  FROM erik_ops.propagation_events
                 WHERE id = %s
            """, (args.event_id,))
            row = cur.fetchone()
    if row is None:
        print(f"no propagation_events row with id={args.event_id}", file=sys.stderr)
        return 2
    (ev_id, rule_kind, status, refuted_edge, src_log_id,
     affected_ids, affected_types, applied_change, reason,
     proposed_at, applied_at, rolled_back_at) = row
    print(json.dumps({
        'id': ev_id, 'rule_kind': rule_kind, 'status': status,
        'refuted_scm_edge_id': refuted_edge, 'source_write_log_id': src_log_id,
        'affected_object_ids': list(affected_ids or []),
        'affected_object_types': list(affected_types or []),
        'reason': reason,
        'proposed_at': str(proposed_at),
        'applied_at': str(applied_at) if applied_at else None,
        'rolled_back_at': str(rolled_back_at) if rolled_back_at else None,
        'applied_change': applied_change if isinstance(applied_change, dict) else {},
    }, indent=2, default=str))
    return 0


def _confirm(prompt: str, auto_yes: bool) -> bool:
    if auto_yes:
        return True
    try:
        reply = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return reply in ('y', 'yes')


def cmd_rollback(args: argparse.Namespace) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT status, rule_kind, affected_object_ids
                  FROM erik_ops.propagation_events
                 WHERE id = %s
            """, (args.event_id,))
            row = cur.fetchone()
    if row is None:
        print(f"no propagation_events row with id={args.event_id}", file=sys.stderr)
        return 2
    status, rule_kind, affected_ids = row
    if status != 'applied':
        print(f"event {args.event_id} has status={status}; only 'applied' can be rolled back",
              file=sys.stderr)
        return 2
    n_affected = len(affected_ids or [])
    if not _confirm(
        f"Roll back propagation_events #{args.event_id} "
        f"({rule_kind}, {n_affected} affected objects)?",
        args.yes,
    ):
        print("aborted.")
        return 1
    with get_connection() as conn:
        result = rollback_event(conn, args.event_id)
    print(json.dumps(result, indent=2))
    return 0 if result.get('rolled_back') else 2


def cmd_rollback_by_source(args: argparse.Namespace) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, rule_kind, status, affected_object_ids
                  FROM erik_ops.propagation_events
                 WHERE source_write_log_id = %s
                   AND status = 'applied'
                 ORDER BY id ASC
            """, (args.source_write_log_id,))
            rows = cur.fetchall()
    if not rows:
        print(f"no applied events tied to source_write_log_id={args.source_write_log_id}")
        return 0
    if not _confirm(
        f"Roll back {len(rows)} applied propagation events "
        f"tied to write_log_id={args.source_write_log_id}?",
        args.yes,
    ):
        print("aborted.")
        return 1
    outcomes = []
    with get_connection() as conn:
        for ev_id, rule_kind, _status, _ids in rows:
            r = rollback_event(conn, ev_id)
            outcomes.append({'event_id': ev_id, 'rule_kind': rule_kind, **r})
    print(json.dumps(outcomes, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='propagation_rollback')
    p.add_argument('--yes', action='store_true', help='skip confirmation prompts')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('list', help='list recent propagation events')
    s.add_argument('--limit', type=int, default=20)
    s.set_defaults(func=cmd_list)

    s = sub.add_parser('show', help='show one propagation event in full detail')
    s.add_argument('event_id', type=int)
    s.set_defaults(func=cmd_show)

    s = sub.add_parser('rollback', help='roll back one applied event by id')
    s.add_argument('event_id', type=int)
    s.set_defaults(func=cmd_rollback)

    s = sub.add_parser('rollback-by-source',
                       help='roll back all applied events tied to a source write_log_id')
    s.add_argument('source_write_log_id', type=int)
    s.set_defaults(func=cmd_rollback_by_source)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
