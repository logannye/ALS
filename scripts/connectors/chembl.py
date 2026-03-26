"""ChEMBLConnector — queries local ChEMBL 36 SQLite for bioactivity data.

ChEMBL 36 lives at /Volumes/Databank/databases/chembl_36.db and is treated
as a read-only external reference database.  sqlite3 is the one allowed
exception to the "never use sqlite3" rule because ChEMBL is not part of
the Erik operational state.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem
from targets.als_targets import ALS_TARGETS

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "/Volumes/Databank/databases/chembl_36.db"

# Priority ALS targets for the bulk fetch shortcut
PRIORITY_TARGET_KEYS = ["SIGMAR1", "EAAT2", "mTOR", "CSF1R", "TDP-43"]


# ---------------------------------------------------------------------------
# Free function: build the bioactivity SQL query
# ---------------------------------------------------------------------------

def _build_bioactivity_query(
    uniprot_id: str,
    activity_type: str,
    max_results: int,
) -> tuple[str, tuple]:
    """Return (sql, params) for the ChEMBL bioactivity JOIN query.

    Joins: activities → assays → target_dictionary → target_components
           → component_sequences → molecule_dictionary

    Parameters
    ----------
    uniprot_id:
        UniProt accession to filter on (via component_sequences.accession).
    activity_type:
        Activity standard type, e.g. "IC50", "Ki", "EC50".
    max_results:
        LIMIT clause value.

    Returns
    -------
    (sql_string, params_tuple)
    """
    sql = """
        SELECT
            md.chembl_id          AS molecule_chembl_id,
            md.pref_name          AS molecule_name,
            td.chembl_id          AS target_chembl_id,
            td.pref_name          AS target_name,
            act.standard_type     AS activity_type,
            act.standard_value    AS activity_value,
            act.standard_units    AS activity_units
        FROM activities act
        JOIN assays ass
            ON act.assay_id = ass.assay_id
        JOIN target_dictionary td
            ON ass.tid = td.tid
        JOIN target_components tc
            ON td.tid = tc.tid
        JOIN component_sequences cs
            ON tc.component_id = cs.component_id
        JOIN molecule_dictionary md
            ON act.molregno = md.molregno
        WHERE cs.accession = ?
          AND act.standard_type = ?
          AND act.standard_value IS NOT NULL
          AND act.standard_relation = '='
        ORDER BY act.standard_value ASC
        LIMIT ?
    """
    params = (uniprot_id, activity_type, max_results)
    return sql.strip(), params


# ---------------------------------------------------------------------------
# ChEMBLConnector
# ---------------------------------------------------------------------------

class ChEMBLConnector(BaseConnector):
    """Connector for the local ChEMBL 36 SQLite reference database.

    Queries bioactivity data for ALS-relevant targets and converts rows
    into EvidenceItem objects.

    sqlite3 is used here intentionally — ChEMBL is an external reference DB
    and is never part of Erik's operational PostgreSQL state.
    """

    def __init__(
        self,
        *,
        db_path: str = DEFAULT_DB_PATH,
        store=None,
    ) -> None:
        self.db_path = db_path
        self._store = store

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(self, **kwargs) -> ConnectorResult:
        """Top-level fetch: delegates to fetch_for_priority_targets."""
        return self.fetch_for_priority_targets()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_bioactivity(
        self,
        uniprot_id: str,
        activity_type: str = "IC50",
        max_results: int = 100,
    ) -> ConnectorResult:
        """Fetch bioactivity rows for a UniProt accession → EvidenceItems.

        Parameters
        ----------
        uniprot_id:
            UniProt accession, e.g. "Q99720" (SIGMAR1).
        activity_type:
            ChEMBL standard_type filter, e.g. "IC50".
        max_results:
            Maximum rows to return.

        Returns
        -------
        ConnectorResult with evidence_items_added count and any errors.
        """
        result = ConnectorResult()
        sql, params = _build_bioactivity_query(uniprot_id, activity_type, max_results)
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        except Exception as e:
            result.errors.append(f"Cannot open ChEMBL DB at {self.db_path!r}: {e}")
            return result

        try:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
        except Exception as e:
            result.errors.append(f"Query failed for uniprot={uniprot_id}: {e}")
            conn.close()
            return result
        finally:
            try:
                conn.close()
            except Exception:
                pass

        for row in rows:
            try:
                item = self._row_to_evidence_item(row)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                result.errors.append(
                    f"Failed to convert row {dict(row)}: {e}"
                )

        return result

    def fetch_compounds_for_target(
        self,
        target_name: str,
        activity_type: str = "IC50",
        max_results: int = 100,
    ) -> ConnectorResult:
        """Fetch compounds for a named ALS target.

        Looks up the UniProt ID from als_targets.ALS_TARGETS, then calls
        fetch_bioactivity.

        Parameters
        ----------
        target_name:
            Canonical ALS target key, e.g. "SIGMAR1".
        activity_type:
            ChEMBL standard_type filter.
        max_results:
            Maximum rows to return.

        Returns
        -------
        ConnectorResult with evidence_items_added count and any errors.
        """
        result = ConnectorResult()
        target = ALS_TARGETS.get(target_name)
        if target is None:
            result.errors.append(
                f"Unknown target {target_name!r} — not found in ALS_TARGETS"
            )
            return result

        uniprot_id = target.get("uniprot_id", "")
        if not uniprot_id:
            result.errors.append(
                f"Target {target_name!r} has no uniprot_id"
            )
            return result

        sub = self.fetch_bioactivity(
            uniprot_id, activity_type=activity_type, max_results=max_results
        )
        result.evidence_items_added += sub.evidence_items_added
        result.skipped_duplicates += sub.skipped_duplicates
        result.errors.extend(sub.errors)
        return result

    def fetch_for_priority_targets(
        self,
        max_per_target: int = 50,
    ) -> ConnectorResult:
        """Fetch bioactivity for the five highest-priority ALS drug targets.

        Priority targets: SIGMAR1, EAAT2, mTOR, CSF1R, TDP-43.

        Parameters
        ----------
        max_per_target:
            Maximum compounds to retrieve per target.

        Returns
        -------
        Aggregated ConnectorResult across all priority targets.
        """
        combined = ConnectorResult()
        for key in PRIORITY_TARGET_KEYS:
            sub = self.fetch_compounds_for_target(
                key, max_results=max_per_target
            )
            combined.evidence_items_added += sub.evidence_items_added
            combined.skipped_duplicates += sub.skipped_duplicates
            combined.errors.extend(sub.errors)
        return combined

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _row_to_evidence_item(self, row: sqlite3.Row) -> EvidenceItem:
        """Convert a ChEMBL bioactivity row to an EvidenceItem."""
        molecule_chembl_id = row["molecule_chembl_id"] or ""
        target_chembl_id = row["target_chembl_id"] or ""
        molecule_name = row["molecule_name"] or molecule_chembl_id
        target_name = row["target_name"] or target_chembl_id
        activity_type = row["activity_type"] or ""
        activity_value = row["activity_value"]
        activity_units = row["activity_units"] or ""

        claim = (
            f"{molecule_name} has {activity_type} = {activity_value} "
            f"{activity_units} against {target_name}"
        )

        return EvidenceItem(
            id=f"evi:chembl:{molecule_chembl_id}_{target_chembl_id}",
            claim=claim,
            direction=EvidenceDirection.supports,
            strength=EvidenceStrength.preclinical,
            provenance=Provenance(
                source_system=SourceSystem.database,
                asserted_by="chembl_connector",
            ),
            body={
                "protocol_layer": "",
                "mechanism_target": target_name,
                "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
                "erik_eligible": True,
                "pch_layer": 2,
                "molecule_chembl_id": molecule_chembl_id,
                "target_chembl_id": target_chembl_id,
                "activity_type": activity_type,
                "activity_value": activity_value,
                "activity_units": activity_units,
            },
        )
