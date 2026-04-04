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


def _build_compound_properties_query(
    molecule_chembl_id: str,
) -> tuple[str, tuple]:
    """Return (sql, params) for compound physicochemical properties.

    Joins: molecule_dictionary → compound_properties + compound_structures
    """
    sql = """
        SELECT
            md.chembl_id            AS molecule_chembl_id,
            md.pref_name            AS molecule_name,
            cp.mw_freebase          AS mw_freebase,
            cp.full_mwt             AS full_mwt,
            cp.alogp                AS alogp,
            cp.hba                  AS hba,
            cp.hbd                  AS hbd,
            cp.psa                  AS psa,
            cp.num_ro5_violations   AS num_ro5_violations,
            cp.aromatic_rings       AS aromatic_rings,
            cp.heavy_atoms          AS heavy_atoms,
            cp.qed_weighted         AS qed_weighted,
            cs.canonical_smiles     AS canonical_smiles
        FROM molecule_dictionary md
        JOIN compound_properties cp
            ON md.molregno = cp.molregno
        LEFT JOIN compound_structures cs
            ON md.molregno = cs.molregno
        WHERE md.chembl_id = ?
    """
    return sql.strip(), (molecule_chembl_id,)


def _build_drug_mechanism_query(
    target_chembl_id: str | None,
    max_results: int,
) -> tuple[str, tuple]:
    """Return (sql, params) for drug mechanism-of-action data.

    Joins: drug_mechanism → molecule_dictionary + target_dictionary
    Optionally filtered by target_chembl_id.
    """
    sql = """
        SELECT
            md.chembl_id            AS molecule_chembl_id,
            md.pref_name            AS molecule_name,
            td.chembl_id            AS target_chembl_id,
            td.pref_name            AS target_name,
            dm.mechanism_of_action  AS mechanism_of_action,
            dm.action_type          AS action_type,
            dm.direct_interaction   AS direct_interaction
        FROM drug_mechanism dm
        JOIN molecule_dictionary md
            ON dm.molregno = md.molregno
        LEFT JOIN target_dictionary td
            ON dm.tid = td.tid
    """
    if target_chembl_id is not None:
        sql += "    WHERE td.chembl_id = ?\n"
        sql += "    LIMIT ?"
        return sql.strip(), (target_chembl_id, max_results)
    sql += "    LIMIT ?"
    return sql.strip(), (max_results,)


def _build_metabolism_query(
    molecule_chembl_id: str,
) -> tuple[str, tuple]:
    """Return (sql, params) for metabolism/metabolite data.

    Joins: metabolism → compound_records (drug) → molecule_dictionary (drug)
           metabolism → compound_records (metabolite) → molecule_dictionary (metabolite)
    """
    sql = """
        SELECT
            md_drug.chembl_id       AS molecule_chembl_id,
            md_drug.pref_name       AS molecule_name,
            met.enzyme_name         AS enzyme_name,
            met.met_conversion      AS met_conversion,
            md_met.chembl_id        AS metabolite_chembl_id,
            md_met.pref_name        AS metabolite_name
        FROM metabolism met
        JOIN compound_records cr_drug
            ON met.drug_record_id = cr_drug.record_id
        JOIN molecule_dictionary md_drug
            ON cr_drug.molregno = md_drug.molregno
        LEFT JOIN compound_records cr_met
            ON met.metabolite_record_id = cr_met.record_id
        LEFT JOIN molecule_dictionary md_met
            ON cr_met.molregno = md_met.molregno
        WHERE md_drug.chembl_id = ?
    """
    return sql.strip(), (molecule_chembl_id,)


def _build_drug_indication_query(
    molecule_chembl_id: str,
) -> tuple[str, tuple]:
    """Return (sql, params) for drug indication (therapeutic area) data.

    Joins: drug_indication → molecule_dictionary
    Ordered by max_phase_for_ind descending.
    """
    sql = """
        SELECT
            md.chembl_id            AS molecule_chembl_id,
            md.pref_name            AS molecule_name,
            di.max_phase_for_ind    AS max_phase_for_ind,
            di.mesh_id              AS mesh_id,
            di.mesh_heading         AS mesh_heading,
            di.efo_id               AS efo_id,
            di.efo_term             AS efo_term
        FROM drug_indication di
        JOIN molecule_dictionary md
            ON di.molregno = md.molregno
        WHERE md.chembl_id = ?
        ORDER BY di.max_phase_for_ind DESC
    """
    return sql.strip(), (molecule_chembl_id,)


def _build_properties_by_target_query(
    uniprot_id: str,
    max_results: int,
) -> tuple[str, tuple]:
    """Return (sql, params) for compound properties of molecules active against a target.

    Big join: activities → assays → target_dictionary → target_components
              → component_sequences + compound_properties + compound_structures
    """
    sql = """
        SELECT DISTINCT
            md.chembl_id            AS molecule_chembl_id,
            md.pref_name            AS molecule_name,
            cp.mw_freebase          AS mw_freebase,
            cp.full_mwt             AS full_mwt,
            cp.alogp                AS alogp,
            cp.hba                  AS hba,
            cp.hbd                  AS hbd,
            cp.psa                  AS psa,
            cp.num_ro5_violations   AS num_ro5_violations,
            cp.aromatic_rings       AS aromatic_rings,
            cp.heavy_atoms          AS heavy_atoms,
            cp.qed_weighted         AS qed_weighted,
            cs.canonical_smiles     AS canonical_smiles
        FROM activities act
        JOIN assays ass
            ON act.assay_id = ass.assay_id
        JOIN target_dictionary td
            ON ass.tid = td.tid
        JOIN target_components tc
            ON td.tid = tc.tid
        JOIN component_sequences cseq
            ON tc.component_id = cseq.component_id
        JOIN molecule_dictionary md
            ON act.molregno = md.molregno
        JOIN compound_properties cp
            ON md.molregno = cp.molregno
        LEFT JOIN compound_structures cs
            ON md.molregno = cs.molregno
        WHERE cseq.accession = ?
          AND act.standard_value IS NOT NULL
        LIMIT ?
    """
    return sql.strip(), (uniprot_id, max_results)


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
    # ADME/Tox expansion: compound-level queries
    # ------------------------------------------------------------------

    def fetch_compound_properties(
        self,
        molecule_chembl_id: str,
    ) -> ConnectorResult:
        """Fetch physicochemical properties for a single molecule.

        Parameters
        ----------
        molecule_chembl_id:
            ChEMBL molecule ID, e.g. "CHEMBL25".

        Returns
        -------
        ConnectorResult with one EvidenceItem per row (typically one).
        """
        result = ConnectorResult()
        sql, params = _build_compound_properties_query(molecule_chembl_id)
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
            result.errors.append(
                f"Properties query failed for {molecule_chembl_id}: {e}"
            )
            conn.close()
            return result
        finally:
            try:
                conn.close()
            except Exception:
                pass

        for row in rows:
            try:
                item = self._props_to_evidence_item(row)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                result.errors.append(
                    f"Failed to convert properties row {dict(row)}: {e}"
                )
        return result

    def fetch_drug_mechanisms(
        self,
        target_chembl_id: str | None = None,
        max_results: int = 100,
    ) -> ConnectorResult:
        """Fetch mechanism-of-action records, optionally filtered by target.

        Parameters
        ----------
        target_chembl_id:
            Optional ChEMBL target ID filter.  If None, returns all MOAs.
        max_results:
            LIMIT clause value.

        Returns
        -------
        ConnectorResult with one EvidenceItem per MOA record.
        """
        result = ConnectorResult()
        sql, params = _build_drug_mechanism_query(target_chembl_id, max_results)
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
            result.errors.append(
                f"Drug mechanism query failed (target={target_chembl_id}): {e}"
            )
            conn.close()
            return result
        finally:
            try:
                conn.close()
            except Exception:
                pass

        for row in rows:
            try:
                item = self._moa_to_evidence_item(row)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                result.errors.append(
                    f"Failed to convert MOA row {dict(row)}: {e}"
                )
        return result

    def fetch_metabolism(
        self,
        molecule_chembl_id: str,
    ) -> ConnectorResult:
        """Fetch metabolism/metabolite data for a molecule.

        Parameters
        ----------
        molecule_chembl_id:
            ChEMBL molecule ID, e.g. "CHEMBL25".

        Returns
        -------
        ConnectorResult with one EvidenceItem per metabolism row.
        """
        result = ConnectorResult()
        sql, params = _build_metabolism_query(molecule_chembl_id)
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
            result.errors.append(
                f"Metabolism query failed for {molecule_chembl_id}: {e}"
            )
            conn.close()
            return result
        finally:
            try:
                conn.close()
            except Exception:
                pass

        for row in rows:
            try:
                item = self._metabolism_to_evidence_item(row)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                result.errors.append(
                    f"Failed to convert metabolism row {dict(row)}: {e}"
                )
        return result

    def fetch_drug_indications(
        self,
        molecule_chembl_id: str,
    ) -> ConnectorResult:
        """Fetch drug indication (therapeutic area) data for a molecule.

        Parameters
        ----------
        molecule_chembl_id:
            ChEMBL molecule ID, e.g. "CHEMBL25".

        Returns
        -------
        ConnectorResult with one EvidenceItem per indication row.
        """
        result = ConnectorResult()
        sql, params = _build_drug_indication_query(molecule_chembl_id)
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
            result.errors.append(
                f"Indication query failed for {molecule_chembl_id}: {e}"
            )
            conn.close()
            return result
        finally:
            try:
                conn.close()
            except Exception:
                pass

        for row in rows:
            try:
                item = self._indication_to_evidence_item(row)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                result.errors.append(
                    f"Failed to convert indication row {dict(row)}: {e}"
                )
        return result

    def fetch_full_profile(
        self,
        uniprot_id: str,
        max_compounds: int = 50,
    ) -> ConnectorResult:
        """Fetch a full ADME/Tox profile for a target: bioactivity + properties + MOA.

        This is the primary entry point for the ADME/Tox expansion.  It:
        1. Fetches bioactivity for the target (IC50)
        2. For each discovered molecule, fetches compound properties
        3. Fetches all MOA records for the target

        Parameters
        ----------
        uniprot_id:
            UniProt accession, e.g. "Q99720" (SIGMAR1).
        max_compounds:
            Maximum number of compounds from the bioactivity query.

        Returns
        -------
        Aggregated ConnectorResult across all sub-queries.
        """
        combined = ConnectorResult()

        # 1) Bioactivity
        bio_result = self.fetch_bioactivity(
            uniprot_id, activity_type="IC50", max_results=max_compounds
        )
        combined.evidence_items_added += bio_result.evidence_items_added
        combined.skipped_duplicates += bio_result.skipped_duplicates
        combined.errors.extend(bio_result.errors)

        # 2) Compound properties for each molecule found
        #    We need to re-query to get the molecule_chembl_ids.
        molecule_ids = self._get_molecule_ids_for_target(uniprot_id, max_compounds)
        for mol_id in molecule_ids:
            sub = self.fetch_compound_properties(mol_id)
            combined.evidence_items_added += sub.evidence_items_added
            combined.skipped_duplicates += sub.skipped_duplicates
            combined.errors.extend(sub.errors)

        # 3) MOA — look up the ChEMBL target ID first
        target_chembl_id = self._get_target_chembl_id(uniprot_id)
        if target_chembl_id:
            moa_result = self.fetch_drug_mechanisms(
                target_chembl_id=target_chembl_id, max_results=max_compounds
            )
            combined.evidence_items_added += moa_result.evidence_items_added
            combined.skipped_duplicates += moa_result.skipped_duplicates
            combined.errors.extend(moa_result.errors)

        return combined

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_molecule_ids_for_target(
        self,
        uniprot_id: str,
        max_results: int,
    ) -> list[str]:
        """Return distinct molecule_chembl_ids active against a UniProt target."""
        sql, params = _build_bioactivity_query(uniprot_id, "IC50", max_results)
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        except Exception:
            return []
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            seen: set[str] = set()
            ids: list[str] = []
            for row in rows:
                mid = row["molecule_chembl_id"]
                if mid and mid not in seen:
                    seen.add(mid)
                    ids.append(mid)
            return ids
        except Exception:
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _get_target_chembl_id(self, uniprot_id: str) -> str | None:
        """Look up the ChEMBL target ID for a UniProt accession."""
        sql = """
            SELECT td.chembl_id
            FROM target_dictionary td
            JOIN target_components tc ON td.tid = tc.tid
            JOIN component_sequences cs ON tc.component_id = cs.component_id
            WHERE cs.accession = ?
            LIMIT 1
        """
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        except Exception:
            return None
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(sql, (uniprot_id,))
            row = cur.fetchone()
            return row["chembl_id"] if row else None
        except Exception:
            return None
        finally:
            try:
                conn.close()
            except Exception:
                pass

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

    def _props_to_evidence_item(self, row: sqlite3.Row) -> EvidenceItem:
        """Convert a compound_properties row to an EvidenceItem."""
        molecule_chembl_id = row["molecule_chembl_id"] or ""
        molecule_name = row["molecule_name"] or molecule_chembl_id
        mw = row["mw_freebase"]
        alogp = row["alogp"]
        hba = row["hba"]
        hbd = row["hbd"]
        psa = row["psa"]
        ro5 = row["num_ro5_violations"]

        claim = (
            f"{molecule_name} has MW={mw}, ALogP={alogp}, "
            f"HBA={hba}, HBD={hbd}, PSA={psa}, RO5_violations={ro5}"
        )

        return EvidenceItem(
            id=f"evi:chembl_props:{molecule_chembl_id}",
            claim=claim,
            direction=EvidenceDirection.supports,
            strength=EvidenceStrength.preclinical,
            provenance=Provenance(
                source_system=SourceSystem.database,
                asserted_by="chembl_connector",
            ),
            body={
                "pch_layer": 2,
                "molecule_chembl_id": molecule_chembl_id,
                "molecule_name": molecule_name,
                "mw_freebase": mw,
                "full_mwt": row["full_mwt"],
                "alogp": alogp,
                "hba": hba,
                "hbd": hbd,
                "psa": psa,
                "num_ro5_violations": ro5,
                "aromatic_rings": row["aromatic_rings"],
                "heavy_atoms": row["heavy_atoms"],
                "qed_weighted": row["qed_weighted"],
                "canonical_smiles": row["canonical_smiles"] or "",
            },
        )

    def _moa_to_evidence_item(self, row: sqlite3.Row) -> EvidenceItem:
        """Convert a drug_mechanism row to an EvidenceItem."""
        molecule_chembl_id = row["molecule_chembl_id"] or ""
        target_chembl_id = row["target_chembl_id"] or ""
        molecule_name = row["molecule_name"] or molecule_chembl_id
        target_name = row["target_name"] or target_chembl_id
        moa = row["mechanism_of_action"] or ""
        action_type = row["action_type"] or ""

        claim = (
            f"{molecule_name} acts on {target_name} via {moa} "
            f"(action_type={action_type})"
        )

        return EvidenceItem(
            id=f"evi:chembl_moa:{molecule_chembl_id}_{target_chembl_id}",
            claim=claim,
            direction=EvidenceDirection.supports,
            strength=EvidenceStrength.preclinical,
            provenance=Provenance(
                source_system=SourceSystem.database,
                asserted_by="chembl_connector",
            ),
            body={
                "pch_layer": 2,
                "molecule_chembl_id": molecule_chembl_id,
                "target_chembl_id": target_chembl_id,
                "molecule_name": molecule_name,
                "target_name": target_name,
                "mechanism_of_action": moa,
                "action_type": action_type,
                "direct_interaction": row["direct_interaction"],
            },
        )

    def _metabolism_to_evidence_item(self, row: sqlite3.Row) -> EvidenceItem:
        """Convert a metabolism row to an EvidenceItem."""
        molecule_chembl_id = row["molecule_chembl_id"] or ""
        enzyme_name = row["enzyme_name"] or ""
        met_conversion = row["met_conversion"] or ""
        metabolite_chembl_id = row["metabolite_chembl_id"] or ""
        molecule_name = row["molecule_name"] or molecule_chembl_id

        # Normalize enzyme name for the ID (lowercase, replace spaces)
        enzyme_norm = enzyme_name.lower().replace(" ", "_").replace("/", "_")

        claim = (
            f"{molecule_name} is metabolized by {enzyme_name} "
            f"({met_conversion}) to {metabolite_chembl_id}"
        )

        return EvidenceItem(
            id=f"evi:chembl_met:{molecule_chembl_id}_{enzyme_norm}",
            claim=claim,
            direction=EvidenceDirection.supports,
            strength=EvidenceStrength.preclinical,
            provenance=Provenance(
                source_system=SourceSystem.database,
                asserted_by="chembl_connector",
            ),
            body={
                "pch_layer": 2,
                "molecule_chembl_id": molecule_chembl_id,
                "molecule_name": molecule_name,
                "enzyme_name": enzyme_name,
                "met_conversion": met_conversion,
                "metabolite_chembl_id": metabolite_chembl_id,
                "metabolite_name": row["metabolite_name"] or "",
            },
        )

    def _indication_to_evidence_item(self, row: sqlite3.Row) -> EvidenceItem:
        """Convert a drug_indication row to an EvidenceItem."""
        molecule_chembl_id = row["molecule_chembl_id"] or ""
        molecule_name = row["molecule_name"] or molecule_chembl_id
        mesh_id = row["mesh_id"] or ""
        mesh_heading = row["mesh_heading"] or ""
        max_phase = row["max_phase_for_ind"]
        efo_id = row["efo_id"] or ""
        efo_term = row["efo_term"] or ""

        claim = (
            f"{molecule_name} indicated for {mesh_heading} "
            f"(max_phase={max_phase})"
        )

        return EvidenceItem(
            id=f"evi:chembl_ind:{molecule_chembl_id}_{mesh_id}",
            claim=claim,
            direction=EvidenceDirection.supports,
            strength=EvidenceStrength.preclinical,
            provenance=Provenance(
                source_system=SourceSystem.database,
                asserted_by="chembl_connector",
            ),
            body={
                "pch_layer": 2,
                "molecule_chembl_id": molecule_chembl_id,
                "molecule_name": molecule_name,
                "max_phase_for_ind": max_phase,
                "mesh_id": mesh_id,
                "mesh_heading": mesh_heading,
                "efo_id": efo_id,
                "efo_term": efo_term,
            },
        )
