"""BindingDB connector — experimental protein-ligand binding affinities.

Queries the local BindingDB TSV (7.9GB, 3.2M measurements) for binding
data between ALS drug candidates and their targets. Provides Ki, IC50, Kd
values with quantitative precision — essential for drug design and ranking.

File: /Volumes/Databank/databases/bindingdb/BindingDB_All.tsv
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope

_DEFAULT_PATH = "/Volumes/Databank/databases/bindingdb/BindingDB_All.tsv"

# Pre-index: ALS target UniProt IDs for fast matching
_ALS_TARGET_UNIPROTS = {
    "Q13148": "TARDBP", "P00441": "SOD1", "P35637": "FUS",
    "Q96LT7": "C9orf72", "Q93045": "STMN2", "Q99720": "SIGMAR1",
    "P43004": "SLC1A2", "P42345": "MTOR", "P07333": "CSF1R",
    "Q9UHD2": "TBK1", "P01031": "C5", "P23560": "BDNF",
    "P39905": "GDNF", "Q96CV9": "OPTN", "Q96PY6": "NEK1",
}

# ALS drug names for matching
_ALS_DRUGS = frozenset({
    "riluzole", "edaravone", "rapamycin", "sirolimus", "masitinib",
    "pridopidine", "ibudilast", "memantine", "tofersen", "trehalose",
    "perampanel", "verdiperstat", "zilucoplan",
})


class BindingDBConnector(BaseConnector):
    """Query BindingDB for binding affinities of ALS drug-target pairs."""

    def __init__(self, store: Any = None, file_path: str = _DEFAULT_PATH):
        self._store = store
        self._file_path = file_path

    def fetch(self, *, gene: str = "", drug: str = "", **kwargs) -> ConnectorResult:
        result = ConnectorResult()

        if not Path(self._file_path).exists():
            result.errors.append(f"BindingDB not found: {self._file_path}")
            return result

        try:
            hits = self._search(gene=gene, drug=drug)
            if not hits:
                return result

            # Build evidence items from binding data
            for hit in hits[:10]:
                ki = hit.get("ki", "")
                ic50 = hit.get("ic50", "")
                kd = hit.get("kd", "")
                drug_name = hit.get("drug_name", drug)
                target_name = hit.get("target_name", gene)

                # Pick the best affinity value
                affinity_str = ""
                if ki and ki != "":
                    affinity_str = f"Ki = {ki} nM"
                elif ic50 and ic50 != "":
                    affinity_str = f"IC50 = {ic50} nM"
                elif kd and kd != "":
                    affinity_str = f"Kd = {kd} nM"
                else:
                    continue

                claim = (
                    f"BindingDB: {drug_name} binds {target_name} with {affinity_str}. "
                )
                # Assess potency
                try:
                    val = float(ki or ic50 or kd or "99999")
                    if val < 10:
                        claim += "HIGHLY POTENT binder (< 10 nM)."
                    elif val < 100:
                        claim += "Potent binder (< 100 nM)."
                    elif val < 1000:
                        claim += "Moderate binder (< 1 uM)."
                    else:
                        claim += "Weak binder (> 1 uM)."
                except ValueError:
                    pass

                evi_id = f"evi:bindingdb_{drug_name}_{target_name}".lower().replace(" ", "_")[:80]
                evi = BaseEnvelope(
                    id=evi_id,
                    type="EvidenceItem",
                    status="active",
                    body={
                        "claim": claim,
                        "source": "bindingdb",
                        "drug_name": drug_name,
                        "target_name": target_name,
                        "ki_nm": ki,
                        "ic50_nm": ic50,
                        "kd_nm": kd,
                        "evidence_strength": "strong",
                        "pch_layer": 2,
                        "protocol_layer": "root_cause_suppression",
                        "experiment_type": "binding_affinity",
                    },
                )

                if self._store:
                    try:
                        self._store.upsert_object(evi)
                        result.evidence_items_added += 1
                    except Exception:
                        pass

        except Exception as e:
            result.errors.append(f"BindingDB search failed: {e}")

        return result

    def _search(self, gene: str = "", drug: str = "", max_results: int = 50) -> list[dict]:
        """Stream-search the large BindingDB TSV for matching drug-target pairs."""
        hits: list[dict] = []
        gene_lower = gene.lower() if gene else ""
        drug_lower = drug.lower() if drug else ""

        with open(self._file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                target = (row.get("Target Name") or "").lower()
                ligand = (row.get("BindingDB Ligand Name") or "").lower()

                # Match by gene/target name
                if gene_lower and gene_lower not in target:
                    continue
                # Match by drug name
                if drug_lower and drug_lower not in ligand:
                    continue
                # Must match at least one criterion
                if not gene_lower and not drug_lower:
                    # If neither specified, look for any ALS drug
                    if not any(d in ligand for d in _ALS_DRUGS):
                        continue

                ki = row.get("Ki (nM)", "")
                ic50 = row.get("IC50 (nM)", "")
                kd = row.get("Kd (nM)", "")

                if not any([ki, ic50, kd]):
                    continue

                hits.append({
                    "drug_name": row.get("BindingDB Ligand Name", drug),
                    "target_name": row.get("Target Name", gene),
                    "ki": ki,
                    "ic50": ic50,
                    "kd": kd,
                    "smiles": row.get("Ligand SMILES", ""),
                })

                if len(hits) >= max_results:
                    break

        return hits
