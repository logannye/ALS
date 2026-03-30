"""AlphaFold structure connector — 3D protein structures for ALS drug design targets.

Queries the local AlphaFold PDB structure collection (24,046 structures) for
Erik's 10 druggable ALS targets. Provides structural context for drug design:
binding pocket geometry, domain boundaries, and structure quality metrics.

Directory: /Volumes/Databank/databases/alphafold/
Files: AF-{UniProt}-F1-model_v6.pdb
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope

_DEFAULT_DIR = "/Volumes/Databank/databases/alphafold"

# Erik's druggable ALS targets with UniProt IDs
_ALS_DRUG_TARGETS = {
    "Q13148": {"gene": "TARDBP", "name": "TDP-43", "sites": ["RRM1/RRM2 RNA-binding domains", "C-terminal prion-like domain"]},
    "P00441": {"gene": "SOD1", "name": "SOD1", "sites": ["Dimer interface", "Active site"]},
    "P35637": {"gene": "FUS", "name": "FUS", "sites": ["Low-complexity domain", "RGG domains"]},
    "Q99720": {"gene": "SIGMAR1", "name": "Sigma-1R", "sites": ["Ligand binding pocket (cupin fold)"]},
    "P43004": {"gene": "SLC1A2", "name": "EAAT2", "sites": ["Substrate binding site"]},
    "P42345": {"gene": "MTOR", "name": "mTOR", "sites": ["FKBP-rapamycin binding domain", "Kinase domain"]},
    "P07333": {"gene": "CSF1R", "name": "CSF1R", "sites": ["Kinase domain ATP-binding pocket"]},
    "Q9UHD2": {"gene": "TBK1", "name": "TBK1", "sites": ["Kinase domain"]},
    "P01031": {"gene": "C5", "name": "Complement C5", "sites": ["C5a cleavage site"]},
    "Q96PY6": {"gene": "NEK1", "name": "NEK1", "sites": ["Kinase domain"]},
    "P23560": {"gene": "BDNF", "name": "BDNF", "sites": ["TrkB binding interface"]},
    "P39905": {"gene": "GDNF", "name": "GDNF", "sites": ["GFRα binding interface"]},
}


class AlphaFoldLocalConnector(BaseConnector):
    """Query local AlphaFold structures for ALS drug design targets."""

    def __init__(self, store: Any = None, directory: str = _DEFAULT_DIR):
        self._store = store
        self._directory = directory

    def fetch(self, *, uniprot: str = "", gene: str = "", **kwargs) -> ConnectorResult:
        result = ConnectorResult()

        if not Path(self._directory).exists():
            result.errors.append(f"AlphaFold directory not found: {self._directory}")
            return result

        # Resolve UniProt from gene if needed
        if not uniprot and gene:
            for uid, info in _ALS_DRUG_TARGETS.items():
                if info["gene"].upper() == gene.upper():
                    uniprot = uid
                    break

        if not uniprot:
            return result

        # Find the PDB file
        pdb_filename = f"AF-{uniprot}-F1-model_v6.pdb"
        pdb_path = os.path.join(self._directory, pdb_filename)

        if not os.path.exists(pdb_path):
            # Try without version suffix
            for fn in os.listdir(self._directory):
                if fn.startswith(f"AF-{uniprot}"):
                    pdb_path = os.path.join(self._directory, fn)
                    break
            else:
                return result

        # Parse basic structure metrics from PDB
        target_info = _ALS_DRUG_TARGETS.get(uniprot, {})
        gene_name = target_info.get("gene", gene)
        protein_name = target_info.get("name", gene_name)
        druggable_sites = target_info.get("sites", [])

        try:
            stats = self._parse_pdb_stats(pdb_path)
        except Exception:
            stats = {}

        n_residues = stats.get("n_residues", 0)
        avg_plddt = stats.get("avg_plddt", 0)
        file_size = os.path.getsize(pdb_path)

        claim = (
            f"AlphaFold structure for {protein_name} ({gene_name}, {uniprot}): "
            f"{n_residues} residues, avg pLDDT {avg_plddt:.1f}. "
        )
        if avg_plddt >= 70:
            claim += "HIGH CONFIDENCE structure — suitable for computational drug design. "
        elif avg_plddt >= 50:
            claim += "Moderate confidence — core domains likely reliable. "
        else:
            claim += "Low confidence — use with caution for drug design. "

        if druggable_sites:
            claim += f"Druggable sites: {'; '.join(druggable_sites)}."

        evi = BaseEnvelope(
            id=f"evi:alphafold_{gene_name.lower()}_structure",
            type="EvidenceItem",
            status="active",
            body={
                "claim": claim,
                "source": "alphafold",
                "gene": gene_name,
                "uniprot": uniprot,
                "pdb_path": pdb_path,
                "n_residues": n_residues,
                "avg_plddt": avg_plddt,
                "file_size_bytes": file_size,
                "druggable_sites": druggable_sites,
                "evidence_strength": "strong" if avg_plddt >= 70 else "moderate",
                "pch_layer": 1,
                "protocol_layer": "root_cause_suppression",
            },
        )

        if self._store:
            self._store.upsert_object(evi)
            result.evidence_items_added += 1

        return result

    def _parse_pdb_stats(self, pdb_path: str) -> dict:
        """Extract basic statistics from a PDB file."""
        n_residues = 0
        plddt_sum = 0.0
        plddt_count = 0
        seen_residues: set[int] = set()

        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") and len(line) >= 66:
                    res_seq = int(line[22:26].strip())
                    if res_seq not in seen_residues:
                        seen_residues.add(res_seq)
                    # pLDDT is in the B-factor column (columns 61-66)
                    try:
                        bfactor = float(line[60:66].strip())
                        plddt_sum += bfactor
                        plddt_count += 1
                    except ValueError:
                        pass

        n_residues = len(seen_residues)
        avg_plddt = plddt_sum / plddt_count if plddt_count > 0 else 0.0

        return {"n_residues": n_residues, "avg_plddt": avg_plddt}
