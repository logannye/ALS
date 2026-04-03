"""Molecular property computation for drug discovery.

Uses RDKit to compute drug-likeness, CNS Multi-Parameter Optimization
(MPO) scores, and synthetic accessibility for compounds targeting ALS
drug design targets.  This is the first molecular reasoning capability
in the system — the foundation for docking and de novo generation.

Requires: pip install rdkit-pypi
"""
from __future__ import annotations

import json
import logging
import math
import pathlib
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Drug design targets seed file
_TARGETS_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "seed" / "drug_design_targets.json"


@dataclass
class MolecularProperties:
    """Computed properties for a single molecule."""
    smiles: str
    molecular_weight: float
    logp: float
    hbd: int       # H-bond donors
    hba: int       # H-bond acceptors
    tpsa: float    # Topological polar surface area
    rotatable_bonds: int
    lipinski_violations: int
    cns_mpo: float             # CNS Multi-Parameter Optimization score (0-6)
    synthetic_accessibility: float  # SA score (1-10, lower = easier)
    is_drug_like: bool
    is_cns_penetrant: bool     # CNS MPO >= 4.0


@dataclass
class MolecularResult:
    """Result from a molecular computation experiment."""
    experiment_type: str
    target: str
    success: bool = True
    error: Optional[str] = None
    properties: list[MolecularProperties] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)


def _check_rdkit():
    """Import RDKit and raise a clear error if not installed."""
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except ImportError:
        logger.error("rdkit not installed. Run: pip install rdkit-pypi")
        return False


# ---------------------------------------------------------------------------
# Core molecular property computation
# ---------------------------------------------------------------------------

def compute_drug_properties(smiles: str) -> Optional[MolecularProperties]:
    """Compute all molecular properties for a SMILES string.

    Returns None if the SMILES is invalid.
    """
    if not _check_rdkit():
        return None

    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rotatable = Lipinski.NumRotatableBonds(mol)

    # Lipinski Rule of 5 violations
    violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])

    # CNS MPO score (Wager et al. 2010, ACS Chem Neurosci)
    # 6 desirability functions, each 0-1, summed to 0-6
    cns_mpo = _compute_cns_mpo(mw, logp, hbd, tpsa, Descriptors.MolLogP(mol))

    # Synthetic accessibility (Ertl & Schuffenhauer 2009)
    try:
        from rdkit.Chem import RDConfig
        import os
        import sys
        sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
        if sa_path not in sys.path:
            sys.path.insert(0, sa_path)
        from sascorer import calculateScore
        sa_score = calculateScore(mol)
    except Exception:
        # Fallback: estimate from fragment complexity
        sa_score = min(10.0, max(1.0, mw / 100 + rotatable / 5))

    return MolecularProperties(
        smiles=smiles,
        molecular_weight=round(mw, 2),
        logp=round(logp, 2),
        hbd=hbd,
        hba=hba,
        tpsa=round(tpsa, 2),
        rotatable_bonds=rotatable,
        lipinski_violations=violations,
        cns_mpo=round(cns_mpo, 2),
        synthetic_accessibility=round(sa_score, 2),
        is_drug_like=(violations <= 1),
        is_cns_penetrant=(cns_mpo >= 4.0),
    )


def _compute_cns_mpo(mw: float, logp: float, hbd: int, tpsa: float, clogp: float) -> float:
    """CNS Multi-Parameter Optimization score (Wager et al. 2010).

    6 desirability functions, each 0-1, summed to yield 0-6.
    Higher = better CNS penetration probability.
    """
    def _desirability_decreasing(value: float, low: float, high: float) -> float:
        if value <= low:
            return 1.0
        if value >= high:
            return 0.0
        return 1.0 - (value - low) / (high - low)

    def _desirability_hump(value: float, optimal: float, low: float, high: float) -> float:
        if value <= low or value >= high:
            return 0.0
        if value <= optimal:
            return (value - low) / (optimal - low)
        return (high - value) / (high - optimal)

    # Component desirability functions (Wager 2010 thresholds)
    d_mw = _desirability_decreasing(mw, 360, 500)           # MW: prefer < 360
    d_logp = _desirability_hump(logp, 2.0, -0.5, 5.0)      # LogP: prefer ~2
    d_hbd = _desirability_decreasing(hbd, 0.5, 3.5)         # HBD: prefer 0-1
    d_tpsa = _desirability_hump(tpsa, 45.0, 20.0, 120.0)   # TPSA: prefer ~45
    d_pka = 1.0  # Assume neutral (no pKa calculation without 3D)
    d_clogd = _desirability_hump(clogp, 2.0, -1.0, 4.0)    # CLogD ≈ CLogP for neutral

    return d_mw + d_logp + d_hbd + d_tpsa + d_pka + d_clogd


# ---------------------------------------------------------------------------
# Compound library screening
# ---------------------------------------------------------------------------

def screen_compound_library(
    target_id: str,
    chembl_path: str = "/Volumes/Databank/databases/chembl_36.db",
    max_compounds: int = 50,
) -> MolecularResult:
    """Screen known ChEMBL actives for a drug design target.

    Loads the target's associated compound libraries, retrieves SMILES
    from ChEMBL, computes properties for all, and ranks by CNS MPO +
    drug-likeness.
    """
    if not _check_rdkit():
        return MolecularResult(
            experiment_type="molecular_properties",
            target=target_id,
            success=False,
            error="rdkit not installed",
        )

    # Load drug design target
    try:
        targets = json.loads(_TARGETS_PATH.read_text())
    except Exception as e:
        return MolecularResult(
            experiment_type="molecular_properties",
            target=target_id,
            success=False,
            error=f"Could not load targets: {e}",
        )

    target = next((t for t in targets if t["id"] == target_id), None)
    if not target:
        return MolecularResult(
            experiment_type="molecular_properties",
            target=target_id,
            success=False,
            error=f"Target {target_id} not found",
        )

    # Query ChEMBL for known actives against this target
    smiles_list = _query_chembl_actives(
        target_name=target["target_name"],
        chembl_path=chembl_path,
        max_results=max_compounds,
    )

    if not smiles_list:
        return MolecularResult(
            experiment_type="molecular_properties",
            target=target_id,
            success=True,
            properties=[],
            facts=[],
        )

    # Compute properties for all compounds
    results: list[MolecularProperties] = []
    for smiles in smiles_list:
        props = compute_drug_properties(smiles)
        if props:
            results.append(props)

    # Rank by CNS MPO (descending) then drug-likeness
    results.sort(key=lambda p: (p.cns_mpo, -p.lipinski_violations), reverse=True)

    # Build evidence items for top compounds
    facts: list[dict] = []
    for i, props in enumerate(results[:10]):
        fact = {
            "id": f"evi:mol_{target_id.replace('ddt:', '')}_{i}",
            "type": "EvidenceItem",
            "status": "active",
            "body": {
                "claim": (
                    f"Compound {props.smiles[:50]} targeting {target['target_name']} "
                    f"({target['druggable_site']}): MW={props.molecular_weight}, "
                    f"LogP={props.logp}, CNS_MPO={props.cns_mpo}, "
                    f"Lipinski_violations={props.lipinski_violations}, "
                    f"SA={props.synthetic_accessibility}"
                ),
                "source": "rdkit_computation",
                "experiment_type": "molecular_properties",
                "pch_layer": 2,
                "protocol_layer": "root_cause_suppression",
                "evidence_strength": "preclinical",
                "designed_smiles": props.smiles,
                "molecular_weight": props.molecular_weight,
                "logp": props.logp,
                "hbd": props.hbd,
                "hba": props.hba,
                "tpsa": props.tpsa,
                "cns_mpo_score": props.cns_mpo,
                "synthetic_accessibility": props.synthetic_accessibility,
                "is_drug_like": props.is_drug_like,
                "is_cns_penetrant": props.is_cns_penetrant,
                "target_id": target_id,
                "target_name": target["target_name"],
                "druggable_site": target["druggable_site"],
            },
        }
        facts.append(fact)

    return MolecularResult(
        experiment_type="molecular_properties",
        target=target_id,
        success=True,
        properties=results,
        facts=facts,
    )


# ChEMBL target names don't always match drug design target names.
# This map resolves the most important ALS targets.
_CHEMBL_TARGET_ALIASES: dict[str, str] = {
    "sigma-1 receptor": "Sigma non-opioid intracellular receptor 1",
    "sigma-1": "Sigma non-opioid intracellular receptor 1",
    "sod1": "Superoxide dismutase [Cu-Zn]",
    "tdp-43": "TAR DNA-binding protein 43",
    "tardbp": "TAR DNA-binding protein 43",
    "csf1r": "Macrophage colony-stimulating factor 1 receptor",
    "mtor": "Serine/threonine-protein kinase mTOR",
    "eaat2": "Excitatory amino acid transporter 2",
}


def _query_chembl_actives(
    target_name: str,
    chembl_path: str,
    max_results: int = 50,
) -> list[str]:
    """Query ChEMBL for SMILES of known active compounds against a target.

    Uses an alias map to resolve common ALS target names to ChEMBL
    preferred names, then falls back to LIKE matching on individual words.
    """
    import sqlite3
    import os

    if not os.path.exists(chembl_path):
        logger.warning("ChEMBL database not found at %s", chembl_path)
        return []

    # Resolve alias
    resolved = _CHEMBL_TARGET_ALIASES.get(target_name.lower(), target_name)

    try:
        conn = sqlite3.connect(chembl_path)

        # Try exact-ish match first
        cursor = conn.execute("""
            SELECT DISTINCT cs.canonical_smiles
            FROM compound_structures cs
            JOIN activities a ON cs.molregno = a.molregno
            JOIN assays ay ON a.assay_id = ay.assay_id
            JOIN target_dictionary td ON ay.tid = td.tid
            WHERE td.pref_name LIKE ?
              AND a.pchembl_value >= 5.0
              AND cs.canonical_smiles IS NOT NULL
              AND length(cs.canonical_smiles) > 5
              AND length(cs.canonical_smiles) < 200
            ORDER BY a.pchembl_value DESC
            LIMIT ?
        """, (f"%{resolved}%", max_results))
        smiles_list = [row[0] for row in cursor.fetchall()]

        # Fallback: try matching on key words from the target name
        if not smiles_list and " " in target_name:
            keywords = [w for w in target_name.split() if len(w) > 3]
            for kw in keywords[:2]:
                cursor = conn.execute("""
                    SELECT DISTINCT cs.canonical_smiles
                    FROM compound_structures cs
                    JOIN activities a ON cs.molregno = a.molregno
                    JOIN assays ay ON a.assay_id = ay.assay_id
                    JOIN target_dictionary td ON ay.tid = td.tid
                    WHERE td.pref_name LIKE ?
                      AND a.pchembl_value >= 5.0
                      AND cs.canonical_smiles IS NOT NULL
                      AND length(cs.canonical_smiles) > 5
                      AND length(cs.canonical_smiles) < 200
                    ORDER BY a.pchembl_value DESC
                    LIMIT ?
                """, (f"%{kw}%", max_results))
                smiles_list = [row[0] for row in cursor.fetchall()]
                if smiles_list:
                    break

        conn.close()
        return smiles_list
    except Exception as e:
        logger.warning("ChEMBL query failed: %s", e)
        return []
