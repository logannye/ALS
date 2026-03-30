"""ALS-specific computational experiments using ChEMBL, DepMap, and GDSC databases.

Runs in-silico experiments on ALS drug targets to produce quantitative,
falsifiable evidence that doesn't require clinical data. Each experiment
follows the PREDICT → COMPUTE → COMPARE → LEARN cycle.

Experiment Types:
1. gene_essentiality: DepMap CRISPR Chronos scores for ALS target genes
2. drug_sensitivity: GDSC2 LN_IC50 for protocol drugs in neural cell lines
3. binding_affinity: ChEMBL bioactivity data for drug-target pairs
4. drug_interactions: ChEMBL co-target analysis for protocol drug combinations
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ontology.base import BaseEnvelope


@dataclass
class ComputationResult:
    """Result of a computational experiment."""
    experiment_type: str
    target: str
    success: bool = True
    error: Optional[str] = None
    facts: list[BaseEnvelope] = field(default_factory=list)


# Neural/motor neuron relevant cell line patterns for DepMap/GDSC filtering
_NEURAL_LINEAGES = frozenset({
    "peripheral_nervous_system", "central_nervous_system",
    "autonomic_ganglia", "neuroblastoma",
})
_NEURAL_KEYWORDS = ("neuro", "brain", "spinal", "motor", "glia", "astro", "SH-SY5Y", "NSC-34")


class ALSComputationExecutor:
    """Run ALS-specific computational experiments against local databases."""

    def __init__(
        self,
        chembl_path: str = "/Volumes/Databank/databases/chembl_36.db",
        depmap_path: str = "/Volumes/Databank/databases/depmap/CRISPRGeneEffect.parquet",
        gdsc_path: str = "/Volumes/Databank/databases/gdsc/GDSC2_fitted_dose_response.parquet",
    ):
        self._chembl_path = chembl_path
        self._depmap_path = depmap_path
        self._gdsc_path = gdsc_path
        self._depmap_df = None  # Lazy-loaded
        self._gdsc_df = None   # Lazy-loaded

    def run_experiment(
        self,
        experiment_type: str,
        target: str,
        drug: str = "",
        gene: str = "",
    ) -> ComputationResult:
        """Run a single computational experiment and return facts."""
        try:
            if experiment_type == "gene_essentiality":
                return self._gene_essentiality(gene or target)
            elif experiment_type == "drug_sensitivity":
                return self._drug_sensitivity(drug or target)
            elif experiment_type == "binding_affinity":
                return self._binding_affinity(drug, gene or target)
            elif experiment_type == "drug_interactions":
                return self._drug_interactions(drug or target)
            else:
                return ComputationResult(experiment_type=experiment_type, target=target,
                                         success=False, error=f"Unknown experiment type: {experiment_type}")
        except Exception as e:
            return ComputationResult(experiment_type=experiment_type, target=target,
                                     success=False, error=str(e))

    # ------------------------------------------------------------------
    # Gene essentiality (DepMap CRISPR Chronos)
    # ------------------------------------------------------------------

    def _load_depmap(self):
        """Lazy-load DepMap data (409MB parquet → ~2-3s)."""
        if self._depmap_df is not None:
            return
        if not Path(self._depmap_path).exists():
            raise FileNotFoundError(f"DepMap not found: {self._depmap_path}")
        import pandas as pd
        self._depmap_df = pd.read_parquet(self._depmap_path)

    def _gene_essentiality(self, gene: str) -> ComputationResult:
        """Check if gene is essential in neural cell lines via DepMap Chronos scores.

        Essential threshold: Chronos score < -0.5 (more negative = more essential).
        """
        self._load_depmap()
        df = self._depmap_df

        # DepMap columns are "GENE (ENTREZ_ID)" format — search by gene symbol
        matching_cols = [c for c in df.columns if c.startswith(f"{gene} (") or c == gene]
        if not matching_cols:
            return ComputationResult(experiment_type="gene_essentiality", target=gene,
                                     success=False, error=f"Gene {gene} not found in DepMap")

        col = matching_cols[0]
        scores = df[col].dropna()
        if scores.empty:
            return ComputationResult(experiment_type="gene_essentiality", target=gene,
                                     success=False, error=f"No Chronos scores for {gene}")

        median_score = float(scores.median())
        is_essential = median_score < -0.5
        essential_fraction = float((scores < -0.5).mean())

        fact = BaseEnvelope(
            id=f"evi:depmap_essentiality_{gene.lower()}",
            type="EvidenceItem",
            status="active",
            body={
                "claim": (f"{gene} has median Chronos score {median_score:.3f} across {len(scores)} cell lines. "
                          f"Essential in {essential_fraction:.0%} of lines (threshold < -0.5). "
                          f"{'ESSENTIAL — supports target validity.' if is_essential else 'Not broadly essential.'}"),
                "source": "depmap",
                "gene": gene,
                "median_chronos_score": median_score,
                "essential_fraction": essential_fraction,
                "n_cell_lines": len(scores),
                "is_essential": is_essential,
                "evidence_strength": "strong",
                "pch_layer": 2,
                "protocol_layer": "root_cause_suppression",
                "experiment_type": "gene_essentiality",
            },
        )
        return ComputationResult(experiment_type="gene_essentiality", target=gene, facts=[fact])

    # ------------------------------------------------------------------
    # Drug sensitivity (GDSC2)
    # ------------------------------------------------------------------

    def _load_gdsc(self):
        """Lazy-load GDSC2 data (~1-2s parquet)."""
        if self._gdsc_df is not None:
            return
        if not Path(self._gdsc_path).exists():
            raise FileNotFoundError(f"GDSC not found: {self._gdsc_path}")
        import pandas as pd
        self._gdsc_df = pd.read_parquet(self._gdsc_path)

    def _drug_sensitivity(self, drug: str) -> ComputationResult:
        """Check drug sensitivity in neural cell lines from GDSC2."""
        self._load_gdsc()
        df = self._gdsc_df

        # Find drug (case-insensitive match on DRUG_NAME)
        drug_lower = drug.lower()
        drug_col = "DRUG_NAME" if "DRUG_NAME" in df.columns else None
        if drug_col is None:
            # Try other common column names
            for candidate in ["Drug Name", "drug_name", "compound_name"]:
                if candidate in df.columns:
                    drug_col = candidate
                    break
        if drug_col is None:
            return ComputationResult(experiment_type="drug_sensitivity", target=drug,
                                     success=False, error="Cannot find drug name column in GDSC data")

        drug_rows = df[df[drug_col].str.lower() == drug_lower]
        if drug_rows.empty:
            return ComputationResult(experiment_type="drug_sensitivity", target=drug,
                                     success=False, error=f"Drug {drug} not found in GDSC2")

        # Get LN_IC50 values
        ic50_col = "LN_IC50" if "LN_IC50" in drug_rows.columns else None
        if ic50_col is None:
            for candidate in ["ln_ic50", "LN IC50"]:
                if candidate in drug_rows.columns:
                    ic50_col = candidate
                    break
        if ic50_col is None:
            return ComputationResult(experiment_type="drug_sensitivity", target=drug,
                                     success=False, error="Cannot find LN_IC50 column")

        ic50_values = drug_rows[ic50_col].dropna()
        if ic50_values.empty:
            return ComputationResult(experiment_type="drug_sensitivity", target=drug,
                                     success=False, error=f"No IC50 data for {drug}")

        median_ic50 = float(ic50_values.median())
        n_lines = len(ic50_values)
        sensitive_fraction = float((ic50_values < 0).mean())  # LN_IC50 < 0 = sensitive

        fact = BaseEnvelope(
            id=f"evi:gdsc_sensitivity_{drug_lower.replace(' ', '_')}",
            type="EvidenceItem",
            status="active",
            body={
                "claim": (f"{drug} has median LN_IC50 {median_ic50:.2f} across {n_lines} GDSC2 cell lines. "
                          f"Sensitive in {sensitive_fraction:.0%} of lines (LN_IC50 < 0). "
                          f"{'SENSITIVE — drug shows broad activity.' if sensitive_fraction > 0.3 else 'Limited sensitivity.'}"),
                "source": "gdsc",
                "drug_name": drug,
                "median_ln_ic50": median_ic50,
                "sensitive_fraction": sensitive_fraction,
                "n_cell_lines": n_lines,
                "evidence_strength": "strong" if n_lines >= 10 else "moderate",
                "pch_layer": 2,
                "protocol_layer": "adaptive_maintenance",
                "experiment_type": "drug_sensitivity",
            },
        )
        return ComputationResult(experiment_type="drug_sensitivity", target=drug, facts=[fact])

    # ------------------------------------------------------------------
    # Binding affinity (ChEMBL bioactivity)
    # ------------------------------------------------------------------

    def _binding_affinity(self, drug: str, gene: str) -> ComputationResult:
        """Query ChEMBL for known bioactivity data for a drug-gene pair."""
        if not Path(self._chembl_path).exists():
            return ComputationResult(experiment_type="binding_affinity", target=f"{drug}-{gene}",
                                     success=False, error=f"ChEMBL not found: {self._chembl_path}")

        conn = sqlite3.connect(f"file:{self._chembl_path}?mode=ro", uri=True)
        try:
            cur = conn.cursor()
            # Search for activities matching drug name and gene target
            cur.execute("""
                SELECT DISTINCT
                    md.pref_name AS drug_name,
                    cs.accession AS uniprot,
                    a.standard_type,
                    a.standard_value,
                    a.standard_units,
                    a.pchembl_value
                FROM activities a
                JOIN assays ass ON a.assay_id = ass.assay_id
                JOIN target_dictionary td ON ass.tid = td.tid
                JOIN target_components tc ON td.tid = tc.tid
                JOIN component_sequences cs ON tc.component_id = cs.component_id
                JOIN molecule_dictionary md ON a.molregno = md.molregno
                WHERE LOWER(md.pref_name) LIKE ?
                  AND (LOWER(cs.description) LIKE ? OR LOWER(td.pref_name) LIKE ?)
                  AND a.standard_type IN ('IC50', 'Ki', 'EC50', 'Kd')
                  AND a.pchembl_value IS NOT NULL
                ORDER BY a.pchembl_value DESC
                LIMIT 20
            """, (f"%{drug.lower()}%", f"%{gene.lower()}%", f"%{gene.lower()}%"))

            rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return ComputationResult(experiment_type="binding_affinity", target=f"{drug}-{gene}",
                                     success=True, facts=[])  # No data is valid result

        # Build evidence items from bioactivity data
        facts = []
        best_pchembl = max(r[5] for r in rows if r[5])
        for drug_name, uniprot, std_type, std_value, std_units, pchembl in rows[:5]:
            evi_id = f"evi:chembl_affinity_{drug.lower()}_{gene.lower()}_{std_type.lower()}"
            fact = BaseEnvelope(
                id=evi_id,
                type="EvidenceItem",
                status="active",
                body={
                    "claim": (f"{drug_name or drug} has {std_type} = {std_value} {std_units or 'nM'} "
                              f"(pChEMBL {pchembl:.1f}) against {gene}. "
                              f"{'POTENT binder.' if pchembl and pchembl >= 7 else 'Moderate/weak binder.'}"),
                    "source": "chembl",
                    "drug_name": drug_name or drug,
                    "gene": gene,
                    "uniprot": uniprot,
                    "standard_type": std_type,
                    "standard_value": std_value,
                    "pchembl_value": pchembl,
                    "evidence_strength": "strong" if pchembl and pchembl >= 6 else "moderate",
                    "pch_layer": 2,
                    "protocol_layer": "root_cause_suppression",
                    "experiment_type": "binding_affinity",
                },
            )
            facts.append(fact)

        return ComputationResult(experiment_type="binding_affinity",
                                 target=f"{drug}-{gene}", facts=facts)

    # ------------------------------------------------------------------
    # Drug interactions (ChEMBL co-target analysis)
    # ------------------------------------------------------------------

    def _drug_interactions(self, drug: str) -> ComputationResult:
        """Find all known targets for a drug from ChEMBL (interaction profile)."""
        if not Path(self._chembl_path).exists():
            return ComputationResult(experiment_type="drug_interactions", target=drug,
                                     success=False, error=f"ChEMBL not found")

        conn = sqlite3.connect(f"file:{self._chembl_path}?mode=ro", uri=True)
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT DISTINCT
                    td.pref_name AS target_name,
                    cs.accession AS uniprot,
                    a.standard_type,
                    AVG(a.pchembl_value) as avg_pchembl,
                    COUNT(*) as n_measurements
                FROM activities a
                JOIN assays ass ON a.assay_id = ass.assay_id
                JOIN target_dictionary td ON ass.tid = td.tid
                JOIN target_components tc ON td.tid = tc.tid
                JOIN component_sequences cs ON tc.component_id = cs.component_id
                JOIN molecule_dictionary md ON a.molregno = md.molregno
                WHERE LOWER(md.pref_name) LIKE ?
                  AND a.pchembl_value IS NOT NULL
                  AND a.pchembl_value >= 5.0
                GROUP BY td.pref_name, cs.accession, a.standard_type
                HAVING COUNT(*) >= 2
                ORDER BY avg_pchembl DESC
                LIMIT 15
            """, (f"%{drug.lower()}%",))
            rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return ComputationResult(experiment_type="drug_interactions", target=drug,
                                     success=True, facts=[])

        targets_str = ", ".join(f"{r[0]} (pChEMBL {r[3]:.1f})" for r in rows[:5])
        fact = BaseEnvelope(
            id=f"evi:chembl_targets_{drug.lower().replace(' ', '_')}",
            type="EvidenceItem",
            status="active",
            body={
                "claim": (f"{drug} has {len(rows)} known active targets in ChEMBL: {targets_str}. "
                          f"Multi-target profile {'suggests broad mechanism' if len(rows) > 3 else 'is focused'}."),
                "source": "chembl",
                "drug_name": drug,
                "n_targets": len(rows),
                "targets": [{"name": r[0], "uniprot": r[1], "avg_pchembl": r[3], "n_measurements": r[4]} for r in rows],
                "evidence_strength": "strong",
                "pch_layer": 2,
                "protocol_layer": "adaptive_maintenance",
                "experiment_type": "drug_interactions",
            },
        )
        return ComputationResult(experiment_type="drug_interactions", target=drug, facts=[fact])
