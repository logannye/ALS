"""GEO ALS Connector — disease-state gene expression from ALS patient tissue.

GTEx only has healthy tissue. GEO provides the disease contrast: expression
data from ALS patients, iPSC motor neurons, and SOD1 mouse models. This
connector parses local GEO series matrix files to compute differential
expression (fold change + p-value) for queried genes.

Data directory: /Volumes/Databank/databases/geo_als/{accession}/
Each subdirectory contains a series matrix file downloaded from GEO.
"""
from __future__ import annotations

import glob
import logging
import math
import os
import statistics
from pathlib import Path
from typing import Any

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "/Volumes/Databank/databases/geo_als"

_DEFAULT_DATASETS = {
    "GSE124439": {"title": "ALS vs control spinal cord", "tissue": "spinal_cord", "species": "human", "n_disease": 10, "n_control": 10},
    "GSE153960": {"title": "C9orf72 iPSC motor neurons", "tissue": "motor_neuron_ipsc", "species": "human", "n_disease": 3, "n_control": 3},
    "GSE56808":  {"title": "SOD1-G93A mouse spinal cord", "tissue": "spinal_cord", "species": "mouse", "n_disease": 4, "n_control": 4},
    "GSE68605":  {"title": "ALS frontal cortex", "tissue": "frontal_cortex", "species": "human", "n_disease": 10, "n_control": 10},
    "GSE76220":  {"title": "Motor neuron expression in ALS", "tissue": "motor_neuron", "species": "human", "n_disease": 6, "n_control": 6},
    "GSE139384": {"title": "TDP-43 knockdown neurons", "tissue": "neuron", "species": "human", "n_disease": 3, "n_control": 3},
    "GSE118336": {"title": "ALS blood transcriptome", "tissue": "blood", "species": "human", "n_disease": 10, "n_control": 10},
    "GSE40438":  {"title": "ALS ventral horn LCM", "tissue": "ventral_horn", "species": "human", "n_disease": 4, "n_control": 4},
}


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def _parse_series_matrix(file_path: str) -> tuple[list[str], dict[str, list[float]]]:
    """Parse a GEO series matrix file into sample IDs and probe-level values.

    Parameters
    ----------
    file_path:
        Path to a ``*_series_matrix.txt`` file.

    Returns
    -------
    Tuple of (sample_ids, {probe_id: [values_per_sample]}).
    Lines starting with ``!`` are metadata (skipped).
    The ``"ID_REF"`` line provides sample IDs; subsequent rows are data.
    Values are converted to float; missing/NA become 0.0.
    """
    sample_ids: list[str] = []
    probes: dict[str, list[float]] = {}

    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n\r")

            # Skip metadata lines
            if line.startswith("!"):
                continue

            parts = line.split("\t")
            if not parts:
                continue

            # Header row: "ID_REF" followed by sample IDs
            first = parts[0].strip('"')
            if first == "ID_REF":
                sample_ids = [p.strip('"') for p in parts[1:]]
                continue

            # Data row: "probe_id"\tval\tval\t...
            probe_id = first
            if not probe_id:
                continue

            values: list[float] = []
            for v in parts[1:]:
                v = v.strip().strip('"')
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    values.append(0.0)

            probes[probe_id] = values

    return sample_ids, probes


def _compute_fold_change(
    disease_values: list[float],
    control_values: list[float],
) -> tuple[float, float]:
    """Compute log2 fold change and approximate p-value (Welch's t-test).

    Parameters
    ----------
    disease_values:
        Expression values from disease samples.
    control_values:
        Expression values from control samples.

    Returns
    -------
    Tuple of (log2_fold_change, p_value).
    Returns (0.0, 1.0) if computation is not possible.
    """
    eps = 1e-10

    if len(disease_values) < 2 or len(control_values) < 2:
        return 0.0, 1.0

    mean_d = statistics.mean(disease_values)
    mean_c = statistics.mean(control_values)

    # log2 fold change
    fc = math.log2((mean_d + eps) / (mean_c + eps))

    # Welch's t-test approximation
    try:
        sd_d = statistics.stdev(disease_values)
        sd_c = statistics.stdev(control_values)
    except statistics.StatisticsError:
        return fc, 1.0

    n_d = len(disease_values)
    n_c = len(control_values)

    se = math.sqrt((sd_d ** 2 / n_d) + (sd_c ** 2 / n_c) + eps)
    if se < eps:
        return fc, 1.0

    t_stat = abs(mean_d - mean_c) / se

    # Welch-Satterthwaite degrees of freedom
    num = ((sd_d ** 2 / n_d) + (sd_c ** 2 / n_c)) ** 2
    denom_parts = []
    if n_d > 1 and sd_d > 0:
        denom_parts.append((sd_d ** 2 / n_d) ** 2 / (n_d - 1))
    if n_c > 1 and sd_c > 0:
        denom_parts.append((sd_c ** 2 / n_c) ** 2 / (n_c - 1))
    denom = sum(denom_parts)

    if denom < eps:
        df = max(n_d + n_c - 2, 1)
    else:
        df = max(num / denom, 1)

    # Approximate two-tailed p-value using the normal distribution for large df,
    # or a conservative estimate for small df
    # For simplicity: use exp(-0.5 * t^2) * correction — good enough for ranking
    if t_stat > 10:
        p_value = 1e-20  # extremely significant
    elif df >= 30:
        # Normal approximation
        p_value = 2.0 * math.exp(-0.5 * t_stat ** 2) / math.sqrt(2 * math.pi)
    else:
        # Conservative: scale by df penalty
        p_value = 2.0 * math.exp(-0.5 * t_stat ** 2) / math.sqrt(2 * math.pi)
        p_value *= (30.0 / max(df, 1))  # inflate for small df

    p_value = min(max(p_value, 1e-300), 1.0)
    return fc, p_value


_GPL_ANNOTATION_PATH = os.path.join(_DEFAULT_PATH, "gpl570_gene_probes.tsv")
_GPL_CACHE: dict[str, list[str]] | None = None


def _load_gpl_annotation() -> dict[str, list[str]]:
    """Load the GPL570 gene-to-probe mapping (cached after first call)."""
    global _GPL_CACHE
    if _GPL_CACHE is not None:
        return _GPL_CACHE

    mapping: dict[str, list[str]] = {}
    annot_path = _GPL_ANNOTATION_PATH
    if not os.path.exists(annot_path):
        _GPL_CACHE = mapping
        return mapping

    with open(annot_path, "r") as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                gene = parts[0].upper()
                probe_ids = [p.strip() for p in parts[1].split(",") if p.strip()]
                mapping[gene] = probe_ids

    _GPL_CACHE = mapping
    return mapping


def _find_gene_probe(
    probes: dict[str, list[float]],
    gene: str,
) -> tuple[str, list[float]] | None:
    """Find a probe matching the given gene symbol.

    Uses the GPL570 annotation file to map gene symbols to Affymetrix
    probe IDs, then selects the probe with the highest mean expression
    (most reliable signal). Falls back to direct probe ID match.

    Parameters
    ----------
    probes:
        Dict of {probe_id: [values]} from ``_parse_series_matrix``.
    gene:
        Gene symbol to search for (e.g. ``"TARDBP"``).

    Returns
    -------
    ``(probe_id, values)`` if found, else ``None``.
    """
    gene_upper = gene.upper()

    # Try GPL annotation mapping first
    gpl = _load_gpl_annotation()
    candidate_probes = gpl.get(gene_upper, [])
    best_match: tuple[str, list[float]] | None = None
    best_mean = -1.0

    for probe_id in candidate_probes:
        if probe_id in probes:
            values = probes[probe_id]
            mean_val = sum(values) / max(len(values), 1)
            if mean_val > best_mean:
                best_mean = mean_val
                best_match = (probe_id, values)

    if best_match is not None:
        return best_match

    # Fallback: direct probe ID match (for RNA-seq or non-GPL570 data)
    gene_lower = gene.lower()
    for probe_id, values in probes.items():
        if probe_id.lower() == gene_lower:
            return probe_id, values

    return None


# ---------------------------------------------------------------------------
# GEOALSConnector
# ---------------------------------------------------------------------------

class GEOALSConnector(BaseConnector):
    """Connector for local GEO series matrix files — ALS disease expression.

    Parses pre-downloaded GEO datasets to compute differential expression
    (log2 fold change + p-value) for queried genes in ALS vs control tissue.
    """

    def __init__(
        self,
        store: Any = None,
        data_path: str = _DEFAULT_PATH,
        **kwargs: Any,
    ) -> None:
        self._store = store
        self._data_path = data_path
        self._signature_cache: tuple[list[str], list[str]] | None = None

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(self, *, gene: str = "", **kwargs: Any) -> ConnectorResult:
        """Fetch differential expression evidence for a gene across ALS datasets.

        Parameters
        ----------
        gene:
            Gene symbol to query (e.g. ``"TARDBP"``).
        **kwargs:
            Accepts and ignores ``uniprot=`` and other kwargs.

        Returns
        -------
        ConnectorResult with evidence_items_added count and any errors.
        """
        result = ConnectorResult()

        if not gene:
            return result

        data_dir = Path(self._data_path)
        if not data_dir.exists():
            return result  # data not downloaded yet — no error

        for accession, meta in _DEFAULT_DATASETS.items():
            try:
                acc_dir = data_dir / accession
                if not acc_dir.exists():
                    continue

                # Find series matrix file
                matrix_files = glob.glob(str(acc_dir / "*series_matrix*.txt"))
                if not matrix_files:
                    continue

                sample_ids, probes = _parse_series_matrix(matrix_files[0])
                if not probes:
                    continue

                match = _find_gene_probe(probes, gene)
                if match is None:
                    continue

                probe_id, values = match

                # Split into disease (first N) and control (last N)
                n_disease = meta["n_disease"]
                n_control = meta["n_control"]
                total_expected = n_disease + n_control

                if len(values) < total_expected:
                    # Not enough samples — skip
                    continue

                disease_vals = values[:n_disease]
                control_vals = values[n_disease:n_disease + n_control]

                fc, pval = _compute_fold_change(disease_vals, control_vals)

                # Determine direction and strength
                direction = "up" if fc > 0 else "down"
                significant = pval < 0.05 and abs(fc) > 1.0

                if pval < 0.01 and abs(fc) > 1.5:
                    strength = "strong"
                elif pval < 0.05:
                    strength = "moderate"
                else:
                    strength = "emerging"

                comparison = meta["title"]
                claim = (
                    f"GEO {accession}: {gene} is {direction}regulated in "
                    f"{comparison} (log2FC={fc:.2f}, p={pval:.1e})"
                )

                evi = BaseEnvelope(
                    id=f"evi:geo:{accession.lower()}_{gene.lower()}",
                    type="EvidenceItem",
                    status="active",
                    body={
                        "claim": claim,
                        "data_source": "geo",
                        "accession": accession,
                        "gene": gene,
                        "probe_id": probe_id,
                        "tissue": meta["tissue"],
                        "species": meta["species"],
                        "log2_fold_change": round(fc, 4),
                        "p_value": pval,
                        "direction": direction,
                        "significant": significant,
                        "strength": strength,
                        "pch_layer": 2,
                    },
                )

                if self._store:
                    self._store.upsert_object(evi)
                result.evidence_items_added += 1

            except Exception as e:
                logger.warning("GEO %s failed for gene %s: %s", accession, gene, e)
                result.errors.append(f"GEO {accession} failed for {gene}: {e}")

        return result

    # ------------------------------------------------------------------
    # Public API — disease signature for CMap
    # ------------------------------------------------------------------

    def get_disease_signature(self, top_n: int = 100) -> tuple[list[str], list[str]]:
        """Aggregate disease signature across all datasets.

        Finds genes consistently up- or down-regulated across ALS datasets.
        Results are cached after the first call.

        Parameters
        ----------
        top_n:
            Maximum number of genes to return in each direction.

        Returns
        -------
        Tuple of (up_genes, down_genes) — gene symbols sorted by consistency.
        Returns ([], []) if no data is available.
        """
        if self._signature_cache is not None:
            return self._signature_cache

        data_dir = Path(self._data_path)
        if not data_dir.exists():
            self._signature_cache = ([], [])
            return self._signature_cache

        # Accumulate fold-change votes per gene across datasets
        gene_fc: dict[str, list[float]] = {}

        for accession, meta in _DEFAULT_DATASETS.items():
            try:
                acc_dir = data_dir / accession
                if not acc_dir.exists():
                    continue

                matrix_files = glob.glob(str(acc_dir / "*series_matrix*.txt"))
                if not matrix_files:
                    continue

                sample_ids, probes = _parse_series_matrix(matrix_files[0])
                if not probes:
                    continue

                n_disease = meta["n_disease"]
                n_control = meta["n_control"]
                total_expected = n_disease + n_control

                for probe_id, values in probes.items():
                    if len(values) < total_expected:
                        continue

                    disease_vals = values[:n_disease]
                    control_vals = values[n_disease:n_disease + n_control]
                    fc, pval = _compute_fold_change(disease_vals, control_vals)

                    if pval < 0.05:
                        gene_fc.setdefault(probe_id, []).append(fc)

            except Exception as e:
                logger.debug("Signature scan failed for %s: %s", accession, e)

        # Genes consistently up or down across datasets
        up_genes: list[tuple[str, float]] = []
        down_genes: list[tuple[str, float]] = []

        for gene_id, fcs in gene_fc.items():
            if not fcs:
                continue
            mean_fc = statistics.mean(fcs)
            if all(f > 0 for f in fcs) and mean_fc > 0.5:
                up_genes.append((gene_id, mean_fc))
            elif all(f < 0 for f in fcs) and mean_fc < -0.5:
                down_genes.append((gene_id, abs(mean_fc)))

        # Sort by magnitude (strongest first)
        up_genes.sort(key=lambda x: x[1], reverse=True)
        down_genes.sort(key=lambda x: x[1], reverse=True)

        result = (
            [g for g, _ in up_genes[:top_n]],
            [g for g, _ in down_genes[:top_n]],
        )
        self._signature_cache = result
        return result
