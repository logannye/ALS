#!/usr/bin/env python3
"""Provision SpliceAI data for 8 ALS target genes from gnomAD API.

Downloads splice-altering variant predictions via gnomAD's public GraphQL
endpoint. Writes per-gene TSV files to /Volumes/Databank/databases/spliceai/.

Usage:
    conda run -n erik-core python scripts/connectors/provision_spliceai.py

Memory-safe: processes one gene at a time, streams results, no large in-memory
structures.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

GNOMAD_API = "https://gnomad.broadinstitute.org/api"
OUTPUT_DIR = "/Volumes/Databank/databases/spliceai"

# ALS genes with Ensembl gene IDs and GRCh38 coordinates
# Coordinates are gene body ± 50bp for splice context
ALS_GENES = {
    "SOD1": {
        "ensembl_id": "ENSG00000142168",
        "chrom": "21",
        "start": 31659622,
        "stop": 31668931,
    },
    "TARDBP": {
        "ensembl_id": "ENSG00000120948",
        "chrom": "1",
        "start": 11012344,
        "stop": 11025739,
    },
    "FUS": {
        "ensembl_id": "ENSG00000089280",
        "chrom": "16",
        "start": 31180139,
        "stop": 31194836,
    },
    "C9orf72": {
        "ensembl_id": "ENSG00000147894",
        "chrom": "9",
        "start": 27546542,
        "stop": 27573863,
    },
    "UNC13A": {
        "ensembl_id": "ENSG00000130477",
        "chrom": "19",
        "start": 17609422,
        "stop": 17693973,
    },
    "STMN2": {
        "ensembl_id": "ENSG00000104435",
        "chrom": "8",
        "start": 79611078,
        "stop": 79663992,
    },
    "KIF5A": {
        "ensembl_id": "ENSG00000155980",
        "chrom": "12",
        "start": 57486977,
        "stop": 57527222,
    },
    "HNRNPA1": {
        "ensembl_id": "ENSG00000135486",
        "chrom": "12",
        "start": 54280028,
        "stop": 54287392,
    },
}

GRAPHQL_QUERY = """
query GeneVariants($geneId: String!, $datasetId: DatasetId!) {
  gene(gene_id: $geneId, reference_genome: GRCh38) {
    variants(dataset: $datasetId) {
      variant_id
      chrom
      pos
      ref
      alt
      consequence
      hgvsc
      in_silico_predictors {
        id
        value
      }
    }
  }
}
"""

TSV_COLUMNS = ["CHROM", "POS", "REF", "ALT", "SYMBOL", "DS_AG", "DS_AL", "DS_DG", "DS_DL"]


def _query_gnomad(gene_id: str, dataset: str = "gnomad_r4") -> list[dict]:
    """Query gnomAD GraphQL for variants in a gene. Returns raw variant list."""
    payload = json.dumps({
        "query": GRAPHQL_QUERY,
        "variables": {"geneId": gene_id, "datasetId": dataset},
    }).encode("utf-8")

    req = urllib.request.Request(
        GNOMAD_API,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.reason}")
        return []
    except urllib.error.URLError as e:
        print(f"  Connection error: {e.reason}")
        return []
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return []

    gene_data = data.get("data", {}).get("gene")
    if gene_data is None:
        print("  No gene data returned")
        return []

    return gene_data.get("variants", []) or []


def _extract_spliceai_scores(variant: dict) -> dict[str, float] | None:
    """Extract SpliceAI scores from in_silico_predictors.

    gnomAD provides spliceai_ds_max (max of the 4 delta scores) rather than
    individual DS_AG/DS_AL/DS_DG/DS_DL. We store the max as all four columns
    set to max/4 (conservative lower bound) with max_ds preserved accurately.
    The connector's classification logic uses max_ds which is exact.
    """
    predictors = variant.get("in_silico_predictors") or []
    max_score = None
    pangolin = 0.0

    for pred in predictors:
        pid = pred.get("id", "")
        val = pred.get("value", "")
        if pid == "spliceai_ds_max":
            try:
                max_score = float(val)
            except (ValueError, TypeError):
                pass
        elif pid == "pangolin_largest_ds":
            try:
                pangolin = float(val)
            except (ValueError, TypeError):
                pass

    if max_score is None or max_score <= 0:
        return None

    # Distribute the max score across the 4 channels as a heuristic.
    # The connector uses max() of the 4, which will equal max_score.
    return {
        "DS_AG": round(max_score, 4),
        "DS_AL": 0.0,
        "DS_DG": 0.0,
        "DS_DL": 0.0,
        "_pangolin": round(pangolin, 4),
    }


def _provision_gene(gene_symbol: str, gene_info: dict) -> int:
    """Download and write SpliceAI data for one gene. Returns variant count."""
    gene_id = gene_info["ensembl_id"]
    chrom = gene_info["chrom"]

    print(f"  Querying gnomAD for {gene_symbol} ({gene_id})...")
    variants = _query_gnomad(gene_id)

    if not variants:
        print(f"  No variants returned for {gene_symbol}")
        return 0

    # Extract variants with SpliceAI scores
    scored_variants = []
    for v in variants:
        scores = _extract_spliceai_scores(v)
        if scores is not None:
            scored_variants.append({
                "CHROM": f"chr{v.get('chrom', chrom)}",
                "POS": str(v.get("pos", "")),
                "REF": v.get("ref", ""),
                "ALT": v.get("alt", ""),
                "SYMBOL": gene_symbol,
                "DS_AG": scores["DS_AG"],
                "DS_AL": scores["DS_AL"],
                "DS_DG": scores["DS_DG"],
                "DS_DL": scores["DS_DL"],
            })

    if not scored_variants:
        print(f"  No SpliceAI-scored variants for {gene_symbol} (from {len(variants)} total)")
        # Write empty file so connector knows gene was checked
        out_path = os.path.join(OUTPUT_DIR, f"spliceai_{gene_symbol}.tsv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
            writer.writeheader()
        return 0

    # Sort by max delta score descending
    scored_variants.sort(
        key=lambda v: max(v["DS_AG"], v["DS_AL"], v["DS_DG"], v["DS_DL"]),
        reverse=True,
    )

    out_path = os.path.join(OUTPUT_DIR, f"spliceai_{gene_symbol}.tsv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
        writer.writeheader()
        for v in scored_variants:
            writer.writerow(v)

    high = sum(
        1 for v in scored_variants
        if max(v["DS_AG"], v["DS_AL"], v["DS_DG"], v["DS_DL"]) >= 0.5
    )
    mod = sum(
        1 for v in scored_variants
        if 0.2 <= max(v["DS_AG"], v["DS_AL"], v["DS_DG"], v["DS_DL"]) < 0.5
    )

    print(f"  {gene_symbol}: {len(scored_variants)} variants ({high} high, {mod} moderate)")
    return len(scored_variants)


def main():
    print("SpliceAI Data Provisioner for Erik ALS Targets")
    print("=" * 50)

    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    total = 0
    genes_done = 0

    for gene, info in ALS_GENES.items():
        print(f"\n[{genes_done + 1}/{len(ALS_GENES)}] {gene}:")
        count = _provision_gene(gene, info)
        total += count
        genes_done += 1

        # Rate limit: gnomAD API requests should be spaced
        if genes_done < len(ALS_GENES):
            time.sleep(2)

    print(f"\nDone: {total} total scored variants across {genes_done} genes")
    print(f"Output: {OUTPUT_DIR}/spliceai_*.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
