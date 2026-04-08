"""Tests for UniProtAPIConnector, AlphaFoldAPIConnector, GTExAPIConnector.

All HTTP calls are mocked with unittest.mock.patch so these tests run
without network access.  Tests mirror the pattern established in
tests/connectors/test_chembl_api.py.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from connectors.base import BaseConnector, ConnectorResult
from connectors.uniprot_api import UniProtAPIConnector
from connectors.alphafold_api import AlphaFoldAPIConnector
from connectors.gtex_api import GTExAPIConnector
from ontology.evidence import EvidenceItem


# ===========================================================================
# Shared helpers
# ===========================================================================

def _make_response(payload, status: int = 200) -> MagicMock:
    """Create a mock requests.Response with .json() and raise_for_status."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        resp.raise_for_status.return_value = None
    return resp


# ===========================================================================
# UniProtAPIConnector
# ===========================================================================

# ---------------------------------------------------------------------------
# Fixture payloads
# ---------------------------------------------------------------------------

def _uniprot_entry() -> dict:
    """Minimal UniProt JSON entry for TARDBP (Q13148)."""
    return {
        "primaryAccession": "Q13148",
        "genes": [
            {"geneName": {"value": "TARDBP"}}
        ],
        "comments": [
            {
                "commentType": "FUNCTION",
                "texts": [
                    {"value": "DNA and RNA binding protein involved in RNA metabolism."}
                ],
            },
            {
                "commentType": "DISEASE",
                "disease": {
                    "diseaseId": "Amyotrophic lateral sclerosis 10",
                    "description": "A neurodegenerative disease affecting upper and lower motor neurons.",
                },
            },
            {
                "commentType": "SUBCELLULAR_LOCATION",
                "subcellularLocations": [
                    {"location": {"value": "Nucleus"}},
                    {"location": {"value": "Cytoplasm"}},
                ],
            },
        ],
    }


def _uniprot_search_result(entry: dict | None = None) -> dict:
    """Wrap a UniProt entry in a search results envelope."""
    if entry is None:
        entry = _uniprot_entry()
    return {"results": [entry], "totalResults": 1}


def _uniprot_empty_search() -> dict:
    return {"results": [], "totalResults": 0}


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestUniProtInstantiation:
    def test_instantiates_with_no_args(self):
        c = UniProtAPIConnector()
        assert c is not None

    def test_instantiates_with_store(self):
        store = MagicMock()
        c = UniProtAPIConnector(store=store)
        assert c._store is store

    def test_inherits_base_connector(self):
        assert isinstance(UniProtAPIConnector(), BaseConnector)

    def test_base_url_is_uniprot(self):
        c = UniProtAPIConnector()
        assert "rest.uniprot.org" in c.BASE_URL


# ---------------------------------------------------------------------------
# fetch() signature compliance
# ---------------------------------------------------------------------------

class TestUniProtFetchSignature:
    def test_fetch_accepts_gene_kwarg(self):
        c = UniProtAPIConnector()
        result = c.fetch(gene="")
        assert isinstance(result, ConnectorResult)

    def test_fetch_accepts_uniprot_kwarg(self):
        c = UniProtAPIConnector()
        result = c.fetch(uniprot="")
        assert isinstance(result, ConnectorResult)

    def test_fetch_no_args_returns_empty(self):
        c = UniProtAPIConnector()
        result = c.fetch()
        assert result.evidence_items_added == 0
        assert result.errors == []

    def test_fetch_empty_strings_returns_empty(self):
        c = UniProtAPIConnector()
        with patch("requests.get") as mock_get:
            result = c.fetch(gene="", uniprot="")
        mock_get.assert_not_called()
        assert result.evidence_items_added == 0


# ---------------------------------------------------------------------------
# fetch() by accession
# ---------------------------------------------------------------------------

class TestUniProtFetchByAccession:
    def test_fetch_by_accession_returns_evidence_items(self):
        c = UniProtAPIConnector()
        with patch("requests.get", return_value=_make_response(_uniprot_entry())):
            result = c.fetch(uniprot="Q13148")
        assert result.evidence_items_added >= 1
        assert result.errors == []

    def test_fetch_by_accession_produces_function_item(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = UniProtAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_uniprot_entry())):
            c.fetch(uniprot="Q13148")

        ids = [i.id for i in created_items]
        assert any("function" in eid for eid in ids)

    def test_fetch_by_accession_produces_disease_item(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = UniProtAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_uniprot_entry())):
            c.fetch(uniprot="Q13148")

        ids = [i.id for i in created_items]
        assert any("disease" in eid for eid in ids)

    def test_fetch_by_accession_produces_location_item(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = UniProtAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_uniprot_entry())):
            c.fetch(uniprot="Q13148")

        ids = [i.id for i in created_items]
        assert any("location" in eid for eid in ids)

    def test_fetch_by_accession_id_format(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = UniProtAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_uniprot_entry())):
            c.fetch(uniprot="Q13148")

        for item in created_items:
            assert item.id.startswith("evi:uniprot:")
            assert isinstance(item, EvidenceItem)


# ---------------------------------------------------------------------------
# fetch() by gene
# ---------------------------------------------------------------------------

class TestUniProtFetchByGene:
    def test_fetch_by_gene_returns_evidence_items(self):
        c = UniProtAPIConnector()
        with patch("requests.get", return_value=_make_response(_uniprot_search_result())):
            result = c.fetch(gene="TARDBP")
        assert result.evidence_items_added >= 1
        assert result.errors == []

    def test_fetch_by_gene_no_results_returns_empty(self):
        c = UniProtAPIConnector()
        with patch("requests.get", return_value=_make_response(_uniprot_empty_search())):
            result = c.fetch(gene="UNKNOWNGENE999")
        assert result.evidence_items_added == 0
        assert result.errors == []


# ---------------------------------------------------------------------------
# Evidence item properties
# ---------------------------------------------------------------------------

class TestUniProtEvidenceItems:
    def test_items_have_pch_layer(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = UniProtAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_uniprot_entry())):
            c.fetch(uniprot="Q13148")

        for item in created_items:
            assert "pch_layer" in item.body
            assert item.body["pch_layer"] == 1

    def test_items_have_correct_source_system(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = UniProtAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_uniprot_entry())):
            c.fetch(uniprot="Q13148")

        for item in created_items:
            assert item.provenance.source_system.value == "database"
            assert item.provenance.asserted_by == "uniprot_api_connector"

    def test_function_item_claim_contains_gene(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = UniProtAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_uniprot_entry())):
            c.fetch(uniprot="Q13148")

        func_items = [i for i in created_items if "function" in i.id]
        assert len(func_items) >= 1
        assert "TARDBP" in func_items[0].claim or "Q13148" in func_items[0].claim

    def test_store_called_per_item(self):
        store = MagicMock()
        c = UniProtAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_uniprot_entry())):
            result = c.fetch(uniprot="Q13148")

        assert store.upsert_evidence_item.call_count == result.evidence_items_added


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestUniProtErrorHandling:
    def test_http_500_adds_error_returns_zero(self):
        c = UniProtAPIConnector()
        with patch("requests.get", return_value=_make_response({}, 500)):
            result = c.fetch(uniprot="Q13148")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_network_exception_adds_error(self):
        import requests as req_lib
        c = UniProtAPIConnector()
        with patch("requests.get", side_effect=req_lib.ConnectionError("timeout")):
            with patch("time.sleep"):
                result = c.fetch(uniprot="Q13148")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_entry_without_comments_returns_zero(self):
        """An entry with no comments produces no evidence items."""
        c = UniProtAPIConnector()
        bare_entry = {
            "primaryAccession": "Q13148",
            "genes": [{"geneName": {"value": "TARDBP"}}],
            "comments": [],
        }
        with patch("requests.get", return_value=_make_response(bare_entry)):
            result = c.fetch(uniprot="Q13148")
        assert result.evidence_items_added == 0
        assert result.errors == []

    def test_works_without_store(self):
        """Connector counts items even without a store."""
        c = UniProtAPIConnector()
        with patch("requests.get", return_value=_make_response(_uniprot_entry())):
            result = c.fetch(uniprot="Q13148")
        assert result.evidence_items_added >= 1


# ===========================================================================
# AlphaFoldAPIConnector
# ===========================================================================

# ---------------------------------------------------------------------------
# Fixture payloads
# ---------------------------------------------------------------------------

def _alphafold_prediction(plddt: float = 87.5, uniprot: str = "P00441") -> dict:
    """Minimal AlphaFold API prediction entry for SOD1."""
    return {
        "entryId": f"AF-{uniprot}-F1",
        "uniprotAccession": uniprot,
        "globalMetricValue": plddt,
        "uniprotStart": 1,
        "uniprotEnd": 153,
        "pdbUrl": f"https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.pdb",
        "cifUrl": f"https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.cif",
    }


def _alphafold_response(plddt: float = 87.5, uniprot: str = "P00441") -> list:
    """AlphaFold API returns a JSON list."""
    return [_alphafold_prediction(plddt=plddt, uniprot=uniprot)]


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestAlphaFoldInstantiation:
    def test_instantiates_with_no_args(self):
        c = AlphaFoldAPIConnector()
        assert c is not None

    def test_instantiates_with_store(self):
        store = MagicMock()
        c = AlphaFoldAPIConnector(store=store)
        assert c._store is store

    def test_inherits_base_connector(self):
        assert isinstance(AlphaFoldAPIConnector(), BaseConnector)

    def test_base_url_is_alphafold(self):
        c = AlphaFoldAPIConnector()
        assert "alphafold.ebi.ac.uk" in c.BASE_URL


# ---------------------------------------------------------------------------
# fetch() signature compliance
# ---------------------------------------------------------------------------

class TestAlphaFoldFetchSignature:
    def test_fetch_accepts_uniprot_kwarg(self):
        c = AlphaFoldAPIConnector()
        result = c.fetch(uniprot="")
        assert isinstance(result, ConnectorResult)

    def test_fetch_accepts_gene_kwarg(self):
        c = AlphaFoldAPIConnector()
        result = c.fetch(gene="")
        assert isinstance(result, ConnectorResult)

    def test_fetch_no_args_returns_empty(self):
        c = AlphaFoldAPIConnector()
        result = c.fetch()
        assert result.evidence_items_added == 0
        assert result.errors == []

    def test_fetch_gene_only_no_api_call(self):
        """Gene-only is not supported — should return empty without hitting API."""
        c = AlphaFoldAPIConnector()
        with patch("requests.get") as mock_get:
            result = c.fetch(gene="SOD1")
        mock_get.assert_not_called()
        assert result.evidence_items_added == 0

    def test_fetch_empty_uniprot_no_api_call(self):
        c = AlphaFoldAPIConnector()
        with patch("requests.get") as mock_get:
            result = c.fetch(uniprot="")
        mock_get.assert_not_called()
        assert result.evidence_items_added == 0


# ---------------------------------------------------------------------------
# fetch() with valid data
# ---------------------------------------------------------------------------

class TestAlphaFoldFetch:
    def test_fetch_returns_one_evidence_item(self):
        c = AlphaFoldAPIConnector()
        with patch("requests.get", return_value=_make_response(_alphafold_response())):
            result = c.fetch(uniprot="P00441")
        assert result.evidence_items_added == 1
        assert result.errors == []

    def test_fetch_item_has_correct_id(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = AlphaFoldAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_alphafold_response())):
            c.fetch(uniprot="P00441")

        assert len(created_items) == 1
        assert created_items[0].id == "evi:alphafold:p00441"

    def test_fetch_item_has_pch_layer_1(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = AlphaFoldAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_alphafold_response())):
            c.fetch(uniprot="P00441")

        assert created_items[0].body["pch_layer"] == 1

    def test_fetch_item_contains_plddt(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = AlphaFoldAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_alphafold_response(plddt=87.5))):
            c.fetch(uniprot="P00441")

        item = created_items[0]
        assert item.body["global_plddt"] == pytest.approx(87.5)
        assert "87.5" in item.claim

    def test_fetch_high_plddt_strong_strength(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = AlphaFoldAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_alphafold_response(plddt=92.0))):
            c.fetch(uniprot="P00441")

        from ontology.enums import EvidenceStrength
        assert created_items[0].strength == EvidenceStrength.strong

    def test_fetch_low_plddt_emerging_strength(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = AlphaFoldAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_alphafold_response(plddt=35.0))):
            c.fetch(uniprot="P00441")

        from ontology.enums import EvidenceStrength
        assert created_items[0].strength == EvidenceStrength.emerging

    def test_fetch_item_source_system_database(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = AlphaFoldAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_alphafold_response())):
            c.fetch(uniprot="P00441")

        item = created_items[0]
        assert item.provenance.source_system.value == "database"
        assert item.provenance.asserted_by == "alphafold_api_connector"

    def test_fetch_empty_list_response_returns_zero(self):
        c = AlphaFoldAPIConnector()
        with patch("requests.get", return_value=_make_response([])):
            result = c.fetch(uniprot="P00441")
        assert result.evidence_items_added == 0
        assert result.errors == []

    def test_works_without_store(self):
        c = AlphaFoldAPIConnector()
        with patch("requests.get", return_value=_make_response(_alphafold_response())):
            result = c.fetch(uniprot="P00441")
        assert result.evidence_items_added == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestAlphaFoldErrorHandling:
    def test_http_404_adds_error_returns_zero(self):
        c = AlphaFoldAPIConnector()
        with patch("requests.get", return_value=_make_response({}, 404)):
            result = c.fetch(uniprot="P00000")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_network_exception_adds_error(self):
        import requests as req_lib
        c = AlphaFoldAPIConnector()
        with patch("requests.get", side_effect=req_lib.ConnectionError("timeout")):
            with patch("time.sleep"):
                result = c.fetch(uniprot="P00441")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_retries_exhausted_adds_error(self):
        c = AlphaFoldAPIConnector()
        with patch("requests.get", side_effect=Exception("always fails")):
            with patch("time.sleep"):
                result = c.fetch(uniprot="P00441")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0


# ===========================================================================
# GTExAPIConnector
# ===========================================================================

# ---------------------------------------------------------------------------
# Fixture payloads
# ---------------------------------------------------------------------------

_ALL_ALS_TISSUES = [
    "Brain_Spinal_cord_cervical_c-1",
    "Brain_Frontal_Cortex_BA9",
    "Brain_Cortex",
    "Nerve_Tibial",
    "Muscle_Skeletal",
    "Whole_Blood",
]

_NON_ALS_TISSUE = "Skin_Sun_Exposed_Lower_leg"


def _gtex_response(
    gene: str = "TARDBP",
    tissues: list[str] | None = None,
    include_non_als: bool = False,
) -> dict:
    """Build a GTEx /medianGeneExpression API response payload."""
    if tissues is None:
        tissues = _ALL_ALS_TISSUES
    records = []
    for i, tissue in enumerate(tissues):
        records.append({
            "gencodeId": gene,
            "tissueSiteDetailId": tissue,
            "median": float(i * 5 + 1),   # varying expression
            "unit": "TPM",
        })
    if include_non_als:
        records.append({
            "gencodeId": gene,
            "tissueSiteDetailId": _NON_ALS_TISSUE,
            "median": 42.0,
            "unit": "TPM",
        })
    return {"data": records}


def _gtex_empty_response() -> dict:
    return {"data": []}


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestGTExInstantiation:
    def test_instantiates_with_no_args(self):
        c = GTExAPIConnector()
        assert c is not None

    def test_instantiates_with_store(self):
        store = MagicMock()
        c = GTExAPIConnector(store=store)
        assert c._store is store

    def test_inherits_base_connector(self):
        assert isinstance(GTExAPIConnector(), BaseConnector)

    def test_base_url_is_gtex(self):
        c = GTExAPIConnector()
        assert "gtexportal.org" in c.BASE_URL


# ---------------------------------------------------------------------------
# fetch() signature compliance
# ---------------------------------------------------------------------------

class TestGTExFetchSignature:
    def test_fetch_accepts_gene_kwarg(self):
        c = GTExAPIConnector()
        result = c.fetch(gene="")
        assert isinstance(result, ConnectorResult)

    def test_fetch_accepts_uniprot_kwarg(self):
        c = GTExAPIConnector()
        result = c.fetch(uniprot="")
        assert isinstance(result, ConnectorResult)

    def test_fetch_no_args_returns_empty(self):
        c = GTExAPIConnector()
        result = c.fetch()
        assert result.evidence_items_added == 0
        assert result.errors == []

    def test_fetch_empty_gene_no_api_call(self):
        c = GTExAPIConnector()
        with patch("requests.get") as mock_get:
            result = c.fetch(gene="")
        mock_get.assert_not_called()
        assert result.evidence_items_added == 0

    def test_fetch_uniprot_only_no_api_call(self):
        """GTEx is gene-centric; uniprot alone (no gene) returns empty."""
        c = GTExAPIConnector()
        with patch("requests.get") as mock_get:
            result = c.fetch(uniprot="Q13148")
        mock_get.assert_not_called()
        assert result.evidence_items_added == 0


# ---------------------------------------------------------------------------
# fetch() with valid data
# ---------------------------------------------------------------------------

class TestGTExFetch:
    def test_fetch_returns_als_relevant_tissue_items(self):
        c = GTExAPIConnector()
        with patch("requests.get", return_value=_make_response(_gtex_response())):
            result = c.fetch(gene="TARDBP")
        assert result.evidence_items_added == len(_ALL_ALS_TISSUES)
        assert result.errors == []

    def test_fetch_filters_non_als_tissues(self):
        """Non-ALS tissues should not produce evidence items."""
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = GTExAPIConnector(store=store)

        with patch(
            "requests.get",
            return_value=_make_response(_gtex_response(include_non_als=True))
        ):
            result = c.fetch(gene="TARDBP")

        # Should only have items for ALS tissues, not Skin
        ids = [i.id for i in created_items]
        assert not any(_NON_ALS_TISSUE.lower() in eid for eid in ids)
        assert result.evidence_items_added == len(_ALL_ALS_TISSUES)

    def test_fetch_item_id_format(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = GTExAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_gtex_response())):
            c.fetch(gene="TARDBP")

        for item in created_items:
            assert item.id.startswith("evi:gtex:")
            assert "tardbp" in item.id
            assert isinstance(item, EvidenceItem)

    def test_fetch_item_has_pch_layer_1(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = GTExAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_gtex_response())):
            c.fetch(gene="TARDBP")

        for item in created_items:
            assert item.body["pch_layer"] == 1

    def test_fetch_item_contains_tpm(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = GTExAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_gtex_response())):
            c.fetch(gene="TARDBP")

        for item in created_items:
            assert "median_tpm" in item.body
            assert item.body["median_tpm"] >= 0.0

    def test_fetch_high_tpm_strong_strength(self):
        """Tissue with median_tpm >= 10 should yield strong evidence."""
        from ontology.enums import EvidenceStrength
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = GTExAPIConnector(store=store)

        high_payload = {
            "data": [
                {
                    "gencodeId": "TARDBP",
                    "tissueSiteDetailId": "Brain_Spinal_cord_cervical_c-1",
                    "median": 25.0,
                    "unit": "TPM",
                }
            ]
        }
        with patch("requests.get", return_value=_make_response(high_payload)):
            c.fetch(gene="TARDBP")

        assert created_items[0].strength == EvidenceStrength.strong

    def test_fetch_zero_tpm_emerging_strength(self):
        """Tissue with median_tpm == 0 should yield emerging evidence."""
        from ontology.enums import EvidenceStrength
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = GTExAPIConnector(store=store)

        zero_payload = {
            "data": [
                {
                    "gencodeId": "TARDBP",
                    "tissueSiteDetailId": "Brain_Spinal_cord_cervical_c-1",
                    "median": 0.0,
                    "unit": "TPM",
                }
            ]
        }
        with patch("requests.get", return_value=_make_response(zero_payload)):
            c.fetch(gene="TARDBP")

        assert created_items[0].strength == EvidenceStrength.emerging

    def test_fetch_item_source_system_database(self):
        created_items: list[EvidenceItem] = []
        store = MagicMock()
        store.upsert_evidence_item.side_effect = created_items.append
        c = GTExAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_gtex_response())):
            c.fetch(gene="TARDBP")

        for item in created_items:
            assert item.provenance.source_system.value == "database"
            assert item.provenance.asserted_by == "gtex_api_connector"

    def test_fetch_empty_response_returns_zero(self):
        c = GTExAPIConnector()
        with patch("requests.get", return_value=_make_response(_gtex_empty_response())):
            result = c.fetch(gene="UNKNOWNGENE999")
        assert result.evidence_items_added == 0
        assert result.errors == []

    def test_store_called_per_item(self):
        store = MagicMock()
        c = GTExAPIConnector(store=store)

        with patch("requests.get", return_value=_make_response(_gtex_response())):
            result = c.fetch(gene="TARDBP")

        assert store.upsert_evidence_item.call_count == result.evidence_items_added

    def test_works_without_store(self):
        c = GTExAPIConnector()
        with patch("requests.get", return_value=_make_response(_gtex_response())):
            result = c.fetch(gene="TARDBP")
        assert result.evidence_items_added == len(_ALL_ALS_TISSUES)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestGTExErrorHandling:
    def test_http_500_adds_error_returns_zero(self):
        c = GTExAPIConnector()
        with patch("requests.get", return_value=_make_response({}, 500)):
            result = c.fetch(gene="TARDBP")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_network_exception_adds_error(self):
        import requests as req_lib
        c = GTExAPIConnector()
        with patch("requests.get", side_effect=req_lib.ConnectionError("timeout")):
            with patch("time.sleep"):
                result = c.fetch(gene="TARDBP")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_retries_exhausted_adds_error(self):
        c = GTExAPIConnector()
        with patch("requests.get", side_effect=Exception("always fails")):
            with patch("time.sleep"):
                result = c.fetch(gene="TARDBP")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_dataset_id_in_request(self):
        """Verify datasetId=gtex_v8 is passed to the API."""
        c = GTExAPIConnector()
        call_kwargs_list = []

        def capture_get(url, **kwargs):
            call_kwargs_list.append(kwargs)
            return _make_response(_gtex_response())

        with patch("requests.get", side_effect=capture_get):
            c.fetch(gene="TARDBP")

        assert len(call_kwargs_list) >= 1
        params = call_kwargs_list[0].get("params", {})
        assert params.get("datasetId") == "gtex_v8"
