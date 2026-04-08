"""Tests for GWAS, gnomAD, and HPA API connectors — no real HTTP calls.

All HTTP calls are mocked with unittest.mock.patch so these tests run
without network access.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from connectors.base import BaseConnector, ConnectorResult
from connectors.gwas_api import GWASCatalogAPIConnector, _parse_snp_record
from connectors.gnomad_api import GnomADAPIConnector, _parse_constraint_record
from connectors.hpa_api import HPAAPIConnector, _parse_hpa_record
from ontology.evidence import EvidenceItem


# ---------------------------------------------------------------------------
# Helpers: mock API payloads
# ---------------------------------------------------------------------------


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


def _gwas_snp_payload() -> dict:
    """Simulate GWAS Catalog /singleNucleotidePolymorphisms/search/findByGene response."""
    return {
        "_embedded": {
            "singleNucleotidePolymorphisms": [
                {
                    "rsId": "rs1059872",
                    "chromosome_name": "1",
                    "chromosome_position": 11012344,
                    "associations": {
                        "_embedded": {
                            "associations": [
                                {
                                    "loci": [
                                        {
                                            "strongestRiskAlleles": [
                                                {"riskAlleleName": "rs1059872-T"}
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    },
                },
                {
                    "rsId": "rs57088823",
                    "chromosome_name": "1",
                    "chromosome_position": 11020467,
                    "associations": {},
                },
            ]
        },
        "page": {"size": 20, "totalElements": 2, "number": 0},
    }


def _gnomad_constraint_payload() -> dict:
    """Simulate gnomAD GraphQL response for gene constraint query."""
    return {
        "data": {
            "gene": {
                "gnomad_constraint": {
                    "pLI": 0.9987,
                    "oe_lof": 0.06,
                    "oe_lof_upper": 0.12,
                    "oe_mis": 0.62,
                    "mis_z": 3.5,
                }
            }
        }
    }


def _hpa_profile_payload() -> dict:
    """Simulate HPA JSON response for a gene profile."""
    return {
        "geneName": "TARDBP",
        "geneSummary": "TDP-43 is an RNA binding protein involved in RNA processing.",
        "proteinClasses": [
            {"name": "RNA binding proteins"},
            {"name": "Transcription factors"},
        ],
        "biologicalProcess": [
            {"name": "mRNA processing"},
            {"name": "RNA splicing"},
        ],
        "subcellularLocation": [
            {"name": "Nucleus"},
            {"name": "Cytoplasm"},
        ],
        "uniprotIds": ["Q13148"],
        "chromosome": "1",
    }


# ===========================================================================
# GWASCatalogAPIConnector tests
# ===========================================================================


class TestGWASCatalogAPIConnectorInstantiation:
    def test_instantiates_with_no_args(self):
        c = GWASCatalogAPIConnector()
        assert c is not None

    def test_instantiates_with_store(self):
        store = MagicMock()
        c = GWASCatalogAPIConnector(store=store)
        assert c._store is store

    def test_inherits_base_connector(self):
        c = GWASCatalogAPIConnector()
        assert isinstance(c, BaseConnector)

    def test_fetch_method_exists(self):
        c = GWASCatalogAPIConnector()
        assert callable(c.fetch)

    def test_base_url_is_gwas_catalog(self):
        c = GWASCatalogAPIConnector()
        assert "ebi.ac.uk/gwas" in c.BASE_URL


class TestGWASFetchSignature:
    def test_fetch_accepts_gene_kwarg(self):
        c = GWASCatalogAPIConnector()
        result = c.fetch(gene="")
        assert isinstance(result, ConnectorResult)

    def test_fetch_accepts_uniprot_kwarg(self):
        c = GWASCatalogAPIConnector()
        # uniprot is unused but must not crash
        result = c.fetch(uniprot="Q13148")
        assert isinstance(result, ConnectorResult)

    def test_fetch_returns_connector_result_with_no_args(self):
        c = GWASCatalogAPIConnector()
        result = c.fetch()
        assert isinstance(result, ConnectorResult)

    def test_fetch_empty_gene_returns_empty_result(self):
        c = GWASCatalogAPIConnector()
        with patch("requests.get") as mock_get:
            result = c.fetch(gene="")
        mock_get.assert_not_called()
        assert result.evidence_items_added == 0
        assert result.errors == []


class TestGWASFetchByGene:
    def test_fetch_gene_returns_evidence_items(self):
        """fetch(gene=...) parses GWAS SNP records into EvidenceItems."""
        c = GWASCatalogAPIConnector()
        with patch("requests.get", return_value=_make_response(_gwas_snp_payload())):
            result = c.fetch(gene="TARDBP")
        assert result.evidence_items_added >= 1
        assert result.errors == []

    def test_fetch_gene_returns_correct_count(self):
        """Should create one EvidenceItem per SNP in the response."""
        c = GWASCatalogAPIConnector()
        with patch("requests.get", return_value=_make_response(_gwas_snp_payload())):
            result = c.fetch(gene="TARDBP")
        assert result.evidence_items_added == 2

    def test_fetch_gene_empty_snp_list_returns_zero(self):
        """Empty SNP list should return 0 items, no errors."""
        c = GWASCatalogAPIConnector()
        empty = {"_embedded": {"singleNucleotidePolymorphisms": []}}
        with patch("requests.get", return_value=_make_response(empty)):
            result = c.fetch(gene="UNKNOWNGENE")
        assert result.evidence_items_added == 0
        assert result.errors == []

    def test_fetch_gene_missing_embedded_returns_zero(self):
        """Response without _embedded returns 0 items gracefully."""
        c = GWASCatalogAPIConnector()
        with patch("requests.get", return_value=_make_response({})):
            result = c.fetch(gene="TARDBP")
        assert result.evidence_items_added == 0
        assert result.errors == []


class TestGWASEvidenceItemConstruction:
    def test_evidence_item_id_format(self):
        """EvidenceItems should use 'evi:gwas:{rsid}_{gene}' ID format."""
        c = GWASCatalogAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", return_value=_make_response(_gwas_snp_payload())):
            c.fetch(gene="TARDBP")

        assert len(created_items) >= 1
        for item in created_items:
            assert item.id.startswith("evi:gwas:")
            assert isinstance(item, EvidenceItem)

    def test_evidence_item_gene_in_id(self):
        """EvidenceItem IDs should include the gene name."""
        c = GWASCatalogAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", return_value=_make_response(_gwas_snp_payload())):
            c.fetch(gene="TARDBP")

        for item in created_items:
            assert "tardbp" in item.id.lower()

    def test_evidence_item_has_pch_layer_1(self):
        """GWAS EvidenceItems should have pch_layer=1 in body."""
        c = GWASCatalogAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", return_value=_make_response(_gwas_snp_payload())):
            c.fetch(gene="TARDBP")

        for item in created_items:
            assert item.body.get("pch_layer") == 1

    def test_evidence_item_has_database_source_system(self):
        """EvidenceItems provenance should indicate database source."""
        c = GWASCatalogAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", return_value=_make_response(_gwas_snp_payload())):
            c.fetch(gene="TARDBP")

        for item in created_items:
            assert item.provenance.source_system.value == "database"
            assert item.provenance.asserted_by == "gwas_catalog_api_connector"

    def test_evidence_item_store_called_per_snp(self):
        """store.upsert_evidence_item should be called once per SNP."""
        c = GWASCatalogAPIConnector()
        mock_store = MagicMock()
        c._store = mock_store

        with patch("requests.get", return_value=_make_response(_gwas_snp_payload())):
            result = c.fetch(gene="TARDBP")

        assert mock_store.upsert_evidence_item.call_count == result.evidence_items_added

    def test_works_without_store(self):
        """Connector works without a store — items counted but not persisted."""
        c = GWASCatalogAPIConnector()
        with patch("requests.get", return_value=_make_response(_gwas_snp_payload())):
            result = c.fetch(gene="TARDBP")
        assert result.evidence_items_added >= 1
        assert result.errors == []


class TestGWASHTTPErrorHandling:
    def test_http_500_adds_error_returns_zero(self):
        c = GWASCatalogAPIConnector()
        with patch("requests.get", return_value=_make_response({}, 500)):
            result = c.fetch(gene="TARDBP")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_network_exception_adds_error(self):
        import requests as req_lib
        c = GWASCatalogAPIConnector()
        with patch("requests.get", side_effect=req_lib.ConnectionError("timeout")):
            result = c.fetch(gene="TARDBP")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_exhausted_retries_adds_error(self):
        c = GWASCatalogAPIConnector()
        with patch("requests.get", side_effect=Exception("always fails")):
            with patch("time.sleep"):
                result = c.fetch(gene="TARDBP")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0


class TestParseSnpRecord:
    def test_parse_snp_record_returns_evidence_item(self):
        record = {
            "rsId": "rs1059872",
            "chromosome_name": "1",
            "chromosome_position": 11012344,
        }
        item = _parse_snp_record(record, "TARDBP")
        assert isinstance(item, EvidenceItem)

    def test_parse_snp_record_id_format(self):
        record = {"rsId": "rs1059872", "chromosome_name": "1"}
        item = _parse_snp_record(record, "TARDBP")
        assert item.id == "evi:gwas:rs1059872_tardbp"

    def test_parse_snp_record_no_gene_in_id(self):
        record = {"rsId": "rs1059872"}
        item = _parse_snp_record(record, "")
        assert item.id == "evi:gwas:rs1059872"

    def test_parse_snp_record_claim_contains_rsid(self):
        record = {"rsId": "rs1059872", "chromosome_name": "1", "chromosome_position": 11012344}
        item = _parse_snp_record(record, "TARDBP")
        assert "rs1059872" in item.claim


# ===========================================================================
# GnomADAPIConnector tests
# ===========================================================================


class TestGnomADAPIConnectorInstantiation:
    def test_instantiates_with_no_args(self):
        c = GnomADAPIConnector()
        assert c is not None

    def test_instantiates_with_store(self):
        store = MagicMock()
        c = GnomADAPIConnector(store=store)
        assert c._store is store

    def test_inherits_base_connector(self):
        c = GnomADAPIConnector()
        assert isinstance(c, BaseConnector)

    def test_fetch_method_exists(self):
        c = GnomADAPIConnector()
        assert callable(c.fetch)

    def test_base_url_is_gnomad(self):
        c = GnomADAPIConnector()
        assert "gnomad.broadinstitute.org" in c.BASE_URL


class TestGnomADFetchSignature:
    def test_fetch_accepts_gene_kwarg(self):
        c = GnomADAPIConnector()
        result = c.fetch(gene="")
        assert isinstance(result, ConnectorResult)

    def test_fetch_accepts_uniprot_kwarg(self):
        c = GnomADAPIConnector()
        result = c.fetch(uniprot="Q13148")
        assert isinstance(result, ConnectorResult)

    def test_fetch_returns_connector_result_with_no_args(self):
        c = GnomADAPIConnector()
        result = c.fetch()
        assert isinstance(result, ConnectorResult)

    def test_fetch_empty_gene_returns_empty_result(self):
        c = GnomADAPIConnector()
        with patch("requests.post") as mock_post:
            result = c.fetch(gene="")
        mock_post.assert_not_called()
        assert result.evidence_items_added == 0
        assert result.errors == []


class TestGnomADFetchConstraint:
    def test_fetch_gene_returns_evidence_item(self):
        """fetch(gene=...) parses constraint record into an EvidenceItem."""
        c = GnomADAPIConnector()
        with patch("requests.post", return_value=_make_response(_gnomad_constraint_payload())):
            result = c.fetch(gene="TARDBP")
        assert result.evidence_items_added == 1
        assert result.errors == []

    def test_fetch_gene_no_gene_data_returns_zero(self):
        """Response with null gene returns 0 items, no errors."""
        c = GnomADAPIConnector()
        payload = {"data": {"gene": None}}
        with patch("requests.post", return_value=_make_response(payload)):
            result = c.fetch(gene="UNKNOWNGENE")
        assert result.evidence_items_added == 0
        assert result.errors == []

    def test_fetch_gene_no_constraint_data_returns_zero(self):
        """Response with no constraint block returns 0 items."""
        c = GnomADAPIConnector()
        payload = {"data": {"gene": {"gnomad_constraint": None}}}
        with patch("requests.post", return_value=_make_response(payload)):
            result = c.fetch(gene="TARDBP")
        assert result.evidence_items_added == 0
        assert result.errors == []

    def test_uses_post_not_get(self):
        """gnomAD connector must use POST (GraphQL), not GET."""
        c = GnomADAPIConnector()
        with patch("requests.get") as mock_get, \
             patch("requests.post", return_value=_make_response(_gnomad_constraint_payload())) as mock_post:
            c.fetch(gene="TARDBP")
        mock_get.assert_not_called()
        mock_post.assert_called_once()


class TestGnomADEvidenceItemConstruction:
    def test_evidence_item_id_format(self):
        """EvidenceItem should use 'evi:gnomad:{gene}_constraint' ID format."""
        c = GnomADAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.post", return_value=_make_response(_gnomad_constraint_payload())):
            c.fetch(gene="TARDBP")

        assert len(created_items) == 1
        assert created_items[0].id == "evi:gnomad:tardbp_constraint"
        assert isinstance(created_items[0], EvidenceItem)

    def test_evidence_item_has_pch_layer_1(self):
        """gnomAD EvidenceItems should have pch_layer=1 in body."""
        c = GnomADAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.post", return_value=_make_response(_gnomad_constraint_payload())):
            c.fetch(gene="TARDBP")

        assert created_items[0].body.get("pch_layer") == 1

    def test_evidence_item_has_pli_in_body(self):
        """EvidenceItem body should contain pLI value."""
        c = GnomADAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.post", return_value=_make_response(_gnomad_constraint_payload())):
            c.fetch(gene="TARDBP")

        assert "pLI" in created_items[0].body
        assert created_items[0].body["pLI"] == 0.9987

    def test_evidence_item_has_loeuf_in_body(self):
        """EvidenceItem body should contain LOEUF (oe_lof_upper) value."""
        c = GnomADAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.post", return_value=_make_response(_gnomad_constraint_payload())):
            c.fetch(gene="TARDBP")

        assert "LOEUF" in created_items[0].body
        assert created_items[0].body["LOEUF"] == 0.12

    def test_high_pli_gives_strong_strength(self):
        """pLI > 0.9 should result in EvidenceStrength.strong."""
        c = GnomADAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.post", return_value=_make_response(_gnomad_constraint_payload())):
            c.fetch(gene="TARDBP")

        from ontology.enums import EvidenceStrength
        assert created_items[0].strength == EvidenceStrength.strong

    def test_evidence_item_has_database_source_system(self):
        """EvidenceItems provenance should indicate database source."""
        c = GnomADAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.post", return_value=_make_response(_gnomad_constraint_payload())):
            c.fetch(gene="TARDBP")

        assert created_items[0].provenance.source_system.value == "database"
        assert created_items[0].provenance.asserted_by == "gnomad_api_connector"

    def test_works_without_store(self):
        """Connector works without a store — items counted but not persisted."""
        c = GnomADAPIConnector()
        with patch("requests.post", return_value=_make_response(_gnomad_constraint_payload())):
            result = c.fetch(gene="TARDBP")
        assert result.evidence_items_added == 1
        assert result.errors == []


class TestGnomADHTTPErrorHandling:
    def test_http_500_adds_error_returns_zero(self):
        c = GnomADAPIConnector()
        with patch("requests.post", return_value=_make_response({}, 500)):
            result = c.fetch(gene="TARDBP")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_network_exception_adds_error(self):
        import requests as req_lib
        c = GnomADAPIConnector()
        with patch("requests.post", side_effect=req_lib.ConnectionError("timeout")):
            result = c.fetch(gene="TARDBP")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_exhausted_retries_adds_error(self):
        c = GnomADAPIConnector()
        with patch("requests.post", side_effect=Exception("always fails")):
            with patch("time.sleep"):
                result = c.fetch(gene="TARDBP")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0


class TestParseConstraintRecord:
    def test_parse_constraint_returns_evidence_item(self):
        constraint = {"pLI": 0.99, "oe_lof": 0.05, "oe_lof_upper": 0.10, "oe_mis": 0.6, "mis_z": 3.7}
        item = _parse_constraint_record(constraint, "TARDBP")
        assert isinstance(item, EvidenceItem)

    def test_parse_constraint_id_format(self):
        constraint = {"pLI": 0.99}
        item = _parse_constraint_record(constraint, "TARDBP")
        assert item.id == "evi:gnomad:tardbp_constraint"

    def test_parse_constraint_claim_contains_gene(self):
        constraint = {"pLI": 0.99, "oe_lof_upper": 0.12, "mis_z": 3.5}
        item = _parse_constraint_record(constraint, "TARDBP")
        assert "TARDBP" in item.claim

    def test_parse_constraint_low_pli_gives_emerging_strength(self):
        from ontology.enums import EvidenceStrength
        constraint = {"pLI": 0.1}
        item = _parse_constraint_record(constraint, "FUS")
        assert item.strength == EvidenceStrength.emerging

    def test_parse_constraint_moderate_pli_gives_moderate_strength(self):
        from ontology.enums import EvidenceStrength
        constraint = {"pLI": 0.7}
        item = _parse_constraint_record(constraint, "FUS")
        assert item.strength == EvidenceStrength.moderate


# ===========================================================================
# HPAAPIConnector tests
# ===========================================================================


class TestHPAAPIConnectorInstantiation:
    def test_instantiates_with_no_args(self):
        c = HPAAPIConnector()
        assert c is not None

    def test_instantiates_with_store(self):
        store = MagicMock()
        c = HPAAPIConnector(store=store)
        assert c._store is store

    def test_inherits_base_connector(self):
        c = HPAAPIConnector()
        assert isinstance(c, BaseConnector)

    def test_fetch_method_exists(self):
        c = HPAAPIConnector()
        assert callable(c.fetch)

    def test_base_url_is_hpa(self):
        c = HPAAPIConnector()
        assert "proteinatlas.org" in c.BASE_URL


class TestHPAFetchSignature:
    def test_fetch_accepts_gene_kwarg(self):
        c = HPAAPIConnector()
        result = c.fetch(gene="")
        assert isinstance(result, ConnectorResult)

    def test_fetch_accepts_uniprot_kwarg(self):
        c = HPAAPIConnector()
        result = c.fetch(uniprot="Q13148")
        assert isinstance(result, ConnectorResult)

    def test_fetch_returns_connector_result_with_no_args(self):
        c = HPAAPIConnector()
        result = c.fetch()
        assert isinstance(result, ConnectorResult)

    def test_fetch_empty_gene_returns_empty_result(self):
        c = HPAAPIConnector()
        with patch("requests.get") as mock_get:
            result = c.fetch(gene="")
        mock_get.assert_not_called()
        assert result.evidence_items_added == 0
        assert result.errors == []


class TestHPAFetchProfile:
    def test_fetch_gene_returns_evidence_item(self):
        """fetch(gene=...) parses HPA record into an EvidenceItem."""
        c = HPAAPIConnector()
        with patch("requests.get", return_value=_make_response(_hpa_profile_payload())):
            result = c.fetch(gene="TARDBP")
        assert result.evidence_items_added == 1
        assert result.errors == []

    def test_fetch_gene_empty_response_returns_zero(self):
        """Empty response body returns 0 items."""
        c = HPAAPIConnector()
        with patch("requests.get", return_value=_make_response({})):
            result = c.fetch(gene="TARDBP")
        assert result.evidence_items_added == 0
        assert result.errors == []

    def test_fetch_uses_correct_url_format(self):
        """URL should be {BASE_URL}/{gene}.json."""
        c = HPAAPIConnector()
        urls_called = []

        def capture_get(url, **kwargs):
            urls_called.append(url)
            return _make_response(_hpa_profile_payload())

        with patch("requests.get", side_effect=capture_get):
            c.fetch(gene="TARDBP")

        assert len(urls_called) >= 1
        assert "TARDBP.json" in urls_called[0]


class TestHPAEvidenceItemConstruction:
    def test_evidence_item_id_format(self):
        """EvidenceItem should use 'evi:hpa:{gene}_profile' ID format."""
        c = HPAAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", return_value=_make_response(_hpa_profile_payload())):
            c.fetch(gene="TARDBP")

        assert len(created_items) == 1
        assert created_items[0].id == "evi:hpa:tardbp_profile"
        assert isinstance(created_items[0], EvidenceItem)

    def test_evidence_item_has_pch_layer_1(self):
        """HPA EvidenceItems should have pch_layer=1 in body."""
        c = HPAAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", return_value=_make_response(_hpa_profile_payload())):
            c.fetch(gene="TARDBP")

        assert created_items[0].body.get("pch_layer") == 1

    def test_evidence_item_has_protein_class_in_body(self):
        """EvidenceItem body should contain protein_class."""
        c = HPAAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", return_value=_make_response(_hpa_profile_payload())):
            c.fetch(gene="TARDBP")

        assert "protein_class" in created_items[0].body
        assert "RNA binding proteins" in created_items[0].body["protein_class"]

    def test_evidence_item_has_subcellular_location_in_body(self):
        """EvidenceItem body should contain subcellular_location."""
        c = HPAAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", return_value=_make_response(_hpa_profile_payload())):
            c.fetch(gene="TARDBP")

        assert "subcellular_location" in created_items[0].body
        assert "Nucleus" in created_items[0].body["subcellular_location"]

    def test_evidence_item_has_biological_process_in_body(self):
        """EvidenceItem body should contain biological_process."""
        c = HPAAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", return_value=_make_response(_hpa_profile_payload())):
            c.fetch(gene="TARDBP")

        assert "biological_process" in created_items[0].body
        assert "mRNA processing" in created_items[0].body["biological_process"]

    def test_evidence_item_has_database_source_system(self):
        """EvidenceItems provenance should indicate database source."""
        c = HPAAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", return_value=_make_response(_hpa_profile_payload())):
            c.fetch(gene="TARDBP")

        assert created_items[0].provenance.source_system.value == "database"
        assert created_items[0].provenance.asserted_by == "hpa_api_connector"

    def test_evidence_item_claim_contains_gene_and_protein_class(self):
        """EvidenceItem claim should reference the gene and protein class."""
        c = HPAAPIConnector()
        created_items = []
        mock_store = MagicMock()
        mock_store.upsert_evidence_item.side_effect = lambda item: created_items.append(item)
        c._store = mock_store

        with patch("requests.get", return_value=_make_response(_hpa_profile_payload())):
            c.fetch(gene="TARDBP")

        claim = created_items[0].claim
        assert "TARDBP" in claim
        assert "RNA binding proteins" in claim

    def test_works_without_store(self):
        """Connector works without a store — items counted but not persisted."""
        c = HPAAPIConnector()
        with patch("requests.get", return_value=_make_response(_hpa_profile_payload())):
            result = c.fetch(gene="TARDBP")
        assert result.evidence_items_added == 1
        assert result.errors == []


class TestHPAHTTPErrorHandling:
    def test_http_404_adds_error_returns_zero(self):
        c = HPAAPIConnector()
        with patch("requests.get", return_value=_make_response({}, 404)):
            result = c.fetch(gene="UNKNOWNGENE")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_http_500_adds_error_returns_zero(self):
        c = HPAAPIConnector()
        with patch("requests.get", return_value=_make_response({}, 500)):
            result = c.fetch(gene="TARDBP")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_network_exception_adds_error(self):
        import requests as req_lib
        c = HPAAPIConnector()
        with patch("requests.get", side_effect=req_lib.ConnectionError("timeout")):
            result = c.fetch(gene="TARDBP")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0

    def test_exhausted_retries_adds_error(self):
        c = HPAAPIConnector()
        with patch("requests.get", side_effect=Exception("always fails")):
            with patch("time.sleep"):
                result = c.fetch(gene="TARDBP")
        assert len(result.errors) > 0
        assert result.evidence_items_added == 0


class TestParseHPARecord:
    def test_parse_hpa_record_returns_evidence_item(self):
        data = _hpa_profile_payload()
        item = _parse_hpa_record(data, "TARDBP")
        assert isinstance(item, EvidenceItem)

    def test_parse_hpa_record_id_format(self):
        data = _hpa_profile_payload()
        item = _parse_hpa_record(data, "TARDBP")
        assert item.id == "evi:hpa:tardbp_profile"

    def test_parse_hpa_record_claim_contains_gene(self):
        data = _hpa_profile_payload()
        item = _parse_hpa_record(data, "TARDBP")
        assert "TARDBP" in item.claim

    def test_parse_hpa_record_handles_empty_data(self):
        """Parser should not crash on minimal/empty data."""
        item = _parse_hpa_record({}, "SOD1")
        assert isinstance(item, EvidenceItem)
        assert item.id == "evi:hpa:sod1_profile"

    def test_parse_hpa_record_uniprot_extracted(self):
        data = _hpa_profile_payload()
        item = _parse_hpa_record(data, "TARDBP")
        assert item.body.get("uniprot_id") == "Q13148"
