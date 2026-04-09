"""Tests for PubMedConnector — no network, uses fixture XML."""
import pytest
import xml.etree.ElementTree as ET

from connectors.pubmed import PubMedConnector, _parse_pubmed_article, _infer_strength_from_modality


# ---------------------------------------------------------------------------
# Fixture: realistic PubmedArticle XML element
# ---------------------------------------------------------------------------

SAMPLE_ARTICLE_XML = """\
<PubmedArticle>
  <MedlineCitation>
    <PMID Version="1">39012345</PMID>
    <Article PubModel="Print-Electronic">
      <Journal>
        <Title>Nature Neuroscience</Title>
      </Journal>
      <ArticleTitle>TDP-43 aggregation drives motor neuron degeneration in ALS</ArticleTitle>
      <Abstract>
        <AbstractText>Motor neuron loss in amyotrophic lateral sclerosis (ALS) is strongly associated with TDP-43 proteinopathy. Here we show that small-molecule inhibitors of TDP-43 aggregation rescue motor neuron viability in patient-derived iPSC models. These findings suggest a therapeutic strategy targeting the root cause of sporadic ALS. Further investigation into combination approaches with existing riluzole therapy may yield synergistic benefit for patients.</AbstractText>
      </Abstract>
      <PublicationTypeList>
        <PublicationType UI="D016449">Randomized Controlled Trial</PublicationType>
      </PublicationTypeList>
    </Article>
  </MedlineCitation>
</PubmedArticle>
"""

SAMPLE_ARTICLE_NO_ABSTRACT_XML = """\
<PubmedArticle>
  <MedlineCitation>
    <PMID Version="1">39099999</PMID>
    <Article PubModel="Print">
      <Journal>
        <Title>Lancet Neurology</Title>
      </Journal>
      <ArticleTitle>A brief note on ALS biomarkers</ArticleTitle>
      <PublicationTypeList>
        <PublicationType UI="D016454">Review</PublicationType>
      </PublicationTypeList>
    </Article>
  </MedlineCitation>
</PubmedArticle>
"""


@pytest.fixture
def sample_article_el():
    return ET.fromstring(SAMPLE_ARTICLE_XML)


@pytest.fixture
def sample_article_no_abstract_el():
    return ET.fromstring(SAMPLE_ARTICLE_NO_ABSTRACT_XML)


# ---------------------------------------------------------------------------
# Parse tests
# ---------------------------------------------------------------------------

def test_parse_pubmed_article_id(sample_article_el):
    item = _parse_pubmed_article(sample_article_el)
    assert item.id == "evi:pubmed:39012345"


def test_parse_pubmed_article_claim(sample_article_el):
    item = _parse_pubmed_article(sample_article_el)
    assert item.claim == "TDP-43 aggregation drives motor neuron degeneration in ALS"


def test_parse_pubmed_article_source_refs(sample_article_el):
    item = _parse_pubmed_article(sample_article_el)
    assert "pmid:39012345" in item.source_refs


def test_parse_pubmed_article_modality(sample_article_el):
    item = _parse_pubmed_article(sample_article_el)
    assert item.body["modality"] == "randomized_controlled_trial"


def test_parse_pubmed_article_direction(sample_article_el):
    item = _parse_pubmed_article(sample_article_el)
    from ontology.enums import EvidenceDirection
    assert item.direction == EvidenceDirection.insufficient


def test_parse_pubmed_article_strength(sample_article_el):
    """RCT article should now get strong strength (not unknown)."""
    item = _parse_pubmed_article(sample_article_el)
    from ontology.enums import EvidenceStrength
    assert item.strength == EvidenceStrength.strong


def test_parse_pubmed_article_provenance(sample_article_el):
    item = _parse_pubmed_article(sample_article_el)
    from ontology.enums import SourceSystem
    assert item.provenance.source_system == SourceSystem.literature
    assert item.provenance.asserted_by == "pubmed_connector"


def test_parse_pubmed_article_body_fields(sample_article_el):
    item = _parse_pubmed_article(sample_article_el)
    assert item.body["protocol_layer"] == ""
    assert item.body["mechanism_target"] == ""
    assert "sporadic_tdp43" in item.body["applicable_subtypes"]
    assert "unresolved" in item.body["applicable_subtypes"]
    assert item.body["erik_eligible"] is True
    assert item.body["pch_layer"] == 1
    assert item.body["journal"] == "Nature Neuroscience"


def test_parse_pubmed_article_abstract(sample_article_el):
    item = _parse_pubmed_article(sample_article_el)
    abstract = item.body["abstract"]
    assert len(abstract) <= 500
    assert abstract.startswith("Motor neuron loss")


def test_parse_pubmed_article_abstract_truncation(sample_article_el):
    """Abstract longer than 500 chars should be truncated."""
    item = _parse_pubmed_article(sample_article_el)
    assert len(item.body["abstract"]) <= 500


def test_parse_pubmed_article_no_abstract(sample_article_no_abstract_el):
    item = _parse_pubmed_article(sample_article_no_abstract_el)
    assert item.body["abstract"] == ""


def test_parse_pubmed_article_review_modality(sample_article_no_abstract_el):
    item = _parse_pubmed_article(sample_article_no_abstract_el)
    assert item.body["modality"] == "review"


# ---------------------------------------------------------------------------
# Connector instantiation
# ---------------------------------------------------------------------------

def test_connector_instantiates():
    c = PubMedConnector()
    assert c is not None
    assert c.BASE_URL == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def test_connector_has_tool_and_email():
    c = PubMedConnector()
    assert c.TOOL == "erik_als_engine"
    assert c.EMAIL == "logan@galenhealth.ai"


def test_layer_queries_has_five_entries():
    c = PubMedConnector()
    assert len(c.LAYER_QUERIES) == 5


def test_layer_queries_keys():
    c = PubMedConnector()
    from ontology.enums import ProtocolLayer
    expected_keys = {
        ProtocolLayer.root_cause_suppression,
        ProtocolLayer.pathology_reversal,
        ProtocolLayer.circuit_stabilization,
        ProtocolLayer.regeneration_reinnervation,
        ProtocolLayer.adaptive_maintenance,
    }
    assert set(c.LAYER_QUERIES.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Network tests (skip by default)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Evidence strength inference tests
# ---------------------------------------------------------------------------

def test_rct_gets_strong_strength():
    assert _infer_strength_from_modality("randomized_controlled_trial") == "strong"


def test_meta_analysis_gets_strong():
    assert _infer_strength_from_modality("meta_analysis") == "strong"


def test_clinical_trial_gets_moderate():
    assert _infer_strength_from_modality("clinical_trial") == "moderate"


def test_observational_gets_emerging():
    assert _infer_strength_from_modality("observational_study") == "emerging"


def test_review_gets_emerging():
    assert _infer_strength_from_modality("review") == "emerging"


def test_case_report_gets_preclinical():
    assert _infer_strength_from_modality("case_report") == "preclinical"


def test_letter_gets_preclinical():
    assert _infer_strength_from_modality("letter") == "preclinical"


def test_other_gets_unknown():
    assert _infer_strength_from_modality("other") == "unknown"


def test_parse_rct_evidence_strength_body(sample_article_el):
    """RCT article body should include evidence_strength key."""
    item = _parse_pubmed_article(sample_article_el)
    assert item.body["evidence_strength"] == "strong"


def test_parse_review_evidence_strength_body(sample_article_no_abstract_el):
    """Review article body should include evidence_strength = emerging."""
    item = _parse_pubmed_article(sample_article_no_abstract_el)
    assert item.body["evidence_strength"] == "emerging"


def test_parse_review_strength_enum(sample_article_no_abstract_el):
    """Review article should have EvidenceStrength.emerging."""
    item = _parse_pubmed_article(sample_article_no_abstract_el)
    from ontology.enums import EvidenceStrength
    assert item.strength == EvidenceStrength.emerging


# ---------------------------------------------------------------------------
# Network tests (skip by default)
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_fetch_real_pubmed():
    """Integration test — requires network access."""
    c = PubMedConnector()
    result = c.fetch(query="ALS riluzole", max_results=3)
    assert result.evidence_items_added >= 0


@pytest.mark.network
def test_fetch_by_pmids_real():
    """Integration test — requires network access."""
    c = PubMedConnector()
    result = c.fetch_by_pmids(["37821843"])
    assert result.evidence_items_added >= 0
