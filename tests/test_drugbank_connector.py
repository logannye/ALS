"""Tests for DrugBankConnector — uses fixture XML, no real DrugBank file needed."""
import pytest
import xml.etree.ElementTree as ET

from connectors.drugbank import DrugBankConnector, _parse_drug_entry


# ---------------------------------------------------------------------------
# Fixture XML: a minimal but realistic DrugBank XML drug entry
# ---------------------------------------------------------------------------

DRUGBANK_NS = "http://www.drugbank.ca"

SAMPLE_DRUG_XML = f"""\
<drugbank xmlns="{DRUGBANK_NS}" version="5.1">
  <drug type="small molecule" created="2005-06-13" updated="2024-01-15">
    <drugbank-id primary="true">DB00316</drugbank-id>
    <name>Riluzole</name>
    <description>Riluzole is a benzothiazole derivative used to treat amyotrophic lateral sclerosis (ALS) by reducing glutamate excitotoxicity.</description>
    <indication>Treatment of amyotrophic lateral sclerosis (ALS) to extend survival.</indication>
    <mechanism-of-action>Riluzole inhibits glutamate release, blocks voltage-gated sodium channels, and activates G-protein-dependent signalling processes to reduce motor neuron excitotoxicity in amyotrophic lateral sclerosis.</mechanism-of-action>
    <groups>
      <group>approved</group>
      <group>investigational</group>
    </groups>
    <targets>
      <target>
        <id>BE0000520</id>
        <name>Sodium channel protein type 8 subunit alpha</name>
        <organism>Humans</organism>
        <actions>
          <action>inhibitor</action>
        </actions>
        <polypeptide id="Q9UQD0" source="Swiss-Prot">
          <name>Sodium channel protein type 8 subunit alpha</name>
          <gene-name>SCN8A</gene-name>
        </polypeptide>
      </target>
    </targets>
    <drug-interactions>
      <drug-interaction>
        <drugbank-id>DB00275</drugbank-id>
        <name>Olmesartan</name>
        <description>Olmesartan may increase the hepatotoxic activities of Riluzole.</description>
      </drug-interaction>
    </drug-interactions>
  </drug>
</drugbank>
"""

SAMPLE_DRUG_ALS_KEYWORDS_XML = f"""\
<drugbank xmlns="{DRUGBANK_NS}" version="5.1">
  <drug type="small molecule" created="2020-01-01" updated="2024-01-01">
    <drugbank-id primary="true">DB12345</drugbank-id>
    <name>NeuroprotectX</name>
    <description>A neuroprotect agent being studied for neurodegeneration.</description>
    <indication>Motor neuron disease studies in preclinical models.</indication>
    <mechanism-of-action>Inhibits TDP-43 aggregation and reduces oxidative stress in motor neurons.</mechanism-of-action>
    <groups>
      <group>investigational</group>
    </groups>
    <targets/>
    <drug-interactions/>
  </drug>
</drugbank>
"""

SAMPLE_DRUG_UNRELATED_XML = f"""\
<drugbank xmlns="{DRUGBANK_NS}" version="5.1">
  <drug type="small molecule" created="2010-01-01" updated="2024-01-01">
    <drugbank-id primary="true">DB99999</drugbank-id>
    <name>Aspirin</name>
    <description>Aspirin is a nonsteroidal anti-inflammatory drug used for pain relief.</description>
    <indication>Treatment of pain, fever, and inflammation.</indication>
    <mechanism-of-action>Irreversibly inhibits COX-1 and COX-2 enzymes.</mechanism-of-action>
    <groups>
      <group>approved</group>
    </groups>
    <targets/>
    <drug-interactions/>
  </drug>
</drugbank>
"""

MULTI_DRUG_XML = f"""\
<drugbank xmlns="{DRUGBANK_NS}" version="5.1">
  <drug type="small molecule" created="2005-06-13" updated="2024-01-15">
    <drugbank-id primary="true">DB00316</drugbank-id>
    <name>Riluzole</name>
    <description>Treatment of amyotrophic lateral sclerosis (ALS).</description>
    <indication>ALS indication text here.</indication>
    <mechanism-of-action>Reduces glutamate excitotoxicity in motor neurons.</mechanism-of-action>
    <groups><group>approved</group></groups>
    <targets/>
    <drug-interactions/>
  </drug>
  <drug type="small molecule" created="2010-01-01" updated="2024-01-01">
    <drugbank-id primary="true">DB99999</drugbank-id>
    <name>Aspirin</name>
    <description>Pain reliever used for inflammation and fever.</description>
    <indication>Pain and fever relief.</indication>
    <mechanism-of-action>Inhibits COX enzymes irreversibly.</mechanism-of-action>
    <groups><group>approved</group></groups>
    <targets/>
    <drug-interactions/>
  </drug>
</drugbank>
"""


# ---------------------------------------------------------------------------
# Helpers: parse single drug element from fixture XML
# ---------------------------------------------------------------------------

def _get_drug_element(xml_str: str):
    """Return the first <drug> element and namespace dict from XML string."""
    root = ET.fromstring(xml_str)
    ns = {"db": DRUGBANK_NS}
    drug_el = root.find("db:drug", ns)
    return drug_el, ns


# ---------------------------------------------------------------------------
# _parse_drug_entry: unit tests
# ---------------------------------------------------------------------------

def test_parse_drug_entry_id():
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert intervention.id == "int:drugbank:DB00316"


def test_parse_drug_entry_name():
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert intervention.name == "Riluzole"


def test_parse_drug_entry_class():
    from ontology.enums import InterventionClass
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert intervention.intervention_class == InterventionClass.drug


def test_parse_drug_entry_route_empty():
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert intervention.route == ""


def test_parse_drug_entry_targets():
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert len(intervention.targets) > 0
    assert "Sodium channel protein type 8 subunit alpha" in intervention.targets


def test_parse_drug_entry_provenance():
    from ontology.enums import SourceSystem
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert intervention.provenance.source_system == SourceSystem.database
    assert intervention.provenance.asserted_by == "drugbank_connector"


def test_parse_drug_entry_body_drugbank_id():
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert intervention.body["drugbank_id"] == "DB00316"


def test_parse_drug_entry_body_regulatory_status_approved():
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert intervention.body["regulatory_status"] == "approved"


def test_parse_drug_entry_body_regulatory_status_investigational():
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_ALS_KEYWORDS_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert intervention.body["regulatory_status"] == "investigational"


def test_parse_drug_entry_body_groups():
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert "approved" in intervention.body["groups"]
    assert "investigational" in intervention.body["groups"]


def test_parse_drug_entry_body_description_truncated():
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert len(intervention.body["description"]) <= 300


def test_parse_drug_entry_body_indication_truncated():
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert len(intervention.body["indication"]) <= 300


def test_parse_drug_entry_body_moa_truncated():
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert len(intervention.body["mechanism_of_action"]) <= 500


def test_parse_drug_entry_body_moa_content():
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert "glutamate" in intervention.body["mechanism_of_action"].lower()


def test_parse_drug_entry_body_applicable_subtypes():
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert "sporadic_tdp43" in intervention.body["applicable_subtypes"]
    assert "unresolved" in intervention.body["applicable_subtypes"]


def test_parse_drug_entry_body_drug_interactions():
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    interactions = intervention.body["drug_interactions"]
    assert isinstance(interactions, list)
    assert len(interactions) == 1
    inter = interactions[0]
    assert inter["drugbank_id"] == "DB00275"
    assert inter["name"] == "Olmesartan"
    assert "hepatotoxic" in inter["description"]


def test_parse_drug_entry_body_drug_interactions_empty():
    drug_el, ns = _get_drug_element(SAMPLE_DRUG_ALS_KEYWORDS_XML)
    intervention = _parse_drug_entry(drug_el, ns)
    assert intervention.body["drug_interactions"] == []


# ---------------------------------------------------------------------------
# Connector instantiation
# ---------------------------------------------------------------------------

def test_connector_instantiates():
    c = DrugBankConnector()
    assert c is not None


def test_connector_inherits_base():
    from connectors.base import BaseConnector
    c = DrugBankConnector()
    assert isinstance(c, BaseConnector)


def test_connector_has_als_keywords():
    c = DrugBankConnector()
    assert "amyotrophic lateral sclerosis" in c.ALS_KEYWORDS
    assert "als" in c.ALS_KEYWORDS
    assert "motor neuron disease" in c.ALS_KEYWORDS
    assert "neurodegeneration" in c.ALS_KEYWORDS
    assert "neuroprotect" in c.ALS_KEYWORDS


def test_connector_fetch_method_exists():
    c = DrugBankConnector()
    assert callable(c.fetch)


# ---------------------------------------------------------------------------
# fetch_als_drugs: graceful handling of missing XML file
# ---------------------------------------------------------------------------

def test_fetch_als_drugs_missing_file_returns_error():
    c = DrugBankConnector()
    result = c.fetch_als_drugs(xml_path="/nonexistent/path/drugbank.xml")
    assert len(result.errors) > 0
    assert result.interventions_added == 0


# ---------------------------------------------------------------------------
# fetch_als_drugs: ALS keyword filtering
# ---------------------------------------------------------------------------

def test_fetch_als_drugs_filters_by_keyword(tmp_path):
    """Riluzole (ALS indication) should be included; Aspirin should not."""
    xml_file = tmp_path / "drugbank.xml"
    xml_file.write_text(MULTI_DRUG_XML, encoding="utf-8")

    c = DrugBankConnector()
    result = c.fetch_als_drugs(xml_path=str(xml_file))

    assert result.interventions_added == 1
    assert len(result.errors) == 0


def test_fetch_als_drugs_neuroprotect_keyword(tmp_path):
    """Drugs with 'neuroprotect' in description should be included."""
    xml_file = tmp_path / "drugbank.xml"
    xml_file.write_text(SAMPLE_DRUG_ALS_KEYWORDS_XML, encoding="utf-8")

    c = DrugBankConnector()
    result = c.fetch_als_drugs(xml_path=str(xml_file))

    assert result.interventions_added == 1


def test_fetch_als_drugs_unrelated_excluded(tmp_path):
    """Aspirin (pain reliever, no ALS keyword) should be excluded."""
    xml_file = tmp_path / "drugbank.xml"
    xml_file.write_text(SAMPLE_DRUG_UNRELATED_XML, encoding="utf-8")

    c = DrugBankConnector()
    result = c.fetch_als_drugs(xml_path=str(xml_file))

    assert result.interventions_added == 0


def test_fetch_als_drugs_returns_connector_result(tmp_path):
    from connectors.base import ConnectorResult
    xml_file = tmp_path / "drugbank.xml"
    xml_file.write_text(SAMPLE_DRUG_XML, encoding="utf-8")

    c = DrugBankConnector()
    result = c.fetch_als_drugs(xml_path=str(xml_file))
    assert isinstance(result, ConnectorResult)


# ---------------------------------------------------------------------------
# fetch_drug_interactions
# ---------------------------------------------------------------------------

def test_fetch_drug_interactions_known_ids(tmp_path):
    """Stored Riluzole should return the Olmesartan interaction."""
    xml_file = tmp_path / "drugbank.xml"
    xml_file.write_text(SAMPLE_DRUG_XML, encoding="utf-8")

    c = DrugBankConnector()
    c.fetch_als_drugs(xml_path=str(xml_file))

    interactions = c.fetch_drug_interactions(["DB00316"])
    assert isinstance(interactions, dict)
    assert "DB00316" in interactions
    assert len(interactions["DB00316"]) >= 1


def test_fetch_drug_interactions_unknown_id(tmp_path):
    """Requesting interactions for an unknown ID returns empty list."""
    xml_file = tmp_path / "drugbank.xml"
    xml_file.write_text(SAMPLE_DRUG_XML, encoding="utf-8")

    c = DrugBankConnector()
    c.fetch_als_drugs(xml_path=str(xml_file))

    interactions = c.fetch_drug_interactions(["DB_UNKNOWN_999"])
    assert interactions.get("DB_UNKNOWN_999", []) == []


def test_fetch_drug_interactions_empty_list():
    """Empty drugbank_ids returns empty dict."""
    c = DrugBankConnector()
    result = c.fetch_drug_interactions([])
    assert result == {}
