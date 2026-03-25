# Phase 1B: Evidence Connectors — Design Specification

## 1. Purpose

Build 5 on-demand API connectors that expand the curated evidence seed with live data from PubMed, ClinicalTrials.gov, ChEMBL, OpenTargets, and DrugBank. Each connector produces canonical `EvidenceItem` and `Intervention` objects stored in PostgreSQL via the existing `EvidenceStore`.

## 2. Strategic Decisions

- **On-demand execution**: All connectors are stateless `fetch()` calls, not background daemons. Scheduling deferred to Phase 3 (RL loop).
- **Protocol-first queries**: Default queries are organized by the 5 cure protocol layers, targeting Erik's specific disease programs.
- **Erik eligibility matching**: ClinicalTrials connector includes lightweight eligibility checking against Erik's profile.
- **Connector contract**: Upsert-by-source-ID, exponential backoff (3 retries, 1/2/4s), partial commit with error logging, cross-source intervention dedup by canonical ID.

## 3. Architecture

```
  PubMed E-utilities ──> PubMedConnector ──────┐
  ClinicalTrials.gov v2 ──> TrialConnector ────┤
  ChEMBL 36 (local SQL) ──> ChEMBLConnector ──┤──> EvidenceStore ──> PostgreSQL
  OpenTargets GraphQL ──> OpenTargetsConnector ─┤
  DrugBank XML ──────────> DrugBankConnector ───┘
```

All connectors inherit from `BaseConnector` ABC and produce `ConnectorResult` dataclass.

## 4. BaseConnector Contract

```python
class BaseConnector(ABC):
    MAX_RETRIES = 3
    BACKOFF_SECONDS = [1, 2, 4]

    @abstractmethod
    def fetch(self, **kwargs) -> ConnectorResult: ...

    def _retry_with_backoff(self, fn: Callable, *args, **kwargs) -> Any:
        """Retry fn with exponential backoff. Raises after MAX_RETRIES."""

    def _upsert_results(self, items: list[EvidenceItem], interventions: list[Intervention]) -> ConnectorResult:
        """Upsert items and interventions via EvidenceStore. Partial commit: each item committed individually. Errors logged, not raised."""

@dataclass
class ConnectorResult:
    evidence_items_added: int = 0
    interventions_added: int = 0
    errors: list[str] = field(default_factory=list)
    skipped_duplicates: int = 0
```

**Idempotency**: Source IDs map to canonical IDs deterministically:
- PubMed: `evi:pubmed:{pmid}`
- ClinicalTrials: `evi:trial:{nct_id}`, `int:trial:{nct_id}`
- ChEMBL: `evi:chembl:{molecule_chembl_id}_{target_chembl_id}`
- OpenTargets: `evi:ot:{ensembl_id}_als`
- DrugBank: `int:drugbank:{drugbank_id}`

If an object with the same ID already exists and is `active`, the connector updates `body` and `updated_at` via upsert. **Deviation from parent spec Section 7.1:** The parent spec requires supersedes chains for content changes. Phase 1B uses in-place UPDATE for routine connector refreshes because auto-ingested evidence items are low-confidence and versioning adds complexity without value at this stage. Supersedes chains are reserved for curated evidence (seed items) when substantive changes occur (e.g., trial results update, Erik's genetics arrive). This deviation will be revisited in Phase 2 when evidence quality scoring is implemented.

## 4.1 Provenance and Default Field Mapping

All connectors MUST populate these BaseEnvelope fields:

| Connector | `provenance.source_system` | `provenance.asserted_by` | `direction` default | `uncertainty.confidence` |
|-----------|---------------------------|-------------------------|--------------------|-----------------------|
| PubMed | `literature` | `pubmed_connector` | `insufficient` | `None` |
| ClinicalTrials | `trial` | `trial_connector` | `supports` | `None` |
| ChEMBL | `database` | `chembl_connector` | `supports` | `None` |
| OpenTargets | `database` | `opentargets_connector` | `insufficient` | association_score |
| DrugBank | `database` | `drugbank_connector` | N/A (interventions only) | `None` |

**PubMed** uses `insufficient` because abstract-only ingestion cannot assess directionality.
**ClinicalTrials** uses `supports` because trial hypotheses are framed as supporting the intervention.
**ChEMBL** uses `supports` because bioactivity data supports compound-target engagement.
**OpenTargets** maps `association_score` to `uncertainty.confidence`.

**Connection timeout**: All `urllib.request` calls use `timeout=30` seconds. BaseConnector wraps this.

**Contradiction detection**: Deferred to Phase 2. Connectors do not create contradiction bundles. The parent spec's Section 6.3 requirement is noted and will be implemented when evidence quality scoring exists.

## 5. PubMedConnector

**API**: NCBI E-utilities (`eutils.ncbi.nlm.nih.gov/entrez/eutils/`)
**Auth**: Optional NCBI API key (3 req/s without, 10 req/s with). Key from `erik_config.json` field `ncbi_api_key`.
**Dependencies**: `urllib.request` (stdlib) — no additional packages.

### Methods

`fetch(query: str, max_results: int = 50) -> ConnectorResult`
- ESearch → get PMIDs → EFetch abstracts in XML → parse into EvidenceItems
- Each paper produces one EvidenceItem with these BaseEnvelope fields:
  - `provenance.source_system`: `SourceSystem.literature`
  - `provenance.asserted_by`: `"pubmed_connector"`
  - `direction`: `EvidenceDirection.insufficient` (auto-ingested, not manually assessed)
  - `uncertainty.confidence`: `None` (not assessable from abstract alone)
- And these body fields:
  - `id`: `evi:pubmed:{pmid}`
  - `claim`: paper title
  - `source_refs`: `["pmid:{pmid}"]`
  - `strength`: `unknown` (auto-ingested, not manually curated)
  - `body.modality`: inferred from MeSH terms or publication type (randomized_controlled_trial, review, case_report, etc.)
  - `body.protocol_layer`: best guess from title/abstract keywords
  - `body.mechanism_target`: best guess from title/abstract keywords
  - `body.applicable_subtypes`: `["sporadic_tdp43", "unresolved"]` default for non-subtype-specific
  - `body.erik_eligible`: `true` default for general ALS literature
  - `body.pch_layer`: 1 (associational) default for literature
  - `body.cohort_size`: extracted from abstract if detectable
  - `body.abstract`: first 500 chars of abstract text

`fetch_by_pmids(pmids: list[str]) -> ConnectorResult`
- Direct EFetch for specific PMIDs. Used to enrich seed items with full abstract data.

`fetch_als_treatment_updates() -> ConnectorResult`
- Runs curated queries per protocol layer:
  - Layer A: `"(ALS OR amyotrophic lateral sclerosis) AND (gene therapy OR antisense oligonucleotide OR tofersen OR SOD1 OR TDP-43 intrabody) AND 2025:2026[dp]"`
  - Layer B: `"(ALS) AND (TDP-43 OR cryptic splicing OR STMN2 OR UNC13A OR pridopidine OR sigma-1 receptor OR autophagy) AND 2025:2026[dp]"`
  - Layer C: `"(ALS) AND (excitotoxicity OR riluzole OR masitinib OR neuroinflammation OR complement) AND 2025:2026[dp]"`
  - Layer D: `"(ALS) AND (regeneration OR neurotrophic OR BDNF OR GDNF OR NMJ OR reinnervation) AND 2025:2026[dp]"`
  - Layer E: `"(ALS) AND (neurofilament light OR biomarker OR monitoring OR multidisciplinary care) AND 2025:2026[dp]"`
- Max 20 results per layer query (100 total).

### Rate Limiting
- 0.35s sleep between requests (without API key) or 0.11s (with key)
- Respects NCBI Disclaimer: include `tool=erik_als_engine` and `email` in requests

## 6. ClinicalTrialsConnector

**API**: ClinicalTrials.gov v2 REST (`clinicaltrials.gov/api/v2/studies`)
**Auth**: None required.
**Dependencies**: `urllib.request`, `json` (stdlib).

### Methods

`fetch_active_als_trials(phases: list[str] = ["PHASE2", "PHASE3"]) -> ConnectorResult`
- Query: `query.cond=amyotrophic lateral sclerosis&filter.overallStatus=RECRUITING,NOT_YET_RECRUITING&filter.phase={phases}`
- Pagination: follows `nextPageToken` until exhausted
- Each trial produces:
  - One EvidenceItem (`evi:trial:{nct_id}`): claim = brief title, body includes phase, enrollment, primary outcome, start date, study type
  - One Intervention (`int:trial:{nct_id}`): name = intervention name, class inferred from intervention type, body includes nct_id, sponsor, arm description
- `body.erik_eligible`: result of eligibility check (see Section 8)
- `body.ohio_sites`: list of Ohio study sites (for proximity)
- `body.protocol_layer`: inferred from intervention mechanism keywords

`fetch_trial_details(nct_id: str) -> dict`
- Fetch full study record for a single NCT ID. Returns raw parsed data.

### Rate Limiting
- 1 request per second (ClinicalTrials.gov guideline)

## 7. ChEMBLConnector

**Source**: Local ChEMBL 36 SQLite at `/Volumes/Databank/databases/chembl_36.db`
**Auth**: None (local file).
**Dependencies**: `sqlite3` (stdlib). NOTE: This is the ONE exception to the "never use sqlite3" rule — ChEMBL is a read-only external database, not Erik's operational state.

### Methods

`fetch_bioactivity(uniprot_id: str, activity_type: str = "IC50", max_results: int = 100) -> ConnectorResult`
- Joins `activities`, `assays`, `target_components`, `component_sequences` tables
- Filters: `standard_type = activity_type`, `standard_relation = '='`, `cs.accession = uniprot_id`
- Each active compound → one EvidenceItem:
  - `id`: `evi:chembl:{molecule_chembl_id}_{target_chembl_id}`
  - `claim`: "{compound_name} has {activity_type} = {value} {units} against {target_name}"
  - `body.activity_value`: numeric value
  - `body.activity_units`: nM, uM, etc.
  - `body.molecule_chembl_id`, `body.target_chembl_id`
  - `body.pch_layer`: 2 (interventional — assay data)
  - `body.protocol_layer`: mapped from target via `als_targets.py`

`fetch_compounds_for_target(target_name: str) -> ConnectorResult`
- Looks up UniProt ID from `als_targets.py`, then calls `fetch_bioactivity()`

`fetch_for_priority_targets() -> ConnectorResult`
- Runs `fetch_bioactivity` for all druggable targets in `ALS_TARGETS` where `druggable=True`
- Priority order: SIGMAR1, EAAT2, mTOR, CSF1R, TDP-43 (sporadic-relevant first)

### Notes
- ChEMBL 36 path configurable via `erik_config.json` field `chembl_db_path`
- Read-only connection, no write operations

## 8. Erik Eligibility Engine

Embedded in `ClinicalTrialsConnector`, not a separate module.

```python
ERIK_PROFILE = {
    "age": 67,
    "sex": "male",
    "diagnosis": "ALS",
    "diagnosis_criteria": "gold_coast",
    "onset_region": "lower_limb",
    "alsfrs_r": 43,
    "fvc_percent": 100,
    "disease_duration_months": 14,
    "on_riluzole": True,
    "genetic_status": "pending",
    "comorbidities": ["hypertension", "prediabetes", "cervical_stenosis"],
    "medications": ["riluzole", "amlodipine", "atorvastatin", "ramipril"],
}
```

`check_eligibility(eligibility_criteria: dict) -> str`
- Returns: `"eligible"`, `"ineligible"`, or `"uncertain"`
- Checks: age range, sex, ALSFRS-R minimum, FVC minimum, disease duration, excluded comorbidities, excluded medications
- Conservative: if criteria can't be parsed, returns `"uncertain"` (never false negative)
- If genetic testing results arrive, `ERIK_PROFILE` is updated and all trial eligibility is re-assessed

## 9. OpenTargetsConnector

**API**: GraphQL (`api.platform.opentargets.org/api/v4/graphql`)
**Auth**: None required.
**Dependencies**: `urllib.request`, `json` (stdlib).

### Methods

`fetch_als_targets(min_score: float = 0.1) -> ConnectorResult`
- GraphQL query for disease associations with ALS (EFO_0000253)
- Returns top targets ranked by overall association score
- Each target → one EvidenceItem:
  - `id`: `evi:ot:{ensembl_id}_als`
  - `claim`: "{gene_symbol} is associated with ALS (score={score})"
  - `body.association_score`, `body.genetic_association`, `body.known_drug_count`, `body.tractability`

`fetch_target_tractability(ensembl_id: str) -> dict`
- Returns tractability assessment: small molecule, antibody, other modalities

`fetch_target_drugs(ensembl_id: str) -> list[dict]`
- Returns known drugs targeting this gene, with phase and mechanism

### Rate Limiting
- No published rate limit, but self-limit to 2 req/s

## 10. DrugBankConnector

**Source**: DrugBank XML file (academic download).
**Auth**: Academic license required (free registration at go.drugbank.com).
**Dependencies**: `xml.etree.ElementTree` (stdlib).

### Methods

`fetch_als_drugs(xml_path: str) -> ConnectorResult`
- Parses full DrugBank XML
- Filters drugs by: (a) ALS indication, (b) targets matching `ALS_TARGETS` UniProt IDs
- Each drug → one Intervention:
  - `id`: `int:drugbank:{drugbank_id}`
  - Full MOA, ADMET, pharmacokinetics, interactions in body
  - Cross-references: ChEMBL ID, PubChem CID, KEGG drug ID

`fetch_drug_interactions(drugbank_ids: list[str], xml_path: str) -> list[dict]`
- Returns drug-drug interactions between specified drugs
- Critical for combination protocol safety: riluzole + candidate interactions

### Notes
- DrugBank XML path configurable via `erik_config.json` field `drugbank_xml_path`
- XML parsed once per `fetch_als_drugs()` call (cached in memory for interaction queries)
- If DrugBank XML is not available, connector returns empty result with warning (not error)

## 11. File Structure

```
scripts/
  connectors/
    __init__.py
    base.py                  # BaseConnector ABC + ConnectorResult dataclass
    pubmed.py                # PubMedConnector
    clinical_trials.py       # ClinicalTrialsConnector + ERIK_PROFILE + eligibility checker
    chembl.py                # ChEMBLConnector (local ChEMBL 36 SQL)
    opentargets.py           # OpenTargetsConnector (GraphQL)
    drugbank.py              # DrugBankConnector (XML parser)
tests/
  test_base_connector.py
  test_pubmed_connector.py
  test_clinical_trials_connector.py
  test_chembl_connector.py
  test_opentargets_connector.py
  test_drugbank_connector.py
```

## 12. Config Additions

Add to `data/erik_config.json`:
```json
{
  "ncbi_api_key": null,
  "chembl_db_path": "/Volumes/Databank/databases/chembl_36.db",
  "drugbank_xml_path": null,
  "connector_max_retries": 3,
  "connector_backoff_seconds": [1, 2, 4]
}
```

## 13. Testing Strategy

**Unit tests (no network, no DB):**
- BaseConnector retry logic with mock callables
- PubMed XML parsing with fixture XML
- ClinicalTrials JSON parsing with fixture JSON
- ChEMBL SQL query construction (mock connection)
- OpenTargets GraphQL response parsing with fixture JSON
- DrugBank XML parsing with fixture XML snippet
- Erik eligibility checker with various trial criteria

**Integration tests (network required, marked with `@pytest.mark.network`):**
- PubMed: fetch 5 results for "ALS riluzole" — verify EvidenceItems produced
- ClinicalTrials: fetch active ALS Phase 3 trials — verify NCT IDs present
- OpenTargets: fetch ALS targets — verify SOD1/TARDBP in results

**Integration tests (DB required, use `db_available` fixture):**
- Full pipeline: fetch from connector → upsert to store → query back

**ChEMBL integration (requires local DB, marked with `@pytest.mark.chembl`):**
- Query SIGMAR1 (Q99720) bioactivity — verify results returned

## 14. Success Criteria

Phase 1B is complete when:
1. All 5 connectors produce valid `EvidenceItem`/`Intervention` objects
2. PubMed `fetch_als_treatment_updates()` returns recent ALS papers tagged by protocol layer
3. ClinicalTrials `fetch_active_als_trials()` returns recruiting Phase 2/3 trials with Erik eligibility assessment
4. ChEMBL `fetch_for_priority_targets()` returns bioactive compounds for sporadic-ALS-relevant targets
5. OpenTargets `fetch_als_targets()` returns ranked target associations
6. DrugBank `fetch_als_drugs()` returns drug profiles with interaction data (when XML available)
7. All results flow through EvidenceStore to PostgreSQL
8. Erik's eligibility is assessed against every active trial
9. Unit tests pass without network access
10. Integration tests pass with network access
