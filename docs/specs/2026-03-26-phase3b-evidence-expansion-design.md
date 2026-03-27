# Phase 3B: Evidence Expansion — Design Specification

## 1. Purpose

Phase 3 built the autonomous research loop with 5 connectors and 10 action types. Phase 3B expands the evidence fabric with 7 new data sources that address critical gaps in pathway-level reasoning, population benchmarking, genetic interpretation, and drug safety — transforming Erik from a literature-aware system into a deeply mechanistic, population-grounded, pharmacogenomically-safe cure engine.

## 2. The 7 New Data Sources

### 2.1 Reactome (Curated Pathway Cascades)

**What it provides:** Peer-curated biological pathways with reaction steps, catalysts, regulators, and compartment annotations. For any protein (by UniProt ID), returns every pathway it participates in and every reaction step in that pathway.

**Why Erik needs it:** Causal chains built by the LLM are only as reliable as the model's training data. Reactome provides verified pathway cascades that serve as ground truth for mechanism links. When the system asserts "sigma-1R activation → ER calcium homeostasis," Reactome can confirm or deny this link with a curated pathway entry.

**API:** Reactome Content Service REST API (`https://reactome.org/ContentService`). Free, no authentication required.
- `GET /data/pathways/low/entity/{UniProtID}` — pathways containing a protein
- `GET /data/pathway/{stableId}/containedEvents` — reaction steps in a pathway

**ID format:** `evi:reactome:{pathway_stable_id}_{uniprot_id}`
**Provenance:** `source_system=database, asserted_by=reactome_connector`

### 2.2 KEGG (Pathway Ontology + Gene-Pathway Mapping)

**What it provides:** Gene-to-pathway mappings, pathway descriptions, and cross-references. Complementary to Reactome — KEGG emphasizes metabolic and signaling contexts while Reactome emphasizes molecular mechanisms.

**Why Erik needs it:** KEGG pathway annotations enable the system to identify off-target effects of kinase inhibitors (e.g., masitinib inhibits CSF1R but also c-Kit → mast cell pathway), and to detect pathway redundancy where multiple interventions hit the same cascade.

**API:** KEGG REST API (`https://rest.kegg.jp`). Free for academic use.
- `GET /link/pathway/hsa:{gene_id}` — pathways for a human gene
- `GET /get/{pathway_id}` — pathway description and gene list

**ID format:** `evi:kegg:{pathway_id}_{gene_symbol}`
**Provenance:** `source_system=database, asserted_by=kegg_connector`

### 2.3 STRING (Protein-Protein Interaction Network)

**What it provides:** Predicted and experimentally validated protein-protein interactions with confidence scores (0-1000). For any protein, returns its interaction partners ranked by evidence strength.

**Why Erik needs it:** ALS is a network disease. The system needs to know whether intervention targets physically interact with disease proteins. STRING answers: "Does sigma-1R (SIGMAR1) interact with TDP-43 (TARDBP)?" with quantitative confidence. This validates or weakens causal chain links and enables discovery of indirect mechanism connections.

**API:** STRING API (`https://string-db.org/api`). Free.
- `GET /api/json/network?identifiers={protein}&species=9606&required_score=400` — interaction network
- Returns: `preferredName_A`, `preferredName_B`, `score`, `experimentally_determined_interaction`, etc.

**ID format:** `evi:string:{gene_a}_{gene_b}`
**Provenance:** `source_system=database, asserted_by=string_connector`

### 2.4 PRO-ACT (ALS Natural History Cohort)

**What it provides:** 10,600+ ALS patient records from 23 Phase 2/3 clinical trials: ALSFRS-R trajectories, demographics, laboratory values, vital signs, survival data. The largest aggregated ALS clinical dataset.

**Why Erik needs it:** Without population data, Erik cannot benchmark his disease progression (-0.39 ALSFRS-R points/month) against similar patients. PRO-ACT enables:
- **Trajectory prediction**: Where will Erik be in 6/12/18 months?
- **Cohort matching**: Find patients with similar onset region, age, baseline ALSFRS-R — what was their trajectory?
- **Treatment effect estimation**: How much did riluzole slow decline in matched cohorts?
- **Urgency quantification**: Is Erik's decline rate fast, average, or slow?

**Data access:** Downloadable CSV from the PRO-ACT website (requires registration). ~500MB. Local storage, no API calls at runtime.

**Integration approach:** Unlike the API connectors, PRO-ACT is a local data loader. The `ProACTAnalyzer` reads the CSV files at startup, builds an in-memory index by demographics, and exposes cohort-matching queries. Results are not upserted as EvidenceItems — they are used directly by the trajectory predictor and state materializer.

**Key tables:** ALSFRS (functional scores), Demographics, Lab values, Vital signs, Treatment (riluzole use), Death (survival)

### 2.5 ClinVar (Genetic Variant Pathogenicity)

**What it provides:** Clinical significance classifications for genetic variants (pathogenic, likely pathogenic, uncertain significance, benign). When Erik's Invitae results arrive, ClinVar tells the system whether each variant is disease-causing.

**Why Erik needs it:** The subtype posterior is currently dominated by "sporadic TDP-43" (0.65) with "C9orf72" at 0.12. Genetic testing can shift this dramatically. ClinVar provides the authoritative pathogenicity assessment that drives the posterior update.

**API:** NCBI E-utilities (same infrastructure as PubMed). Free.
- `GET esearch.fcgi?db=clinvar&term=ALS[disease]+AND+{gene}[gene]` — search ALS variants
- `GET efetch.fcgi?db=clinvar&id={variant_id}` — variant details with clinical significance

**ID format:** `evi:clinvar:{variation_id}`
**Provenance:** `source_system=database, asserted_by=clinvar_connector`

### 2.6 OMIM (Gene-Phenotype Mapping)

**What it provides:** Authoritative gene-phenotype mappings: which genes cause which ALS phenotypes, inheritance patterns, and phenotypic series groupings.

**Why Erik needs it:** When genetics arrive, OMIM maps specific genes to ALS phenotypes (e.g., TARDBP mutations → ALS10, FUS mutations → ALS6). This helps refine subtype inference: a TARDBP variant in a "sporadic" patient shifts the posterior toward TARDBP-driven ALS with specific prognostic implications.

**API:** OMIM API (`https://api.omim.org/api`). Requires free API key (registration at omim.org).
- `GET /api/entry?mimNumber={mim_number}&include=all` — full entry with phenotype map

**ID format:** `evi:omim:{mim_number}`
**Provenance:** `source_system=database, asserted_by=omim_connector`

### 2.7 PharmGKB (Pharmacogenomics + Drug Safety)

**What it provides:** Curated drug-gene interactions, CYP enzyme metabolism pathways, clinical dosing guidelines, and drug interaction severity ratings. Safety-critical pharmacogenomic knowledge.

**Why Erik needs it:** Erik takes riluzole (CYP1A2 substrate) plus amlodipine (CYP3A4), atorvastatin (CYP3A4), and ramipril. Any new intervention added to the protocol must be checked for CYP competition, hepatotoxicity stacking, and pharmacogenomic contraindications. PharmGKB provides this at a depth DrugBank cannot.

**API:** PharmGKB REST API (`https://api.pharmgkb.org/v1/data`). Free for academic use.
- `GET /data/drug?name={drug_name}` — drug info with clinical annotations
- `GET /data/clinicalAnnotation?location.genes.symbol={gene}` — gene-drug annotations

**ID format:** `evi:pharmgkb:{accession_id}`
**Provenance:** `source_system=database, asserted_by=pharmgkb_connector`

## 3. Wiring Into the Research Loop

### 3.1 New Action Types

Add 5 new actions to the `ActionType` enum:

| Action | Description | Connector | When Selected |
|--------|-------------|-----------|---------------|
| `QUERY_PATHWAYS` | Get Reactome + KEGG pathways for a target | Reactome + KEGG | During DEEPEN_CAUSAL_CHAIN when a link needs pathway grounding |
| `QUERY_INTERACTIONS` | Get STRING PPI network for a target | STRING | During hypothesis validation, causal chain extension |
| `MATCH_COHORT` | Find PRO-ACT patients matching Erik | PRO-ACT (local) | After state materialization, for trajectory estimation |
| `INTERPRET_VARIANT` | Check ClinVar + OMIM for a genetic variant | ClinVar + OMIM | When genetic results arrive (manual trigger or config flag) |
| `CHECK_PHARMACOGENOMICS` | Check PharmGKB for drug-gene interactions | PharmGKB | During protocol assembly, interaction checking |

### 3.2 Causal Chain Enhancement

The `DEEPEN_CAUSAL_CHAIN` action is enhanced:
1. Before asking the LLM for the next chain link, query Reactome for known pathway steps
2. If Reactome returns a pathway reaction connecting the current chain endpoint to a known downstream target, use that as the link (confidence = 0.95, evidence_ref = `evi:reactome:...`)
3. If no Reactome pathway exists, fall back to LLM inference (confidence = 0.5-0.7)
4. Validate each link against STRING: does protein A physically interact with protein B? If combined_score > 700, boost confidence by 0.1

### 3.3 Policy Updates

The `select_action` policy gains two new triggers:
- When a protocol intervention has a shallow causal chain AND the target has a UniProt ID → prefer `QUERY_PATHWAYS` before `DEEPEN_CAUSAL_CHAIN`
- When genetics arrive (config flag `genetics_received=true`) → immediately trigger `INTERPRET_VARIANT` for each reported variant

### 3.4 Trajectory Prediction

PRO-ACT data enables a new capability: given Erik's current state, predict his ALSFRS-R trajectory under current treatment. This is not a connector but a new module (`scripts/research/trajectory.py`) that:
1. Loads PRO-ACT ALSFRS data
2. Finds patients matching Erik's demographics (age 60-70, male, limb-onset, baseline ALSFRS-R 40-46)
3. Returns median trajectory, 25th/75th percentile bounds, and median survival

This feeds into the state materializer's `ReversibilityWindowEstimate` and the protocol assembler's timing decisions.

## 4. File Structure

```
scripts/
  connectors/
    reactome.py           # CREATE: Reactome Content Service API
    kegg.py               # CREATE: KEGG REST API
    string_db.py          # CREATE: STRING PPI API (string_db to avoid shadowing)
    clinvar.py            # CREATE: ClinVar via NCBI E-utilities
    omim.py               # CREATE: OMIM REST API
    pharmgkb.py           # CREATE: PharmGKB REST API
  research/
    trajectory.py         # CREATE: PRO-ACT cohort matching + trajectory prediction
    actions.py            # MODIFY: add 5 new ActionType values
    policy.py             # MODIFY: add pathway/genetics/pharmacogenomics triggers
    loop.py               # MODIFY: add 5 new _exec_* functions + dispatch entries
    causal_chains.py      # MODIFY: add pathway-grounded chain construction
tests/
  test_reactome_connector.py
  test_kegg_connector.py
  test_string_connector.py
  test_clinvar_connector.py
  test_omim_connector.py
  test_pharmgkb_connector.py
  test_trajectory.py
  test_expanded_actions.py
```

## 5. Config Additions

```json
{
  "reactome_base_url": "https://reactome.org/ContentService",
  "kegg_base_url": "https://rest.kegg.jp",
  "string_base_url": "https://string-db.org/api",
  "string_min_score": 400,
  "clinvar_enabled": true,
  "omim_api_key": null,
  "pharmgkb_base_url": "https://api.pharmgkb.org/v1",
  "proact_data_dir": null,
  "genetics_received": false
}
```

## 6. Testing Strategy

**Unit tests (no network, no LLM):**
- Each connector: mock HTTP responses → verify EvidenceItem construction, ID format, provenance
- Trajectory: mock PRO-ACT CSV data → verify cohort matching, trajectory computation
- Expanded actions: verify new ActionType values, dispatch routing
- Causal chain enhancement: mock pathway data → verify confidence boosting

**Network integration tests (@pytest.mark.network):**
- Each API connector: real fetch with small query → verify valid response parsing

## 7. Success Criteria

Phase 3B is complete when:
1. All 7 data sources are connected and producing correctly-formatted EvidenceItems
2. 5 new action types are wired into the research loop with proper dispatch
3. Causal chains use Reactome pathway data as ground truth where available
4. PRO-ACT cohort matching returns trajectory predictions for Erik-like patients
5. PharmGKB interaction checking validates the protocol's drug combination safety
6. ClinVar/OMIM are ready to process Erik's genetic results when they arrive
7. All unit tests pass without network; integration tests pass with network
8. README is updated to reflect the expanded architecture
