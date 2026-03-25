# Phase 1: Evidence Fabric — Design Specification

## 1. Purpose

Build the minimum sufficient evidence layer to generate and rank cure protocol candidates for Erik Draper, a 67-year-old male with newly diagnosed limb-onset sporadic ALS (ALSFRS-R 43/48, NfL 5.82 elevated, FVC 100%, genetic testing pending).

The evidence fabric is organized around the 5 cure protocol layers, not around data sources. Every evidence item answers one question: **what intervention, targeting what mechanism, with what evidence strength, could be part of Erik's cure protocol?**

## 2. Strategic Decisions

- **Protocol-first**: Evidence selection is driven backward from the cure protocol structure. We ingest what's needed to evaluate candidate interventions, not an encyclopedia.
- **Curated seed + automated expansion**: Hand-build ~110 highest-value evidence items (papers, therapies, compounds), then build API connectors to keep the corpus current.
- **All subtypes covered**: Genetic testing pending — design supports SOD1, C9orf72, FUS, TARDBP, and sporadic TDP-43 paths. Narrow when results arrive.
- **Off-label and experimental included**: Any compound with a credible mechanistic link to ALS disease programs is in scope, regardless of original indication or regulatory status.
- **Computational drug design supported**: If no existing compound adequately addresses a critical mechanism, the fabric stores structural biology and compound library data to support bespoke molecule design.

## 3. Protocol Layer → Evidence Mapping

```
Layer A: Root-cause suppression
  → Gene-targeted therapies (ASOs, gene therapy, intrabodies)
  → Per-subtype: SOD1 (tofersen, Sodesta), C9orf72 (next-gen after BIIB078),
    FUS (jacifusen), sporadic TDP-43 (VTx-002 intrabody)
  → Target engagement biomarkers, timing evidence

Layer B: Pathology reversal
  → TDP-43 nuclear reimport / aggregation clearance
  → Cryptic splicing correction (STMN2, UNC13A restoration)
  → Proteostasis enhancers (autophagy activators, UPS modulators)
  → Stress granule dissolution
  → Pridopidine (sigma-1R agonist, Phase 3 PREVAiLS)

Layer C: Circuit stabilization
  → Anti-excitotoxicity (riluzole MOA, perampanel, memantine, retigabine)
  → Glial modulation (masitinib, ibudilast, complement inhibition)
  → Neuroinflammation suppression
  → NMJ stabilization

Layer D: Regeneration and reinnervation
  → Neurotrophic factors (BDNF, GDNF, neurturin delivery)
  → NMJ reoccupation strategies
  → Axonal regrowth factors
  → Cell replacement (iPSC motor neurons, glial progenitors)

Layer E: Adaptive maintenance
  → NfL pharmacodynamic monitoring
  → ALSFRS-R monitoring cadence
  → Respiratory decline detection
  → Combination protocol design principles
  → Taper/continuation logic
```

## 4. Data Sources

### Tier 1 — Build immediately (days 1-3)

| Source | Access | API | Value |
|--------|--------|-----|-------|
| PubMed | Free, no registration | E-utilities REST (JSON/XML) | Mechanistic papers per protocol layer, trial results, biomarker studies |
| ClinicalTrials.gov | Free, no registration | REST v2 (JSON) | Active Phase 2/3 ALS trials, eligibility criteria, sites |
| ChEMBL 36 | Already local at `/Volumes/Databank/databases/chembl_36.db` | Local SQL | Compounds targeting ALS proteins, bioactivity (IC50/EC50), SAR |
| OpenTargets | Free, no registration | GraphQL | Ranked target-disease associations, tractability, clinical precedent |
| DrugBank | Free academic registration | XML download | Full MOA, ADMET, drug-drug interactions for ALS therapies |

### Tier 2 — Build next (days 4-7)

| Source | Access | API | Value |
|--------|--------|-----|-------|
| UniProt | Free | REST | Protein structure, domains, PTMs, variants for TDP-43/SOD1/FUS/C9orf72 |
| STRING DB | Free | REST | PPI network around ALS targets, druggable node identification |
| AlphaFold/PDB | Free | REST | 3D structures for computational drug design |
| ClinVar | Free | E-utilities | ALS variant catalog (critical once genetics arrive) |
| Reactome | Free | REST | Pathway models: autophagy, proteostasis, UPR, axonal transport |

### Tier 3 — Background ingestion (week 2+)

| Source | Access | Value |
|--------|--------|-------|
| PRO-ACT | Free registration | Erik vs 8,600 historical trajectories |
| GEO GSE174332 | Free download | Motor cortex cell-type vulnerability atlas |
| KEGG/ALSoD/OMIM | Free/registration | Pathway scaffolds, gene credibility |

## 5. Curated Evidence Seed

~110 hand-curated evidence items organized by protocol layer. These are the highest-value evidence items for Erik's case, selected for mechanistic specificity and evidence strength.

### Layer A: Root-cause suppression (~25 items)

**SOD1 path:**
- Tofersen VALOR RCT (Miller et al 2022, NEJM) — target engagement + NfL reduction
- Tofersen 3.5-year OLE (JAMA Neurology Dec 2025) — early initiation benefit
- Sodesta AAV gene therapy (FDA accelerated approval 2025) — one-time intrathecal

**C9orf72 path:**
- BIIB078 Phase 1 failure analysis (Cell 2025) — why DPR/pTDP-43 persisted despite ASO distribution
- Next-gen C9orf72 strategies (repeat RNA targeting, DPR-specific approaches)
- C9orf72 loss-of-function biology (autophagy/immune, O'Rourke et al 2016)

**FUS path:**
- Jacifusen expanded access (12 patients, NfL -82.8%, one presymptomatic stable 3y)
- FUSION Phase 3 trial design
- FUS liquid-to-solid transition biology (Patel et al 2015)

**Sporadic TDP-43 path (most relevant for Erik):**
- VTx-002 TDP-43 intrabody gene therapy (FDA Fast Track Jan 2026, PIONEER-ALS Phase 1/2)
- TDP-43 aggregation mechanisms (Neumann et al 2006, Ling et al 2015)
- TDP-43 nuclear import/export biology
- AP-2 TDP-43 balance restoration (Molefy Pharma, preclinical)

### Layer B: Pathology reversal (~25 items)

**Cryptic splicing:**
- UNC13A cryptic exon mechanism (Brown et al 2022, Ma et al 2022)
- STMN2 repression under TDP-43 loss (Klim et al 2019, Melamed et al 2019)
- ASO approaches to block cryptic exon inclusion

**Proteostasis:**
- Autophagy activators: rapamycin/everolimus in TDP-43 models
- TFEB activation strategies
- UPS enhancement approaches
- XBP1/UPR modulation (Hetz et al 2009 — protective in SOD1 models)

**Sigma-1R agonism:**
- Pridopidine Phase 2 results (32% slowing, 62% respiratory benefit)
- PREVAiLS Phase 3 design (500 patients, 60 centers, enrolling Jan 2026)
- Sigma-1R mechanism: ER-mitochondria tethering, calcium homeostasis, proteostasis

**Stress granule biology:**
- LLPS hardening in ALS RBPs (Patel et al 2015, Molliex et al 2015)
- Dissolution strategies, kinase inhibitors

### Layer C: Circuit stabilization (~20 items)

**Anti-excitotoxicity:**
- Riluzole MOA and clinical evidence (Bensimon 1994, Lacomblez 1996)
- EAAT2/GLT-1 upregulation strategies
- Perampanel (AMPA antagonist) — ALS case series
- Retigabine/ezogabine (Kv7 opener) — iPSC hyperexcitability data (Wainger et al 2014)

**Glial modulation:**
- Masitinib (tyrosine kinase inhibitor, Phase 3 ALS data — survival benefit in subgroup)
- Ibudilast (PDE inhibitor, neuroinflammation)
- Non-cell-autonomous toxicity (Boillee et al 2006, Di Giorgio et al 2008)

**Complement:**
- Zilucoplan C5 inhibition (HEALEY platform — negative primary, but biomarker subgroup question open)

### Layer D: Regeneration (~15 items)

- BDNF/GDNF delivery strategies (viral vector, direct infusion)
- NMJ reoccupation biology (Gould et al 2006 — dying-back model)
- Agrin/MuSK pathway for NMJ stabilization
- iPSC-derived motor neuron transplantation (current state of art)
- Glial progenitor transplantation strategies

### Layer E: Adaptive maintenance (~15 items)

- NfL as pharmacodynamic biomarker (Benatar et al 2023)
- ALSFRS-R sensitivity and monitoring cadence
- Combination protocol design principles from oncology (relevant paradigm)
- Respiratory monitoring: FVC trends, SNIP, nocturnal oximetry
- Riluzole combination safety data

### Computational drug design targets (~10 items)

- TDP-43 RRM1/RRM2 domain structures (PDB + AlphaFold)
- TDP-43 C-terminal domain / aggregation-prone region structure
- UNC13A/STMN2 cryptic exon sequences (ASO design targets)
- Sigma-1R crystal structure (PDB: 5HK1)
- SOD1 aggregation interface structures
- ZINC purchasable compound library (for virtual screening)
- ChEMBL SAR data for existing TDP-43 binders

## 6. Evidence Object Schema

### 6.1 Model changes required (Phase 0 compatibility)

The existing `EvidenceItem.strength` field is `float`. This spec requires `strength` as a string enum for human-readable protocol reasoning. **Implementation must**:
1. Add `EvidenceStrength` enum to `enums.py`: `strong`, `moderate`, `emerging`, `preclinical`, `unknown`
2. Change `EvidenceItem.strength` from `float` to `EvidenceStrength` (default `unknown`)
3. Add `gene_therapy`, `cell_therapy`, `peptide` to `InterventionClass` enum
4. Update any existing tests that pass numeric `strength` values

### 6.2 EvidenceItem schema

Every evidence item becomes a canonical `EvidenceItem` in the Erik ontology:

```python
EvidenceItem(
    id="evi:{intervention}_{mechanism}_{source}",
    claim="Human-readable evidence claim",
    direction=EvidenceDirection.SUPPORTS | REFUTES | MIXED,
    source_refs=["pmid:XXXXX"],
    strength=EvidenceStrength.STRONG,
    body={
        # Protocol targeting
        "protocol_layer": "root_cause_suppression",
        "mechanism_target": "SOD1_toxic_gain_of_function",
        "intervention_ref": "int:tofersen",
        "intervention_class": "aso",

        # Regulatory and applicability
        "regulatory_status": "approved|experimental|preclinical|off_label",
        "applicable_subtypes": ["sod1"],
        "erik_eligible": True | False | "pending_genetics",
        "trial_phase": "approved",
        "pch_layer": 2,

        # Evidence objectization (per parent spec Section 11.3)
        "modality": "rct",  # REQUIRED: rct|observational|case_series|preclinical|in_vitro|computational|review|meta_analysis
        "cohort_size": 108,
        "tissue_context": "csf_and_plasma",
        "time_structure": "28_week_rct_plus_ole",
        "access_restrictions": "published",

        # Endpoints
        "key_endpoints": {"nfl_reduction": "60%", "alsfrs_r_slope": "not_significant_28wk"},
    }
)
```

### 6.3 Contradiction handling

Per parent spec Section 4.1: "Contradictions MUST be represented explicitly."

When two `EvidenceItem` objects with opposing `direction` values reference the same `mechanism_target`:
1. The seed builder creates an `EvidenceBundle` with both items in `evidence_item_refs` and both IDs in `contradiction_refs`
2. The bundle's `topic` is set to `"contradiction:{mechanism_target}"`
3. Example: AMX0035 (Phase 2 positive → Phase 3 PHOENIX negative) generates a contradiction bundle

Connectors must also detect contradictions: if a newly ingested paper's `direction` opposes an existing item for the same `mechanism_target`, flag for contradiction bundle creation.

### 6.4 Evidence lifecycle (versioning)

Per parent spec Section 4.1: "Interpretations MUST be versioned and supersedable."

Evidence items follow append-only semantics:
- **New evidence**: creates a new `EvidenceItem` with `status=active`
- **Updated evidence** (e.g., trial results update): creates a new `EvidenceItem` with `supersedes_ref` pointing to the previous version. Previous item is marked `status=superseded`.
- **Retracted evidence**: previous item is marked `status=deprecated` with audit log entry
- **Erik eligibility change** (e.g., genetics arrive): creates new version with updated `erik_eligible` and `applicable_subtypes`, supersedes prior version
- `EvidenceBundle` always resolves to latest `active` version of each referenced item

### 6.5 Intervention schema

`Intervention` objects are created for each therapy:

```python
Intervention(
    id="int:tofersen",
    name="Tofersen (QALSODY)",
    intervention_class=InterventionClass.ASO,
    targets=["SOD1"],
    protocol_layer=ProtocolLayer.ROOT_CAUSE_SUPPRESSION,
    route="intrathecal",
    intended_effects=["reduce_SOD1_mRNA", "reduce_SOD1_protein", "reduce_NfL"],
    known_risks=["CSF_pleocytosis", "injection_site_reaction", "myelitis_rare"],
    body={
        "regulatory_status": "approved_accelerated",
        "applicable_subtypes": ["sod1"],
        "dosing": "100mg intrathecal q4w (after loading)",
        "evidence_bundle_ref": "evb:tofersen_evidence",
        "drugbank_id": "DB16638",
        "chembl_id": "CHEMBL4594302",
    }
)
```

## 7. Automated Connectors

Five API connectors, each producing canonical `EvidenceItem` and `Intervention` objects.

### 7.1 Connector Contract (BaseConnector)

All connectors implement a shared contract:

**Idempotency**: Upsert-by-source-ID. Each source has a canonical ID (PMID, NCT ID, ChEMBL ID, DrugBank ID). If an item with the same source ID already exists and is `active`, the connector either skips (if content unchanged) or creates a new version with `supersedes_ref` (if content changed).

**Rate limiting**: Exponential backoff — 3 attempts with 1s/2s/4s delays. PubMed: 10 req/sec with API key, 3/sec without. ClinicalTrials.gov: 1 req/sec. Others: respect published rate limits.

**Partial failure**: Committed per-item. If 3 of 100 results fail to parse, the 97 successful items are committed and 3 failures are logged to `erik_ops.audit_events` with `event_type=connector_parse_error`.

**Cross-source deduplication**: Interventions are deduped by canonical ID (`int:tofersen`). If ChEMBL, DrugBank, and OpenTargets all mention tofersen, only one `Intervention` object exists. Evidence items from different sources are NOT deduped — each represents distinct evidence with its own provenance.

**Source system mapping**: PubMed → `SourceSystem.LITERATURE`, ClinicalTrials.gov → `SourceSystem.TRIAL`, ChEMBL → `SourceSystem.DATABASE`, OpenTargets → `SourceSystem.DATABASE`, DrugBank → `SourceSystem.DATABASE`.

### 7.2 PubMedConnector
- Queries: curated search terms per protocol layer (stored in config)
- Output: EvidenceItem per relevant paper (claim extracted from abstract)
- Rate limit: 10 req/sec with API key, 3/sec without
- Refresh: daily for "ALS treatment" alerts, weekly for broader queries

### ClinicalTrialsConnector
- Queries: `condition=ALS AND status=RECRUITING,NOT_YET_RECRUITING AND phase=PHASE2,PHASE3`
- Output: EvidenceItem per trial + Intervention per experimental arm
- Eligibility matching: compare Erik's profile against inclusion/exclusion criteria
- Refresh: weekly

### 7.4 ChEMBLConnector
- Queries: local SQL against ChEMBL 36 for ALS target bioactivity
- Targets: TDP-43 (Q13148), SOD1 (P00441), FUS (P35637), EAAT2/SLC1A2, SIGMAR1, etc.
- Output: EvidenceItem per compound-target pair with activity data
- Refresh: on ChEMBL version update

### 7.5 OpenTargetsConnector
- Queries: GraphQL for ALS (MONDO_0004976) target associations
- Output: EvidenceItem per target with tractability and evidence scores
- Refresh: quarterly (follows OpenTargets release cycle)

### 7.6 DrugBankConnector
- Input: XML database download (academic license)
- Output: Intervention objects with full MOA, ADMET, drug-drug interaction data
- Critical for: combination protocol safety checking
- Refresh: on DrugBank release

## 8. Computational Drug Design Data Layer

For targets where no adequate existing compound exists:

### Structure Repository
- AlphaFold predicted structures for ALS targets (TDP-43, FUS, C9orf72, EAAT2, SIGMAR1)
- PDB experimental structures where available
- Stored as: structure files (PDB/mmCIF) + metadata in `erik_core.objects`

### Compound Libraries
- ChEMBL bioactivity data for ALS-relevant targets (already local)
- ZINC purchasable compound subset (for future virtual screening)
- Stored as: compound records with SMILES, activity data, ADMET predictions

### Binding Site Data
- Pocket annotations from fpocket/P2Rank on target structures
- Known binding sites from co-crystal structures
- Stored as: structured annotations on protein entity objects

This layer does NOT perform molecular dynamics or docking in Phase 1. It stores the data needed for a future computational chemistry module.

## 9. File Structure (New Files)

```
scripts/
  evidence/
    __init__.py
    seed_builder.py          # Build curated evidence seed (Layer A-E items)
    evidence_store.py        # CRUD for EvidenceItem + EvidenceBundle in PostgreSQL
  connectors/
    __init__.py
    base.py                  # BaseConnector ABC (query, parse, rate_limit)
    pubmed.py                # PubMed E-utilities connector
    clinical_trials.py       # ClinicalTrials.gov v2 API connector
    chembl.py                # ChEMBL local SQL connector
    opentargets.py           # OpenTargets GraphQL connector
    drugbank.py              # DrugBank XML parser
  targets/
    __init__.py
    als_targets.py           # Canonical ALS target definitions (proteins, genes, pathways)
    structure_store.py       # AlphaFold/PDB structure retrieval and storage
tests/
  test_seed_builder.py
  test_evidence_store.py
  test_pubmed_connector.py
  test_clinical_trials_connector.py
  test_chembl_connector.py
  test_opentargets_connector.py
  test_drugbank_connector.py
  test_als_targets.py
data/
  seed/
    layer_a_root_cause.json    # Curated seed: root-cause suppression evidence
    layer_b_pathology.json     # Curated seed: pathology reversal evidence
    layer_c_circuit.json       # Curated seed: circuit stabilization evidence
    layer_d_regeneration.json  # Curated seed: regeneration evidence
    layer_e_maintenance.json   # Curated seed: adaptive maintenance evidence
    interventions.json         # All intervention objects
    drug_design_targets.json   # Computational drug design target definitions
```

## 10. Database Schema Extensions

```sql
-- New indexes for evidence queries
CREATE INDEX IF NOT EXISTS idx_objects_body_layer
  ON erik_core.objects USING gin ((body->'protocol_layer'));
CREATE INDEX IF NOT EXISTS idx_objects_body_subtype
  ON erik_core.objects USING gin ((body->'applicable_subtypes'));
CREATE INDEX IF NOT EXISTS idx_objects_body_target
  ON erik_core.objects USING gin ((body->'mechanism_target'));
```

## 11. Key ALS Trial Intelligence (as of March 2026)

Integrated into seed as highest-priority evidence items:

| Therapy | Target | Phase | Relevance to Erik |
|---------|--------|-------|-------------------|
| Riluzole | Glutamate/excitotoxicity | Approved | Already taking. Layer C baseline. |
| Tofersen | SOD1 | Approved (accel.) | Layer A — only if SOD1+ on genetics |
| Sodesta (AAV) | SOD1 | Approved (accel.) | Layer A — only if SOD1+ |
| Pridopidine | Sigma-1R | Phase 3 (PREVAiLS) | Layer B — **highest relevance**, sporadic ALS, enrolling now |
| VTx-002 | TDP-43 aggregates | Phase 1/2 | Layer A — **highest relevance**, targets 97% of ALS |
| Jacifusen | FUS | Phase 3 (FUSION) | Layer A — only if FUS+ |
| Masitinib | Tyrosine kinases/glia | Phase 3 | Layer C — sporadic eligible, survival subgroup signal |
| ATLX-1282 | Undisclosed | Phase 2 | Monitor — Lilly $415M deal, H2 2026 readout |
| AP-2 | TDP-43 balance | Preclinical | Layer A/B — early but direct mechanism |

## 12. Success Criteria

Phase 1 is complete when:
1. ~110 curated evidence items are stored as canonical objects in PostgreSQL
2. ~30 intervention objects cover all approved + investigational ALS therapies
3. All 5 protocol layers have evidence coverage
4. Erik's eligibility is assessed against every active Phase 2/3 trial
5. PubMed and ClinicalTrials.gov connectors can refresh evidence automatically
6. ChEMBL connector can query local DB for compound-target bioactivity
7. OpenTargets connector returns ranked ALS targets
8. Structural biology data (AlphaFold/PDB) is stored for top 10 drug design targets
9. All evidence items carry protocol_layer, mechanism_target, applicable_subtypes, erik_eligible, and pch_layer tags
10. Tests verify evidence integrity, connector functionality, and seed completeness
