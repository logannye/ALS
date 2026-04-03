"""Extract entities and relationships from evidence items into the knowledge graph.

Scans EvidenceItem objects in erik_core.objects and populates erik_core.entities
and erik_core.relationships from structured body fields. Respects Pearl Causal
Hierarchy constraints — observational relation types can NEVER be promoted to L3.

Usage:
    from knowledge_quality.entity_extractor import extract_kg_from_evidence
    stats = extract_kg_from_evidence(batch_size=50)
    # stats = {"entities_created": N, "relationships_created": M, "items_processed": K}
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Optional

from ontology.relations import is_observational


# Entity types we extract from evidence body fields
_FIELD_TO_ENTITY_TYPE: dict[str, str] = {
    "mechanism_target": "mechanism",
    "intervention_ref": "drug",
    "gene_a": "gene",
    "gene_b": "gene",
    "target_name": "protein",
    "source_name": "protein",
    "gene_symbol": "gene",
    "gene": "gene",  # PharmGKB, ClinVar, DrugBank use 'gene' not 'gene_symbol'
    "drug_name": "drug",
}

# Known ALS genes for recognition in claim text
_ALS_GENES = frozenset({
    "SOD1", "FUS", "TARDBP", "TDP-43", "C9orf72", "STMN2", "UNC13A",
    "SIGMAR1", "SLC1A2", "EAAT2", "BDNF", "GDNF", "OPTN", "TBK1",
    "NEK1", "CSF1R", "MTOR", "SQSTM1", "VCP", "UBQLN2", "DCTN1",
    "ANG", "VAPB", "PFN1", "HNRNPA1", "MATR3", "TUBA4A", "ANXA11",
    "KIF5A", "NEFH", "PRPH",
})

# Known ALS drugs for recognition
_ALS_DRUGS = frozenset({
    "riluzole", "edaravone", "tofersen", "rapamycin", "pridopidine",
    "masitinib", "jacifusen", "ibudilast", "retigabine", "zilucoplan",
    "ceftriaxone", "sodesta",
})


def _make_entity_id(entity_type: str, name: str) -> str:
    """Create canonical entity ID: {type}:{name} (lowercase, underscored)."""
    clean = re.sub(r"[^a-z0-9_]", "_", name.lower().strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return f"{entity_type}:{clean}"


def _make_relationship_id(source_id: str, target_id: str, rel_type: str) -> str:
    """Create deterministic relationship ID from endpoints + type."""
    raw = f"{source_id}|{rel_type}|{target_id}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"rel:{h}"


def _strength_to_confidence(strength: str) -> float:
    """Map evidence strength string to 0-1 confidence."""
    return {
        "strong": 0.9,
        "moderate": 0.7,
        "emerging": 0.5,
        "preclinical": 0.4,
        "unknown": 0.3,
    }.get(strength, 0.3)


def _extract_entities_from_body(body: dict, evidence_id: str) -> list[dict]:
    """Extract entity dicts from an evidence item's body fields."""
    entities: list[dict] = []
    seen_ids: set[str] = set()

    # 1. Extract from known structured fields
    for field, etype in _FIELD_TO_ENTITY_TYPE.items():
        value = body.get(field)
        if not value or not isinstance(value, str):
            continue
        # Skip intervention refs that are IDs (int:xxx) — use as-is
        if field == "intervention_ref" and value.startswith("int:"):
            name = value.replace("int:", "").replace("_", " ")
            eid = _make_entity_id("drug", name)
        else:
            name = value
            eid = _make_entity_id(etype, name)

        if eid not in seen_ids:
            seen_ids.add(eid)
            pch = body.get("pch_layer", 1)
            conf = _strength_to_confidence(body.get("evidence_strength", "unknown"))
            entities.append({
                "id": eid,
                "entity_type": etype,
                "name": name,
                "confidence": conf,
                "sources": [evidence_id],
                "pch_layer": min(pch, 2) if etype in ("gene", "protein") else pch,
                "evidence_type": "extracted_from_evidence",
                "provenance": f"entity_extractor:{evidence_id}",
            })

    # 2. Scan claim text for known ALS genes
    claim = body.get("claim", "")
    for gene in _ALS_GENES:
        if gene in claim:
            eid = _make_entity_id("gene", gene)
            if eid not in seen_ids:
                seen_ids.add(eid)
                entities.append({
                    "id": eid,
                    "entity_type": "gene",
                    "name": gene,
                    "confidence": 0.5,
                    "sources": [evidence_id],
                    "pch_layer": 1,
                    "evidence_type": "extracted_from_claim",
                    "provenance": f"entity_extractor:{evidence_id}",
                })

    # 3. Scan claim text for known ALS drugs
    claim_lower = claim.lower()
    for drug in _ALS_DRUGS:
        if drug in claim_lower:
            eid = _make_entity_id("drug", drug)
            if eid not in seen_ids:
                seen_ids.add(eid)
                entities.append({
                    "id": eid,
                    "entity_type": "drug",
                    "name": drug,
                    "confidence": 0.6,
                    "sources": [evidence_id],
                    "pch_layer": 1,
                    "evidence_type": "extracted_from_claim",
                    "provenance": f"entity_extractor:{evidence_id}",
                })

    return entities


def _infer_relationships(entities: list[dict], body: dict, evidence_id: str) -> list[dict]:
    """Infer relationships between co-occurring entities in the same evidence."""
    relationships: list[dict] = []
    if len(entities) < 2:
        return relationships

    genes = [e for e in entities if e["entity_type"] == "gene"]
    drugs = [e for e in entities if e["entity_type"] == "drug"]
    mechanisms = [e for e in entities if e["entity_type"] == "mechanism"]
    proteins = [e for e in entities if e["entity_type"] == "protein"]

    pch = body.get("pch_layer", 1)
    conf = _strength_to_confidence(body.get("evidence_strength", "unknown"))
    claim = body.get("claim", "")

    # Drug → gene: "targets" relationship
    for drug in drugs:
        for gene in genes:
            rel_type = "targets"
            rid = _make_relationship_id(drug["id"], gene["id"], rel_type)
            relationships.append({
                "id": rid,
                "source_id": drug["id"],
                "target_id": gene["id"],
                "relationship_type": rel_type,
                "confidence": conf,
                "evidence": claim[:200] if claim else f"{drug['name']} targets {gene['name']}",
                "sources": [evidence_id],
                "pch_layer": min(pch, 2),  # Drug-target is at most L2
                "evidence_type": "inferred_from_evidence",
            })

    # Drug → mechanism: "suppresses" or "treats"
    for drug in drugs:
        for mech in mechanisms:
            direction = body.get("direction", "supports")
            rel_type = "suppresses" if direction == "supports" else "associated_with"
            rid = _make_relationship_id(drug["id"], mech["id"], rel_type)
            relationships.append({
                "id": rid,
                "source_id": drug["id"],
                "target_id": mech["id"],
                "relationship_type": rel_type,
                "confidence": conf,
                "evidence": claim[:200] if claim else f"{drug['name']} {rel_type} {mech['name']}",
                "sources": [evidence_id],
                "pch_layer": pch,
                "evidence_type": "inferred_from_evidence",
            })

    # Gene → gene: "associated_with" (from protein interaction data)
    for i, g1 in enumerate(genes):
        for g2 in genes[i + 1:]:
            rel_type = "associated_with"
            rid = _make_relationship_id(g1["id"], g2["id"], rel_type)
            # Observational — always L1
            relationships.append({
                "id": rid,
                "source_id": g1["id"],
                "target_id": g2["id"],
                "relationship_type": rel_type,
                "confidence": conf,
                "evidence": claim[:200] if claim else f"{g1['name']} associated with {g2['name']}",
                "sources": [evidence_id],
                "pch_layer": 1,  # Always L1 for observational
                "evidence_type": "inferred_from_evidence",
            })

    # Gene → mechanism: "contributes_to"
    for gene in genes:
        for mech in mechanisms:
            rel_type = "contributes_to"
            rid = _make_relationship_id(gene["id"], mech["id"], rel_type)
            relationships.append({
                "id": rid,
                "source_id": gene["id"],
                "target_id": mech["id"],
                "relationship_type": rel_type,
                "confidence": conf,
                "evidence": claim[:200] if claim else f"{gene['name']} contributes to {mech['name']}",
                "sources": [evidence_id],
                "pch_layer": min(pch, 2),
                "evidence_type": "inferred_from_evidence",
            })

    # Drug → protein: "binds" (from binding affinity data like BindingDB)
    for drug in drugs:
        for prot in proteins:
            rel_type = "binds"
            rid = _make_relationship_id(drug["id"], prot["id"], rel_type)
            relationships.append({
                "id": rid,
                "source_id": drug["id"],
                "target_id": prot["id"],
                "relationship_type": rel_type,
                "confidence": conf,
                "evidence": claim[:200] if claim else f"{drug['name']} binds {prot['name']}",
                "sources": [evidence_id],
                "pch_layer": min(pch, 2),  # Binding is at most L2
                "evidence_type": "inferred_from_evidence",
            })

    # Protein → protein: "interacts_with" (from Galen KG cross-references)
    for i, p1 in enumerate(proteins):
        for p2 in proteins[i + 1:]:
            # Use body's relationship_type if present, otherwise default
            rel_type = body.get("relationship_type", "interacts_with")
            rid = _make_relationship_id(p1["id"], p2["id"], rel_type)
            relationships.append({
                "id": rid,
                "source_id": p1["id"],
                "target_id": p2["id"],
                "relationship_type": rel_type,
                "confidence": conf,
                "evidence": claim[:200] if claim else f"{p1['name']} {rel_type} {p2['name']}",
                "sources": [evidence_id],
                "pch_layer": 1,  # Always L1 — observational
                "evidence_type": "inferred_from_evidence",
            })

    return relationships


def _validate_relationship_pch(rel: dict) -> dict:
    """Enforce Pearl Causal Hierarchy: observational types can never be L3."""
    if is_observational(rel["relationship_type"]) and rel.get("pch_layer", 1) >= 3:
        rel["pch_layer"] = 1
    return rel


def extract_kg_from_evidence(batch_size: int = 50) -> dict[str, int]:
    """Scan unextracted evidence items and populate entities/relationships.

    Returns stats dict: {entities_created, relationships_created, items_processed}.
    Uses ON CONFLICT upsert for idempotency — safe to run repeatedly.
    """
    import json
    from db.pool import get_connection

    stats = {"entities_created": 0, "relationships_created": 0, "items_processed": 0}

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Find evidence items not yet extracted
            cur.execute("""
                SELECT id, body FROM erik_core.objects
                WHERE type = 'EvidenceItem'
                  AND status = 'active'
                  AND (body->>'kg_extracted' IS NULL OR body->>'kg_extracted' = 'false')
                ORDER BY created_at
                LIMIT %s
            """, (batch_size,))
            rows = cur.fetchall()

            for evi_id, body in rows:
                if not body or not isinstance(body, dict):
                    continue

                entities = _extract_entities_from_body(body, evi_id)
                relationships = _infer_relationships(entities, body, evi_id)

                # Upsert entities
                for ent in entities:
                    try:
                        cur.execute("""
                            INSERT INTO erik_core.entities
                                (id, entity_type, name, properties, confidence, sources,
                                 pch_layer, evidence_type, provenance)
                            VALUES (%s, %s, %s, '{}'::jsonb, %s, %s::jsonb, %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET
                                confidence = GREATEST(erik_core.entities.confidence, EXCLUDED.confidence),
                                sources = (
                                    SELECT jsonb_agg(DISTINCT val)
                                    FROM jsonb_array_elements(erik_core.entities.sources || EXCLUDED.sources) AS val
                                ),
                                updated_at = NOW()
                        """, (
                            ent["id"], ent["entity_type"], ent["name"],
                            ent["confidence"],
                            json.dumps(ent["sources"]),
                            ent["pch_layer"], ent["evidence_type"], ent["provenance"],
                        ))
                        stats["entities_created"] += 1
                    except Exception:
                        continue

                # Upsert relationships (with PCH guard)
                for rel in relationships:
                    rel = _validate_relationship_pch(rel)
                    try:
                        cur.execute("""
                            INSERT INTO erik_core.relationships
                                (id, source_id, target_id, relationship_type, properties,
                                 confidence, evidence, sources, pch_layer, evidence_type)
                            VALUES (%s, %s, %s, %s, '{}'::jsonb, %s, %s, %s::jsonb, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET
                                confidence = GREATEST(erik_core.relationships.confidence, EXCLUDED.confidence),
                                sources = (
                                    SELECT jsonb_agg(DISTINCT val)
                                    FROM jsonb_array_elements(erik_core.relationships.sources || EXCLUDED.sources) AS val
                                ),
                                updated_at = NOW()
                        """, (
                            rel["id"], rel["source_id"], rel["target_id"],
                            rel["relationship_type"],
                            rel["confidence"], rel["evidence"][:500] if rel.get("evidence") else "",
                            json.dumps(rel["sources"]),
                            rel["pch_layer"], rel["evidence_type"],
                        ))
                        stats["relationships_created"] += 1
                    except Exception:
                        # Foreign key violations expected when entity wasn't created
                        continue

                # Mark evidence item as extracted
                cur.execute("""
                    UPDATE erik_core.objects
                    SET body = jsonb_set(body, '{kg_extracted}', 'true'::jsonb),
                        updated_at = NOW()
                    WHERE id = %s
                """, (evi_id,))
                stats["items_processed"] += 1

            conn.commit()

    return stats
