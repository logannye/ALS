"""DrugBankConnector — parses DrugBank academic XML for ALS-relevant drugs.

DrugBank provides an academic download of their full database as XML.
This connector filters entries by ALS-related keywords and converts them
into Intervention objects for use in the Erik evidence engine.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Optional

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import InterventionClass, SourceSystem
from ontology.intervention import Intervention

logger = logging.getLogger(__name__)

# DrugBank XML namespace
DRUGBANK_NS = "http://www.drugbank.ca"
NS = {"db": DRUGBANK_NS}

# ALS relevance keywords (all lowercase for case-insensitive matching)
ALS_KEYWORDS = [
    "amyotrophic lateral sclerosis",
    "als",
    "motor neuron disease",
    "neurodegeneration",
    "neuroprotect",
]


# ---------------------------------------------------------------------------
# Free function: parse a single <drug> element
# ---------------------------------------------------------------------------

def _parse_drug_entry(drug_el: ET.Element, ns: dict) -> Intervention:
    """Parse a DrugBank ``<drug>`` XML element into an Intervention.

    Parameters
    ----------
    drug_el:
        An ``xml.etree.ElementTree.Element`` with tag matching ``{ns}drug``.
    ns:
        Namespace prefix mapping, e.g. ``{"db": "http://www.drugbank.ca"}``.

    Returns
    -------
    Intervention with fields populated from the DrugBank entry.
    """
    # Primary DrugBank ID
    drugbank_id = ""
    for id_el in drug_el.findall("db:drugbank-id", ns):
        if id_el.get("primary") == "true":
            drugbank_id = (id_el.text or "").strip()
            break

    name = (drug_el.findtext("db:name", default="", namespaces=ns) or "").strip()

    description = (
        drug_el.findtext("db:description", default="", namespaces=ns) or ""
    ).strip()[:300]

    indication = (
        drug_el.findtext("db:indication", default="", namespaces=ns) or ""
    ).strip()[:300]

    mechanism_of_action = (
        drug_el.findtext("db:mechanism-of-action", default="", namespaces=ns) or ""
    ).strip()[:500]

    # Groups (approved, investigational, etc.)
    groups: list[str] = []
    groups_el = drug_el.find("db:groups", ns)
    if groups_el is not None:
        for g in groups_el.findall("db:group", ns):
            text = (g.text or "").strip()
            if text:
                groups.append(text)

    # Regulatory status: "approved" if in groups, else first group or "unknown"
    if "approved" in groups:
        regulatory_status = "approved"
    elif groups:
        regulatory_status = groups[0]
    else:
        regulatory_status = "unknown"

    # Targets (polypeptide name or target name)
    targets: list[str] = []
    targets_el = drug_el.find("db:targets", ns)
    if targets_el is not None:
        for target_el in targets_el.findall("db:target", ns):
            # Prefer polypeptide name, fall back to target name
            poly_el = target_el.find("db:polypeptide", ns)
            if poly_el is not None:
                poly_name = (
                    poly_el.findtext("db:name", default="", namespaces=ns) or ""
                ).strip()
                if poly_name:
                    targets.append(poly_name)
                    continue
            target_name = (
                target_el.findtext("db:name", default="", namespaces=ns) or ""
            ).strip()
            if target_name:
                targets.append(target_name)

    # Drug interactions
    drug_interactions: list[dict] = []
    di_el = drug_el.find("db:drug-interactions", ns)
    if di_el is not None:
        for di in di_el.findall("db:drug-interaction", ns):
            di_id = (
                di.findtext("db:drugbank-id", default="", namespaces=ns) or ""
            ).strip()
            di_name = (
                di.findtext("db:name", default="", namespaces=ns) or ""
            ).strip()
            di_desc = (
                di.findtext("db:description", default="", namespaces=ns) or ""
            ).strip()
            drug_interactions.append(
                {
                    "drugbank_id": di_id,
                    "name": di_name,
                    "description": di_desc,
                }
            )

    return Intervention(
        id=f"int:drugbank:{drugbank_id}",
        name=name,
        intervention_class=InterventionClass.drug,
        targets=targets,
        route="",
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="drugbank_connector",
        ),
        body={
            "drugbank_id": drugbank_id,
            "regulatory_status": regulatory_status,
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "description": description,
            "indication": indication,
            "mechanism_of_action": mechanism_of_action,
            "groups": groups,
            "drug_interactions": drug_interactions,
        },
    )


# ---------------------------------------------------------------------------
# DrugBankConnector
# ---------------------------------------------------------------------------

class DrugBankConnector(BaseConnector):
    """Connector that parses the DrugBank academic XML download.

    Filters drug entries by ALS-relevant keywords found in description,
    indication, or mechanism-of-action fields, then converts to Interventions.

    Usage:
        connector = DrugBankConnector()
        result = connector.fetch_als_drugs(xml_path="/path/to/drugbank_all_full_database.xml")
    """

    ALS_KEYWORDS: list[str] = ALS_KEYWORDS

    def __init__(self, *, store=None) -> None:
        self._store = store
        # In-memory cache of parsed ALS interventions keyed by drugbank_id
        self._interventions: dict[str, Intervention] = {}

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(self, **kwargs) -> ConnectorResult:
        """Top-level fetch: requires xml_path kwarg, delegates to fetch_als_drugs."""
        xml_path = kwargs.get("xml_path")
        return self.fetch_als_drugs(xml_path=xml_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_als_drugs(
        self,
        xml_path: Optional[str] = None,
    ) -> ConnectorResult:
        """Parse DrugBank XML, filter by ALS keywords, return Interventions.

        Parameters
        ----------
        xml_path:
            Path to the DrugBank XML file (full database download).
            If None, returns an error in the result.

        Returns
        -------
        ConnectorResult with interventions_added count and any errors.
        """
        result = ConnectorResult()

        if xml_path is None:
            result.errors.append("xml_path is required — no DrugBank XML file specified")
            return result

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except FileNotFoundError:
            result.errors.append(f"DrugBank XML file not found: {xml_path!r}")
            return result
        except ET.ParseError as e:
            result.errors.append(f"Failed to parse DrugBank XML at {xml_path!r}: {e}")
            return result
        except Exception as e:
            result.errors.append(f"Cannot open DrugBank XML at {xml_path!r}: {e}")
            return result

        for drug_el in root.findall("db:drug", NS):
            try:
                if not self._is_als_relevant(drug_el):
                    continue

                intervention = _parse_drug_entry(drug_el, NS)

                # Cache by drugbank_id for interaction lookups
                db_id = intervention.body.get("drugbank_id", "")
                if db_id:
                    self._interventions[db_id] = intervention

                if self._store:
                    self._store.upsert_intervention(intervention)

                result.interventions_added += 1

            except Exception as e:
                name_el = drug_el.find("db:name", NS)
                name = (name_el.text or "unknown") if name_el is not None else "unknown"
                result.errors.append(f"Failed to parse drug {name!r}: {e}")

        return result

    def fetch_drug_interactions(
        self,
        drugbank_ids: list[str],
    ) -> dict[str, list[dict]]:
        """Return cross-interaction data for the given DrugBank IDs.

        Looks up stored interventions (populated by a prior fetch_als_drugs call)
        and returns the drug_interactions list for each requested ID.

        Parameters
        ----------
        drugbank_ids:
            List of DrugBank IDs to look up, e.g. ["DB00316"].

        Returns
        -------
        Dict mapping each requested ID to its list of interaction dicts.
        IDs not found in the store are mapped to an empty list.
        """
        if not drugbank_ids:
            return {}

        result: dict[str, list[dict]] = {}
        for db_id in drugbank_ids:
            intervention = self._interventions.get(db_id)
            if intervention is None:
                result[db_id] = []
            else:
                result[db_id] = intervention.body.get("drug_interactions", [])
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_als_relevant(self, drug_el: ET.Element) -> bool:
        """Return True if the drug entry contains any ALS keyword.

        Checks description, indication, and mechanism-of-action fields.
        Matching is case-insensitive substring search.
        """
        fields_to_check = [
            (drug_el.findtext("db:description", default="", namespaces=NS) or ""),
            (drug_el.findtext("db:indication", default="", namespaces=NS) or ""),
            (
                drug_el.findtext(
                    "db:mechanism-of-action", default="", namespaces=NS
                )
                or ""
            ),
        ]
        combined = " ".join(fields_to_check).lower()
        return any(kw in combined for kw in self.ALS_KEYWORDS)
