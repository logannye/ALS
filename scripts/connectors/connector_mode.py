"""Connector mode resolver — maps local connectors to API variants on Railway."""
from __future__ import annotations

import os

# Map: local class path → API class path
_API_VARIANTS: dict[str, str] = {
    "connectors.chembl.ChEMBLConnector": "connectors.chembl_api.ChEMBLAPIConnector",
    "connectors.clinvar_local.ClinVarLocalConnector": "connectors.clinvar.ClinVarConnector",
    "connectors.reactome_local.ReactomeLocalConnector": "connectors.reactome.ReactomeConnector",
    "connectors.uniprot.UniProtConnector": "connectors.uniprot_api.UniProtAPIConnector",
    "connectors.alphafold_local.AlphaFoldLocalConnector": "connectors.alphafold_api.AlphaFoldAPIConnector",
    "connectors.gtex.GTExConnector": "connectors.gtex_api.GTExAPIConnector",
    "connectors.gwas_catalog.GWASCatalogConnector": "connectors.gwas_api.GWASCatalogAPIConnector",
    "connectors.gnomad.GnomADConnector": "connectors.gnomad_api.GnomADAPIConnector",
    "connectors.hpa.HPAConnector": "connectors.hpa_api.HPAAPIConnector",
    "connectors.galen_kg.GalenKGConnector": "connectors.galen_kg_api.GalenKGAPIConnector",
}


def resolve_connector_class(
    local_class_path: str,
    mode: str | None = None,
) -> str:
    """Return the connector class path appropriate for the current mode.

    Args:
        local_class_path: Dotted import path of the local connector.
        mode: 'local' or 'api'. Defaults to CONNECTOR_MODE env var, then 'local'.

    Returns:
        The (possibly remapped) class path.
    """
    if mode is None:
        mode = os.environ.get("CONNECTOR_MODE", "local").lower()
    if mode == "api":
        return _API_VARIANTS.get(local_class_path, local_class_path)
    return local_class_path
