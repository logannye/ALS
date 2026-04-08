import os
import pytest
from connectors.connector_mode import resolve_connector_class


def test_local_mode_returns_local():
    """In local mode, resolver returns the original class path."""
    result = resolve_connector_class("connectors.chembl.ChEMBLConnector", mode="local")
    assert result == "connectors.chembl.ChEMBLConnector"


def test_api_mode_returns_api_variant():
    """In API mode, resolver returns the API class path when available."""
    result = resolve_connector_class("connectors.chembl.ChEMBLConnector", mode="api")
    assert result == "connectors.chembl_api.ChEMBLAPIConnector"


def test_api_mode_falls_back_for_unknown():
    """If no API variant exists, fall back to the original."""
    result = resolve_connector_class("connectors.unknown.Foo", mode="api")
    assert result == "connectors.unknown.Foo"


def test_env_var_controls_mode(monkeypatch):
    """CONNECTOR_MODE env var determines which variant is used."""
    monkeypatch.setenv("CONNECTOR_MODE", "api")
    result = resolve_connector_class("connectors.chembl.ChEMBLConnector")
    assert result == "connectors.chembl_api.ChEMBLAPIConnector"


def test_default_mode_is_local(monkeypatch):
    """Without env var, default to local."""
    monkeypatch.delenv("CONNECTOR_MODE", raising=False)
    result = resolve_connector_class("connectors.chembl.ChEMBLConnector")
    assert result == "connectors.chembl.ChEMBLConnector"
