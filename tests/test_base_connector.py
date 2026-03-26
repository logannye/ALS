import pytest
from connectors.base import BaseConnector, ConnectorResult


def test_connector_result_defaults():
    r = ConnectorResult()
    assert r.evidence_items_added == 0
    assert r.interventions_added == 0
    assert r.errors == []
    assert r.skipped_duplicates == 0


def test_connector_result_accumulate():
    r = ConnectorResult()
    r.evidence_items_added += 5
    r.errors.append("parse error on item X")
    assert r.evidence_items_added == 5
    assert len(r.errors) == 1


class MockConnector(BaseConnector):
    def fetch(self, **kwargs):
        return ConnectorResult(evidence_items_added=1)

def test_mock_connector_fetch():
    c = MockConnector()
    result = c.fetch()
    assert result.evidence_items_added == 1


def test_retry_succeeds_on_second_attempt():
    call_count = 0
    def flaky_fn():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("transient failure")
        return "success"
    c = MockConnector()
    result = c._retry_with_backoff(flaky_fn)
    assert result == "success"
    assert call_count == 2


def test_retry_exhausts_retries():
    def always_fails():
        raise ConnectionError("permanent failure")
    c = MockConnector()
    with pytest.raises(ConnectionError):
        c._retry_with_backoff(always_fails)


def test_connector_has_timeout():
    c = MockConnector()
    assert c.REQUEST_TIMEOUT == 30
