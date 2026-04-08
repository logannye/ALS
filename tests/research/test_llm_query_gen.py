from unittest.mock import MagicMock
from research.policy import _get_llm_query


def test_llm_query_returns_string():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "TDP-43 nuclear depletion cryptic exon STMN2 therapy 2026"
    query = _get_llm_query(
        llm=mock_llm,
        active_hypotheses=["TDP-43 aggregation drives cryptic exon splicing"],
        top_uncertainties=["Missing genetic testing data"],
        layer="root_cause_suppression",
    )
    assert isinstance(query, str)
    assert len(query) > 10


def test_llm_query_falls_back_on_error():
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = Exception("LLM timeout")
    query = _get_llm_query(
        llm=mock_llm,
        active_hypotheses=["Test hypothesis"],
        top_uncertainties=["Missing data"],
        layer="root_cause_suppression",
    )
    assert isinstance(query, str)
    assert len(query) > 5


def test_llm_query_strips_quotes():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '"ALS motor neuron degeneration biomarker 2026"'
    query = _get_llm_query(
        llm=mock_llm,
        active_hypotheses=[],
        top_uncertainties=[],
        layer="circuit_stabilization",
    )
    assert not query.startswith('"')
    assert not query.endswith('"')
