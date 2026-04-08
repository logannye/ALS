"""Tests for the _apply_stagnation_recovery() function and exploration burst mechanism."""
import pytest
from dataclasses import replace

from research.state import ResearchState
from research.loop import _apply_stagnation_recovery


def _make_mock_state(**kwargs) -> ResearchState:
    """Create a minimal ResearchState for testing."""
    defaults = dict(
        subject_ref="traj:test_subject",
        active_hypotheses=["hyp_a", "hyp_b", "hyp_c", "hyp_d"],
        action_posteriors={
            "search_pubmed": (3.0, 2.0),
            "search_trials": (1.5, 4.0),
            "generate_hypothesis": (2.0, 1.0),
        },
        stagnation_resets=0,
        last_stagnation_step=0,
        target_exhaustion={"TARDBP:search_pubmed": 5, "FUS:search_trials": 3},
        expansion_query_history=["ALS TDP-43 2026", "SOD1 therapy 2026"],
        expansion_gene_history={"TARDBP": ["FUS", "HNRNPA1"]},
        exploration_burst_remaining=0,
    )
    defaults.update(kwargs)
    return ResearchState(**defaults)


def test_recovery_sets_exploration_burst():
    """After stagnation recovery, exploration_burst_remaining > 0."""
    state = _make_mock_state()
    new_state = _apply_stagnation_recovery(state, step=250)
    assert new_state.exploration_burst_remaining > 0


def test_recovery_burst_is_20():
    """Burst is exactly 20 steps."""
    state = _make_mock_state()
    new_state = _apply_stagnation_recovery(state, step=250)
    assert new_state.exploration_burst_remaining == 20


def test_recovery_clears_exhaustion():
    """Recovery clears target_exhaustion."""
    state = _make_mock_state()
    assert state.target_exhaustion  # non-empty
    new_state = _apply_stagnation_recovery(state, step=250)
    assert new_state.target_exhaustion == {}


def test_recovery_clears_expansion_query_history():
    """Recovery clears expansion_query_history."""
    state = _make_mock_state()
    assert state.expansion_query_history
    new_state = _apply_stagnation_recovery(state, step=250)
    assert new_state.expansion_query_history == []


def test_recovery_clears_expansion_gene_history():
    """Recovery clears expansion_gene_history."""
    state = _make_mock_state()
    assert state.expansion_gene_history
    new_state = _apply_stagnation_recovery(state, step=250)
    assert new_state.expansion_gene_history == {}


def test_recovery_increments_counter():
    """Recovery increments stagnation_resets by 1."""
    state = _make_mock_state(stagnation_resets=3)
    new_state = _apply_stagnation_recovery(state, step=250)
    assert new_state.stagnation_resets == 4


def test_recovery_increments_counter_from_zero():
    """Recovery increments stagnation_resets from 0 to 1."""
    state = _make_mock_state(stagnation_resets=0)
    new_state = _apply_stagnation_recovery(state, step=250)
    assert new_state.stagnation_resets == 1


def test_recovery_sets_last_stagnation_step():
    """Recovery records the step at which stagnation was detected."""
    state = _make_mock_state()
    new_state = _apply_stagnation_recovery(state, step=350)
    assert new_state.last_stagnation_step == 350


def test_recovery_expires_half_hypotheses():
    """Recovery expires oldest 50% of active hypotheses."""
    state = _make_mock_state(active_hypotheses=["h1", "h2", "h3", "h4"])
    new_state = _apply_stagnation_recovery(state, step=250)
    # floor(4/2) = 2 expired, 2 remain
    assert len(new_state.active_hypotheses) == 2
    assert new_state.active_hypotheses == ["h3", "h4"]


def test_recovery_expires_at_least_one():
    """Recovery expires at least 1 hypothesis even with a single one."""
    state = _make_mock_state(active_hypotheses=["only_hypothesis"])
    new_state = _apply_stagnation_recovery(state, step=250)
    assert new_state.active_hypotheses == []


def test_recovery_with_empty_hypotheses():
    """Recovery does not crash when there are no active hypotheses."""
    state = _make_mock_state(active_hypotheses=[])
    new_state = _apply_stagnation_recovery(state, step=250)
    assert new_state.active_hypotheses == []
    assert new_state.stagnation_resets == 1


def test_recovery_expires_odd_number():
    """Recovery expires floor(n/2) hypotheses for odd n."""
    state = _make_mock_state(active_hypotheses=["h1", "h2", "h3"])
    new_state = _apply_stagnation_recovery(state, step=250)
    # floor(3/2) = 1 expired, 2 remain
    assert len(new_state.active_hypotheses) == 2
    assert new_state.active_hypotheses == ["h2", "h3"]


def test_recovery_resets_posteriors():
    """Recovery resets all Thompson posteriors to (1.0, 1.0)."""
    state = _make_mock_state()
    new_state = _apply_stagnation_recovery(state, step=250)
    for key, (alpha, beta) in new_state.action_posteriors.items():
        assert alpha == 1.0, f"Expected alpha=1.0 for {key}, got {alpha}"
        assert beta == 1.0, f"Expected beta=1.0 for {key}, got {beta}"


def test_recovery_resets_posteriors_preserves_keys():
    """Recovery preserves the same posterior keys, just resets values."""
    state = _make_mock_state()
    original_keys = set(state.action_posteriors.keys())
    new_state = _apply_stagnation_recovery(state, step=250)
    assert set(new_state.action_posteriors.keys()) == original_keys


def test_recovery_does_not_mutate_input_state():
    """Recovery returns a new state; the original is unchanged."""
    state = _make_mock_state(stagnation_resets=0)
    _ = _apply_stagnation_recovery(state, step=250)
    assert state.stagnation_resets == 0
    assert state.exploration_burst_remaining == 0
    assert state.target_exhaustion != {}


def test_burst_honored_in_select_action():
    """select_action forces least-used acquisition action during burst."""
    from research.policy import select_action, _ACQUISITION_ROTATION

    # Create a state with burst active and zero action counts
    state = _make_mock_state(
        exploration_burst_remaining=5,
        action_counts={},
    )
    action, params = select_action(state)
    # During burst, should pick an acquisition action from _ACQUISITION_ROTATION
    assert action in _ACQUISITION_ROTATION


def test_burst_picks_least_used_action():
    """During burst, select_action picks the action with the lowest use count."""
    from research.policy import select_action, _ACQUISITION_ROTATION

    # Make all actions used heavily except QUERY_GWAS
    all_counts = {a.value: 100 for a in _ACQUISITION_ROTATION}
    target_action = _ACQUISITION_ROTATION[-1]  # last in rotation
    all_counts[target_action.value] = 0  # least used

    state = _make_mock_state(
        exploration_burst_remaining=3,
        action_counts=all_counts,
    )
    action, params = select_action(state)
    assert action == target_action


def test_no_burst_uses_normal_policy():
    """Without burst, select_action uses normal cycle/Thompson policy."""
    from research.policy import select_action

    state = _make_mock_state(
        exploration_burst_remaining=0,
        action_counts={},
    )
    # Should not raise and should return a valid action
    action, params = select_action(state)
    assert action is not None
    assert isinstance(params, dict)
