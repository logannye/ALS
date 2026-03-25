"""Integration-level tests for the full Erik Draper trajectory.

Verifies cross-cutting invariants: observation count, decline rate,
FVC, EMG ALS support, brain MRI irrelevance.
"""

from datetime import date

import pytest

from ingestion.patient_builder import build_erik_draper
from ontology.enums import ObservationKind


@pytest.fixture(scope="module")
def erik_data():
    return build_erik_draper()


@pytest.fixture(scope="module")
def patient(erik_data):
    return erik_data[0]


@pytest.fixture(scope="module")
def trajectory(erik_data):
    return erik_data[1]


@pytest.fixture(scope="module")
def observations(erik_data):
    return erik_data[2]


class TestObservationCount:
    def test_at_least_30_observations(self, observations):
        assert len(observations) >= 30

    def test_linked_refs_match(self, trajectory, observations):
        obs_ids = {o.id for o in observations}
        for ref in trajectory.linked_observation_refs:
            assert ref in obs_ids


class TestDeclineRate:
    def test_decline_rate_range(self, trajectory):
        """ALSFRS-R decline ~ -0.38 pts/month (5 lost over ~13 months)."""
        score = trajectory.alsfrs_r_scores[0]
        rate = score.decline_rate_from_onset(trajectory.onset_date)
        # Should be between -0.50 and -0.30 pts/month
        assert -0.50 <= rate <= -0.30, f"Decline rate {rate} outside expected range"


class TestFVC:
    def test_fvc_100_percent(self, observations):
        resps = [o for o in observations
                 if o.observation_kind == ObservationKind.respiratory_metric]
        assert len(resps) >= 1
        spirometry = resps[0]
        rm = spirometry.respiratory_metric
        assert rm is not None
        assert rm.fvc_percent_predicted == 100.0


class TestEMG:
    def test_emg_supports_als(self, observations):
        emgs = [o for o in observations
                if o.observation_kind == ObservationKind.emg_feature]
        # At least the Mar 2026 CC EMG must have supports_als=True
        als_supporting = [o for o in emgs if o.emg_finding.supports_als is True]
        assert len(als_supporting) >= 1


class TestBrainMRI:
    def test_brain_mri_not_als_relevant(self, observations):
        imgs = [o for o in observations
                if o.observation_kind == ObservationKind.imaging_finding]
        brain = [o for o in imgs
                 if "brain" in o.name.lower() or "brain" in o.id.lower()]
        assert len(brain) >= 1
        assert brain[0].imaging_finding.als_relevant is False
