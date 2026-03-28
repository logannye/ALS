"""TrajectoryMatcher — DTW-based cohort matching against PRO-ACT trajectories.

Matches Erik's current disease state against historical PRO-ACT patients using
Dynamic Time Warping on ALSFRS-R sequences, then estimates survival statistics
and domain-specific reversibility-window timing using Kaplan-Meier or percentile
fallback.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default reversibility-window threshold levels (ALSFRS-R scores)
DEFAULT_THRESHOLDS: dict[str, float] = {
    "molecular": 40.0,   # Early: molecular correction plausible
    "nmj": 34.0,         # Mid: NMJ integrity still partially intact
    "functional": 24.0,  # Late: functional reserve critically depleted
}


# ---------------------------------------------------------------------------
# DTW distance
# ---------------------------------------------------------------------------

def _dtw_distance(seq_a: list[float], seq_b: list[float]) -> float:
    """Compute DTW distance between two 1-D sequences using numpy.

    Parameters
    ----------
    seq_a, seq_b:
        Lists of numeric values (e.g. ALSFRS-R scores over time).

    Returns
    -------
    DTW distance as a float.  Returns 0.0 for two empty sequences.
    """
    n = len(seq_a)
    m = len(seq_b)

    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return float(np.sum(np.abs(seq_b)))
    if m == 0:
        return float(np.sum(np.abs(seq_a)))

    a = np.asarray(seq_a, dtype=float)
    b = np.asarray(seq_b, dtype=float)

    # Allocate DP cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j],      # insertion
                                   dtw[i, j - 1],      # deletion
                                   dtw[i - 1, j - 1])  # match

    return float(dtw[n, m])


# ---------------------------------------------------------------------------
# Survival estimation
# ---------------------------------------------------------------------------

def _estimate_survival(cohort: list[dict]) -> dict[str, float]:
    """Estimate survival statistics from a cohort of patient dicts.

    Each dict should have ``survival_months`` (float|None) and
    ``vital_status`` (int|None, 1 = event observed, 0 = censored).

    Tries Kaplan-Meier (lifelines) first; falls back to percentile estimation.

    Returns
    -------
    dict with keys:
        median_months_remaining, p25_months, p75_months
    """
    zeros = {"median_months_remaining": 0.0, "p25_months": 0.0, "p75_months": 0.0}

    if not cohort:
        return zeros

    # Filter to rows with usable survival_months
    valid = [
        p for p in cohort
        if p.get("survival_months") is not None
        and p["survival_months"] >= 0
    ]
    if not valid:
        return zeros

    durations = np.array([p["survival_months"] for p in valid], dtype=float)
    events = np.array(
        [1 if (p.get("vital_status") or 1) == 1 else 0 for p in valid],
        dtype=int,
    )

    # Try Kaplan-Meier via lifelines
    try:
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=events)

        median = float(kmf.median_survival_time_)
        if np.isnan(median) or np.isinf(median):
            median = float(np.median(durations))

        # KM timeline and survival function for percentiles
        timeline = kmf.survival_function_.index.values
        sf = kmf.survival_function_["KM_estimate"].values

        def _km_percentile(q: float) -> float:
            """Return time at which survival fraction drops to q."""
            idx = np.searchsorted(-sf, -q)
            if idx >= len(timeline):
                return float(timeline[-1]) if len(timeline) else 0.0
            return float(timeline[idx])

        p25 = _km_percentile(0.75)   # 25th percentile survival = 75% still alive
        p75 = _km_percentile(0.25)   # 75th percentile survival = 25% still alive

        return {
            "median_months_remaining": median,
            "p25_months": min(p25, median),
            "p75_months": max(p75, median),
        }

    except ImportError:
        logger.debug("lifelines not available, using percentile fallback")
    except Exception as e:
        logger.warning("KaplanMeierFitter failed (%s), using percentile fallback", e)

    # Fallback: simple percentile estimation (ignores censoring)
    median = float(np.median(durations))
    p25 = float(np.percentile(durations, 25))
    p75 = float(np.percentile(durations, 75))

    return {
        "median_months_remaining": median,
        "p25_months": p25,
        "p75_months": p75,
    }


# ---------------------------------------------------------------------------
# Window estimation
# ---------------------------------------------------------------------------

def _estimate_windows(
    trajectories: list[list[dict]],
    current_alsfrs_r: float,
    thresholds: dict[str, float],
) -> dict[str, float]:
    """Estimate median time (months) for matched patients to cross ALSFRS-R thresholds.

    Parameters
    ----------
    trajectories:
        List of patient trajectory lists.  Each trajectory is a list of dicts
        with at least ``time_months`` and ``alsfrs_r_total``.
    current_alsfrs_r:
        Erik's current ALSFRS-R total score.
    thresholds:
        Mapping of layer name → ALSFRS-R threshold value to cross.

    Returns
    -------
    dict mapping layer name → median months until threshold crossing.
    Layers where no patient crosses the threshold return ``float("inf")``.
    """
    if not trajectories or not thresholds:
        return {}

    result: dict[str, float] = {}

    for layer_name, threshold in thresholds.items():
        crossing_times: list[float] = []

        for traj in trajectories:
            if not traj:
                continue

            # Sort by time
            sorted_traj = sorted(traj, key=lambda p: p.get("time_months") or 0)

            # If already below threshold at start, crossing time is 0.0
            first_score = sorted_traj[0].get("alsfrs_r_total")
            if first_score is not None and first_score <= threshold:
                crossing_times.append(0.0)
                continue

            # Find first time point where alsfrs_r_total <= threshold
            crossed = False
            for point in sorted_traj:
                score = point.get("alsfrs_r_total")
                t = point.get("time_months")
                if score is None or t is None:
                    continue
                if score <= threshold:
                    crossing_times.append(float(t))
                    crossed = True
                    break

            # Patient never crossed within observation window — skip (censored)

        if crossing_times:
            result[layer_name] = float(np.median(crossing_times))
        else:
            result[layer_name] = float("inf")

    return result


# ---------------------------------------------------------------------------
# TrajectoryMatcher
# ---------------------------------------------------------------------------

class TrajectoryMatcher:
    """Match Erik's disease trajectory against PRO-ACT historical cohort.

    Uses DTW on ALSFRS-R sequences to find similar trajectories, then
    applies Kaplan-Meier survival estimation and window timing.
    """

    def __init__(
        self,
        cohort_age_window: int = 5,
        top_k: int = 50,
        thresholds: Optional[dict[str, float]] = None,
        pool=None,
    ) -> None:
        self.cohort_age_window = cohort_age_window
        self.top_k = top_k
        self.thresholds = thresholds if thresholds is not None else dict(DEFAULT_THRESHOLDS)
        self._pool = pool

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        age: float,
        sex: str,
        onset_region: str,
        alsfrs_r: float,
        fvc_percent: float,
    ) -> dict:
        """Match against PRO-ACT cohort and return trajectory statistics.

        Parameters
        ----------
        age:            Patient age at current visit.
        sex:            Patient sex string (e.g. "Male", "Female").
        onset_region:   Onset anatomical region (e.g. "Limb", "Bulbar").
        alsfrs_r:       Current ALSFRS-R total score.
        fvc_percent:    Current FVC as % predicted.

        Returns
        -------
        dict compatible with ``TrajectoryMatchResult.model_validate()``.
        """
        empty = {
            "cohort_size": 0,
            "matched_k": 0,
            "median_months_remaining": 0.0,
            "p25_months": 0.0,
            "p75_months": 0.0,
            "window_estimates": {},
            "decline_rate_percentile": 0.0,
        }

        # Query cohort from DB
        try:
            all_patients = self._query_cohort(age, sex, onset_region)
        except Exception as e:
            logger.warning("TrajectoryMatcher._query_cohort failed: %s", e)
            return empty

        if not all_patients:
            return empty

        cohort_size = len(all_patients)

        # Build per-patient ALSFRS-R time-series for DTW matching
        patient_series = self._build_patient_series(all_patients)

        # Reference series: a simple [current_alsfrs_r] singleton (or we could
        # use Erik's own trajectory if available in the future)
        ref_series = [alsfrs_r]

        # Compute DTW distances
        distances: list[tuple[float, dict]] = []
        for pid, series, meta in patient_series:
            dist = _dtw_distance(ref_series, series)
            distances.append((dist, meta))

        # Sort by ascending distance and take top-k
        distances.sort(key=lambda x: x[0])
        top_matches = [m for _, m in distances[: self.top_k]]
        matched_k = len(top_matches)

        # Survival estimation
        survival_stats = _estimate_survival(top_matches)

        # Window estimation: build per-patient full trajectories
        # Map patient_id → list of time-point dicts
        pid_to_traj: dict[str, list[dict]] = {}
        for p in all_patients:
            pid = str(p.get("patient_id", ""))
            if pid not in pid_to_traj:
                pid_to_traj[pid] = []
            pid_to_traj[pid].append(p)

        top_patient_ids = {str(m.get("patient_id", "")) for m in top_matches}
        top_trajectories = [
            pid_to_traj[pid]
            for pid in top_patient_ids
            if pid in pid_to_traj
        ]

        window_estimates = _estimate_windows(
            top_trajectories,
            current_alsfrs_r=alsfrs_r,
            thresholds=self.thresholds,
        )

        # Decline rate percentile: compare Erik's FVC to matched cohort
        decline_rate_percentile = self._compute_fvc_percentile(fvc_percent, top_matches)

        return {
            "cohort_size": cohort_size,
            "matched_k": matched_k,
            "median_months_remaining": survival_stats["median_months_remaining"],
            "p25_months": survival_stats["p25_months"],
            "p75_months": survival_stats["p75_months"],
            "window_estimates": window_estimates,
            "decline_rate_percentile": decline_rate_percentile,
        }

    # ------------------------------------------------------------------
    # DB query (overridable for testing)
    # ------------------------------------------------------------------

    def _query_cohort(
        self,
        age: float,
        sex: str,
        onset_region: str,
    ) -> list[dict]:
        """Query proact_trajectories for age-matched patients.

        Filters by age ± cohort_age_window.  Returns list of row dicts.
        """
        if self._pool is None:
            return []

        sql = """
            SELECT
                patient_id, age_onset, sex, onset_region,
                alsfrs_r_total, fvc_percent, time_months,
                vital_status, survival_months
            FROM erik_ops.proact_trajectories
            WHERE age_onset BETWEEN %s AND %s
            ORDER BY patient_id, time_months
        """
        age_lo = age - self.cohort_age_window
        age_hi = age + self.cohort_age_window

        rows = []
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (age_lo, age_hi))
                    cols = [d[0] for d in cur.description]
                    for row in cur.fetchall():
                        rows.append(dict(zip(cols, row)))
        except Exception as e:
            logger.warning("_query_cohort DB error: %s", e)

        return rows

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_patient_series(
        self, patients: list[dict]
    ) -> list[tuple[str, list[float], dict]]:
        """Aggregate rows by patient_id into (pid, alsfrs_series, meta) tuples.

        ``meta`` is the last row for the patient (used for survival stats).
        """
        pid_points: dict[str, list[tuple[float, float]]] = {}
        pid_meta: dict[str, dict] = {}

        for p in patients:
            pid = str(p.get("patient_id", "unknown"))
            t = p.get("time_months")
            score = p.get("alsfrs_r_total")
            if t is not None and score is not None:
                pid_points.setdefault(pid, []).append((float(t), float(score)))
            pid_meta[pid] = p  # keep last-seen row as meta

        result = []
        for pid, points in pid_points.items():
            # Sort by time and extract just the score sequence
            series = [s for _, s in sorted(points, key=lambda x: x[0])]
            result.append((pid, series, pid_meta[pid]))

        return result

    @staticmethod
    def _compute_fvc_percentile(fvc: float, cohort: list[dict]) -> float:
        """Return percentile rank of fvc within the cohort's FVC values."""
        fvc_values = [
            p["fvc_percent"]
            for p in cohort
            if p.get("fvc_percent") is not None
        ]
        if not fvc_values:
            return 0.0
        arr = np.array(fvc_values, dtype=float)
        return float(np.mean(arr <= fvc) * 100.0)
