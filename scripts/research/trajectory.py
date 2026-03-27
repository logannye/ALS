# scripts/research/trajectory.py
"""PRO-ACT cohort matching and trajectory prediction."""
from __future__ import annotations
import csv, io, os, statistics
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class CohortMatch:
    n_patients: int = 0
    median_decline_rate: float = 0.0
    p25_decline_rate: float = 0.0
    p75_decline_rate: float = 0.0
    median_survival_months: float = 0.0
    erik_percentile: float = 0.0

@dataclass
class _SubjectSummary:
    subject_id: str
    age: Optional[float] = None
    sex: Optional[str] = None
    onset: Optional[str] = None
    baseline_alsfrs_r: Optional[float] = None
    decline_rate: Optional[float] = None

def _parse_alsfrs_csv(csv_text: str) -> list[dict]:
    if not csv_text.strip():
        return []
    reader = csv.DictReader(io.StringIO(csv_text))
    records = []
    for row in reader:
        try:
            records.append({
                "subject_id": row.get("SubjectID", row.get("subject_id", "")),
                "alsfrs_delta": int(float(row.get("ALSFRS_Delta", row.get("alsfrs_delta", 0)))),
                "alsfrs_r_total": float(row.get("ALSFRS_R_Total", row.get("alsfrs_r_total", 0))),
            })
        except (ValueError, TypeError):
            continue
    return records

class ProACTAnalyzer:
    def __init__(self, data_dir: Optional[str] = None):
        self._data_dir = data_dir
        self._loaded = False
        self._subjects: dict[str, _SubjectSummary] = {}

    def _load_from_records(self, records: list[dict]) -> None:
        by_subject: dict[str, list[dict]] = {}
        for rec in records:
            by_subject.setdefault(rec["subject_id"], []).append(rec)
        for sid, recs in by_subject.items():
            recs.sort(key=lambda r: r.get("alsfrs_delta", 0))
            baseline = recs[0].get("alsfrs_r_total")
            last = recs[-1].get("alsfrs_r_total")
            delta_days = recs[-1].get("alsfrs_delta", 0) - recs[0].get("alsfrs_delta", 0)
            decline_rate = None
            if baseline is not None and last is not None and delta_days > 0:
                decline_rate = (last - baseline) / (delta_days / 30.0)
            self._subjects[sid] = _SubjectSummary(
                subject_id=sid, age=recs[0].get("age"), sex=recs[0].get("sex"),
                onset=recs[0].get("onset"), baseline_alsfrs_r=baseline, decline_rate=decline_rate,
            )
        self._loaded = True

    def load(self) -> bool:
        if self._loaded:
            return True
        if not self._data_dir or not os.path.isdir(self._data_dir):
            return False
        alsfrs_path = os.path.join(self._data_dir, "ALSFRS.csv")
        if not os.path.isfile(alsfrs_path):
            return False
        with open(alsfrs_path) as f:
            records = _parse_alsfrs_csv(f.read())
        self._load_from_records(records)
        return True

    def match_cohort(self, age, sex, onset_region, baseline_alsfrs_r, decline_rate, age_range=10.0, alsfrs_range=6.0) -> CohortMatch:
        if not self._loaded or not self._subjects:
            return CohortMatch()
        onset_map = {"lower_limb": "Limb", "upper_limb": "Limb", "bulbar": "Bulbar", "limb": "Limb"}
        target_onset = onset_map.get(onset_region.lower(), onset_region)
        matched_rates = []
        for subj in self._subjects.values():
            if subj.decline_rate is None or subj.baseline_alsfrs_r is None:
                continue
            if subj.age is not None and abs(subj.age - age) > age_range:
                continue
            if subj.onset is not None and target_onset and subj.onset != target_onset:
                continue
            if abs(subj.baseline_alsfrs_r - baseline_alsfrs_r) > alsfrs_range:
                continue
            matched_rates.append(subj.decline_rate)
        if not matched_rates:
            return CohortMatch()
        matched_rates.sort()
        n = len(matched_rates)
        p25 = matched_rates[n // 4] if n >= 4 else matched_rates[0]
        p75 = matched_rates[3 * n // 4] if n >= 4 else matched_rates[-1]
        slower_count = sum(1 for r in matched_rates if r <= decline_rate)
        return CohortMatch(
            n_patients=n, median_decline_rate=round(statistics.median(matched_rates), 3),
            p25_decline_rate=round(p25, 3), p75_decline_rate=round(p75, 3),
            erik_percentile=round((slower_count / n) * 100.0, 1),
        )
