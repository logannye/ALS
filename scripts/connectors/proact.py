"""ProactConnector — loads PRO-ACT clinical trial CSV data into erik_ops.proact_trajectories.

PRO-ACT (Pooled Resource Open-Access ALS Clinical Trials) is a large ALS clinical
dataset. This connector loads CSV files from a local directory and idempotently
ingests them into the operational PostgreSQL database for trajectory matching.
"""
from __future__ import annotations

import csv
import io
import logging
import os
from pathlib import Path
from typing import Optional

from connectors.base import BaseConnector, ConnectorResult

logger = logging.getLogger(__name__)

# Expected canonical output keys from _parse_row
_ROW_KEYS = (
    "patient_id",
    "age_onset",
    "sex",
    "onset_region",
    "alsfrs_r_total",
    "fvc_percent",
    "time_months",
    "vital_status",
    "survival_months",
)

# Mapping from common CSV column name variants → canonical key
_COLUMN_MAP: dict[str, str] = {
    # patient ID
    "patient_id": "patient_id",
    "subjectid": "patient_id",
    "subject_id": "patient_id",
    "patientid": "patient_id",
    # age at onset
    "age_onset": "age_onset",
    "age_at_onset": "age_onset",
    "onset_age": "age_onset",
    "age": "age_onset",
    # sex
    "sex": "sex",
    "gender": "sex",
    # onset region
    "onset_region": "onset_region",
    "onset_location": "onset_region",
    "site_of_onset": "onset_region",
    "onset": "onset_region",
    # alsfrs-r
    "alsfrs_r_total": "alsfrs_r_total",
    "alsfrs_r": "alsfrs_r_total",
    "alsfrsr_total": "alsfrs_r_total",
    "total_alsfrs": "alsfrs_r_total",
    # fvc
    "fvc_percent": "fvc_percent",
    "fvc_pct": "fvc_percent",
    "fvc_%": "fvc_percent",
    "fvc": "fvc_percent",
    # time
    "time_months": "time_months",
    "delta": "time_months",
    "alsfrs_delta": "time_months",
    "time_since_onset_months": "time_months",
    "months_from_onset": "time_months",
    # vital status
    "vital_status": "vital_status",
    "status": "vital_status",
    "death_status": "vital_status",
    # survival
    "survival_months": "survival_months",
    "overall_survival_months": "survival_months",
    "survival_time_months": "survival_months",
}

_NUMERIC_FIELDS = {"age_onset", "alsfrs_r_total", "fvc_percent", "time_months", "survival_months"}
_INT_FIELDS = {"vital_status"}


def _safe_float(value: str) -> Optional[float]:
    """Convert string to float; return None on failure."""
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except (ValueError, TypeError):
        return None


def _safe_int(value: str) -> Optional[int]:
    """Convert string to int; return None on failure."""
    f = _safe_float(value)
    if f is None:
        return None
    return int(f)


class ProactConnector(BaseConnector):
    """Connector that loads PRO-ACT CSV data into erik_ops.proact_trajectories.

    Idempotent: if the table already contains data, fetch() is a no-op.
    """

    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS erik_ops.proact_trajectories (
            id              SERIAL PRIMARY KEY,
            patient_id      TEXT NOT NULL,
            age_onset       NUMERIC,
            sex             TEXT,
            onset_region    TEXT,
            alsfrs_r_total  NUMERIC,
            fvc_percent     NUMERIC,
            time_months     NUMERIC,
            vital_status    INTEGER,
            survival_months NUMERIC,
            loaded_at       TIMESTAMPTZ DEFAULT NOW()
        );
    """

    def __init__(self, *, data_dir: Optional[str] = None, pool=None) -> None:
        self.data_dir = data_dir
        self._pool = pool

    # ------------------------------------------------------------------
    # BaseConnector contract
    # ------------------------------------------------------------------

    def fetch(self, **kwargs) -> ConnectorResult:
        """Load all CSV files in data_dir into proact_trajectories (idempotent).

        Returns immediately if the table already has rows or if data_dir is
        None / does not exist.
        """
        result = ConnectorResult()

        if not self.data_dir:
            logger.info("ProactConnector: no data_dir configured, skipping.")
            return result

        data_path = Path(self.data_dir)
        if not data_path.is_dir():
            result.errors.append(f"proact data_dir not found: {self.data_dir!r}")
            return result

        if self._pool is None:
            result.errors.append("ProactConnector: no db pool provided, cannot load data.")
            return result

        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    # Ensure table exists
                    cur.execute(self.CREATE_TABLE_SQL)

                    # Idempotency check
                    cur.execute("SELECT COUNT(*) FROM erik_ops.proact_trajectories")
                    row = cur.fetchone()
                    existing = row[0] if row else 0
                    if existing > 0:
                        logger.info(
                            "ProactConnector: table already has %d rows, skipping load.",
                            existing,
                        )
                        result.skipped_duplicates = existing
                        return result

                    # Load CSV files
                    csv_files = sorted(data_path.glob("*.csv"))
                    if not csv_files:
                        logger.warning("ProactConnector: no CSV files found in %s", self.data_dir)
                        return result

                    for csv_path in csv_files:
                        sub = self._load_csv(csv_path, cur)
                        result.evidence_items_added += sub.evidence_items_added
                        result.errors.extend(sub.errors)

                    conn.commit()

        except Exception as e:
            result.errors.append(f"ProactConnector.fetch failed: {e}")

        return result

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_row(row: dict) -> Optional[dict[str, object]]:
        """Normalise a raw CSV row dict into canonical keys.

        Returns None if the row has no patient_id and no time_months
        (i.e. completely empty / unusable).

        All numeric fields use None for missing/unparseable values.
        """
        if not row:
            return None

        # Normalise keys to lowercase, strip whitespace
        normalised: dict[str, str] = {
            k.strip().lower(): v for k, v in row.items()
        }

        # Map to canonical names
        canonical: dict[str, str] = {}
        for raw_key, raw_val in normalised.items():
            mapped = _COLUMN_MAP.get(raw_key)
            if mapped and mapped not in canonical:
                canonical[mapped] = raw_val

        # Require at minimum a patient_id OR a time_months to be present
        patient_id = canonical.get("patient_id")
        time_months_raw = canonical.get("time_months")

        if patient_id is None and time_months_raw is None:
            return None

        # Build output dict with all canonical keys
        out: dict[str, object] = {}
        for key in _ROW_KEYS:
            raw_val = canonical.get(key)
            if key in _NUMERIC_FIELDS:
                out[key] = _safe_float(raw_val)
            elif key in _INT_FIELDS:
                out[key] = _safe_int(raw_val)
            else:
                # string fields
                if raw_val is not None and str(raw_val).strip():
                    out[key] = str(raw_val).strip()
                else:
                    out[key] = None

        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_csv(self, csv_path: Path, cur) -> ConnectorResult:
        """Parse a single CSV file and INSERT rows into proact_trajectories."""
        result = ConnectorResult()
        try:
            text = csv_path.read_text(encoding="utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(text))
            for raw_row in reader:
                parsed = self._parse_row(dict(raw_row))
                if parsed is None:
                    result.skipped_duplicates += 1
                    continue
                try:
                    cur.execute(
                        """
                        INSERT INTO erik_ops.proact_trajectories
                            (patient_id, age_onset, sex, onset_region,
                             alsfrs_r_total, fvc_percent, time_months,
                             vital_status, survival_months)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            parsed["patient_id"],
                            parsed["age_onset"],
                            parsed["sex"],
                            parsed["onset_region"],
                            parsed["alsfrs_r_total"],
                            parsed["fvc_percent"],
                            parsed["time_months"],
                            parsed["vital_status"],
                            parsed["survival_months"],
                        ),
                    )
                    result.evidence_items_added += 1
                except Exception as e:
                    result.errors.append(f"Row insert error: {e}")
        except Exception as e:
            result.errors.append(f"Failed to read {csv_path}: {e}")
        return result
