# Erik — ALS Causal Research Engine

## Project Structure
- `scripts/` — All Python source modules
- `scripts/ontology/` — Canonical Pydantic models (base envelope, ALS types, relations)
- `scripts/db/` — PostgreSQL schema and connection pool
- `scripts/ingestion/` — Clinical document parsing and patient trajectory building
- `scripts/audit/` — Immutable append-only event log
- `scripts/config/` — Hot-reloadable JSON config
- `tests/` — Pytest test suite (mirrors scripts/ structure)
- `data/` — Runtime config and structured patient data

## Key Conventions
- Python env: conda `erik-core` (Python 3.12)
- Database: PostgreSQL `erik_kg` with schemas `erik_core` and `erik_ops`
- All canonical objects inherit from `BaseEnvelope` (scripts/ontology/base.py)
- Entity IDs: `f"{type}:{name}".lower().replace(" ", "_")`
- Import paths: `from ontology.base import BaseEnvelope` (scripts/ is on PYTHONPATH)
- Config file: `data/erik_config.json` (hot-reloaded, never restart for config changes)
- NEVER use sqlite3 — PostgreSQL is the single source of truth
- OBSERVATIONAL_RELATIONSHIP_TYPES must never be upgraded to L3 (causal)
- Tests: TDD — write failing test first, then minimal implementation
- Test command: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v`
