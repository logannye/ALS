"""BaseConnector ABC and ConnectorResult for all evidence connectors."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ConnectorResult:
    """Result of a connector fetch operation."""
    evidence_items_added: int = 0
    interventions_added: int = 0
    errors: list[str] = field(default_factory=list)
    skipped_duplicates: int = 0


class BaseConnector(ABC):
    """Abstract base for all evidence connectors.

    Contract:
    - Upsert-by-source-ID (deterministic canonical IDs)
    - Exponential backoff retries (3 attempts, 1/2/4s)
    - 30s request timeout
    """
    MAX_RETRIES = 3
    BACKOFF_SECONDS = [1, 2, 4]
    REQUEST_TIMEOUT = 30

    @abstractmethod
    def fetch(self, **kwargs) -> ConnectorResult:
        ...

    def _retry_with_backoff(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call fn with exponential backoff on failure. Raises after MAX_RETRIES."""
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.BACKOFF_SECONDS[attempt])
        raise last_error
