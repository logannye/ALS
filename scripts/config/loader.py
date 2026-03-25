"""Hot-reloadable JSON config loader for the Erik ALS engine.

The config file at `data/erik_config.json` is read at construction time.
Call `reload_if_changed()` periodically (every N steps) to pick up edits
without restarting the process.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

_DEFAULT_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "erik_config.json"


class ConfigLoader:
    """Reads a JSON config file and provides hot-reload capability.

    Args:
        path: Path to the JSON config file.  Defaults to ``data/erik_config.json``
              relative to the project root.
    """

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self._path = pathlib.Path(path) if path is not None else _DEFAULT_PATH
        self._data: dict[str, Any] = {}
        self._mtime: float | None = None
        self.reload()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Read the config file unconditionally and update the internal dict."""
        text = self._path.read_text(encoding="utf-8")
        self._data = json.loads(text)
        self._mtime = self._path.stat().st_mtime

    def reload_if_changed(self) -> bool:
        """Reload only if the file's mtime has changed since the last load.

        Returns:
            True if the file was reloaded, False otherwise.
        """
        try:
            current_mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            return False

        if self._mtime is None or current_mtime != self._mtime:
            self.reload()
            return True
        return False

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if the key is absent."""
        return self._data.get(key, default)

    def get_all(self) -> dict[str, Any]:
        """Return a shallow copy of the entire config dict."""
        return dict(self._data)
