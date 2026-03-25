"""Tests for Task 11: Hot-reloadable JSON config loader."""

import json
import pathlib
import time
import pytest


class TestConfigLoaderImport:
    def test_config_loader_importable(self):
        from config.loader import ConfigLoader  # noqa: F401

    def test_config_loader_is_class(self):
        from config.loader import ConfigLoader
        assert isinstance(ConfigLoader, type)


class TestConfigLoaderReadsFile:
    def test_reads_from_temp_file(self, tmp_path):
        from config.loader import ConfigLoader
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"foo": "bar", "count": 42}))
        loader = ConfigLoader(path=cfg_file)
        assert loader.get("foo") == "bar"
        assert loader.get("count") == 42

    def test_get_returns_default_for_missing_key(self, tmp_path):
        from config.loader import ConfigLoader
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"foo": "bar"}))
        loader = ConfigLoader(path=cfg_file)
        assert loader.get("missing_key") is None
        assert loader.get("missing_key", "fallback") == "fallback"
        assert loader.get("missing_key", 0) == 0

    def test_get_all_returns_full_dict(self, tmp_path):
        from config.loader import ConfigLoader
        data = {"a": 1, "b": 2, "c": True}
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(data))
        loader = ConfigLoader(path=cfg_file)
        result = loader.get_all()
        assert result == data

    def test_get_all_returns_copy(self, tmp_path):
        from config.loader import ConfigLoader
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"x": 1}))
        loader = ConfigLoader(path=cfg_file)
        copy1 = loader.get_all()
        copy1["injected"] = True
        copy2 = loader.get_all()
        assert "injected" not in copy2

    def test_boolean_and_float_values(self, tmp_path):
        from config.loader import ConfigLoader
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"flag": False, "temp": 1.5}))
        loader = ConfigLoader(path=cfg_file)
        assert loader.get("flag") is False
        assert loader.get("temp") == pytest.approx(1.5)

    def test_nested_dict_value(self, tmp_path):
        from config.loader import ConfigLoader
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"nested": {"a": 1}}))
        loader = ConfigLoader(path=cfg_file)
        assert loader.get("nested") == {"a": 1}


class TestConfigLoaderHotReload:
    def test_reload_picks_up_changed_value(self, tmp_path):
        from config.loader import ConfigLoader
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"version": "1.0"}))
        loader = ConfigLoader(path=cfg_file)
        assert loader.get("version") == "1.0"

        cfg_file.write_text(json.dumps({"version": "2.0"}))
        loader.reload()
        assert loader.get("version") == "2.0"

    def test_reload_if_changed_returns_false_when_unchanged(self, tmp_path):
        from config.loader import ConfigLoader
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"k": "v"}))
        loader = ConfigLoader(path=cfg_file)
        result = loader.reload_if_changed()
        assert result is False

    def test_reload_if_changed_returns_true_and_updates_after_modification(self, tmp_path):
        from config.loader import ConfigLoader
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"k": "original"}))
        loader = ConfigLoader(path=cfg_file)

        # Force a different mtime by advancing it by 1 second
        current_mtime = cfg_file.stat().st_mtime
        new_content = json.dumps({"k": "updated"})
        cfg_file.write_text(new_content)
        # Explicitly set mtime to be different
        import os
        os.utime(cfg_file, (current_mtime + 2.0, current_mtime + 2.0))

        result = loader.reload_if_changed()
        assert result is True
        assert loader.get("k") == "updated"

    def test_reload_if_changed_handles_missing_file(self, tmp_path):
        from config.loader import ConfigLoader
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"k": "v"}))
        loader = ConfigLoader(path=cfg_file)
        cfg_file.unlink()
        result = loader.reload_if_changed()
        assert result is False
        # Original data still accessible
        assert loader.get("k") == "v"

    def test_reload_adds_new_key(self, tmp_path):
        from config.loader import ConfigLoader
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"existing": 1}))
        loader = ConfigLoader(path=cfg_file)
        assert loader.get("new_key") is None

        cfg_file.write_text(json.dumps({"existing": 1, "new_key": "hello"}))
        loader.reload()
        assert loader.get("new_key") == "hello"

    def test_reload_removes_deleted_key(self, tmp_path):
        from config.loader import ConfigLoader
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"keep": 1, "drop": 2}))
        loader = ConfigLoader(path=cfg_file)
        assert loader.get("drop") == 2

        cfg_file.write_text(json.dumps({"keep": 1}))
        loader.reload()
        assert loader.get("drop") is None


class TestDefaultConfigFile:
    """Verify the default data/erik_config.json loads correctly."""

    def test_default_config_file_loads(self):
        from config.loader import ConfigLoader
        # Uses default path — data/erik_config.json
        loader = ConfigLoader()
        assert loader.get("version") is not None

    def test_default_config_has_expected_keys(self):
        from config.loader import ConfigLoader
        loader = ConfigLoader()
        assert loader.get("database_name") == "erik_kg"
        assert loader.get("audit_enabled") is True
        assert loader.get("action_timeout_s") == 120
        assert loader.get("exploration_epsilon") == pytest.approx(0.30)
        assert loader.get("hot_reload_interval_steps") == 10
