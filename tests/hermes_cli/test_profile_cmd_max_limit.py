"""``profiles.max`` enforced through the CLI (`hermes profile create`) — the third of the three
callers of the ``create_profile()`` chokepoint (alongside the REST route and the RPC), and the
only one that turns the refusal into a plain-text exit rather than a structured error."""

import argparse
from pathlib import Path

import pytest

from hermes_cli.profile_cmd import _profile_create


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Isolated environment for profile tests (mirrors tests/hermes_cli/test_profiles.py)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def _args(name: str) -> argparse.Namespace:
    return argparse.Namespace(profile_name=name)


class TestProfileCreateCliMaxLimit:
    def test_refused_at_the_limit_dies_with_the_shared_wording(self, profile_env, capsys):
        default_home = profile_env / ".hermes"
        (default_home / "config.yaml").write_text("profiles:\n  max: 1\n", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            _profile_create(_args("onemore"))

        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert (
            "Error: This gateway allows 1 profiles and already has 1. "
            "Delete a profile first, or raise profiles.max in config.yaml." in out
        )
        assert not (default_home / "profiles" / "onemore").exists()

    def test_succeeds_below_the_limit(self, profile_env, capsys):
        default_home = profile_env / ".hermes"
        (default_home / "config.yaml").write_text("profiles:\n  max: 2\n", encoding="utf-8")

        _profile_create(_args("roomleft"))

        assert (default_home / "profiles" / "roomleft").is_dir()
        assert "created at" in capsys.readouterr().out
