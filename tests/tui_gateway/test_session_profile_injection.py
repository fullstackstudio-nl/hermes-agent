"""Contract test: tui_gateway._set_session_context must bind the session's own
profile into HERMES_SESSION_PROFILE.

_set_session_context passed no ``profile`` to set_session_vars, so every
dashboard/WebSocket turn bound "" even though the record already knows its
``profile_home`` and HERMES_HOME plainly names the profile. Readers then fall
back to whatever the process happens to be: the persistent-Docker container key
(tools/terminal_tool._resolve_container_task_id) collapses onto the launch
profile's container, and the kanban notifier
(tools/kanban_tools) attributes a notification to the launch profile.

The name comes from the record's own home, not from the process, because a
multiplexed gateway serves every profile from one process. Sibling guard for the
API-server route: tests/gateway/test_multiplex_api_server_routing.py.
"""
import types

import pytest

from gateway.session_context import (
    _UNSET,
    _VAR_MAP,
    get_session_env,
)
import tui_gateway.server as server


@pytest.fixture(autouse=True)
def _reset_contextvars():
    """Reset all session contextvars to _UNSET between tests (see
    test_session_id_injection.py: tests share one thread context)."""
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)


class _FakeAgent:
    def __init__(self, session_id):
        self.session_id = session_id


def _install_session(monkeypatch, *, session_key, **extra):
    sess = {
        "session_key": session_key,
        "source": "desktop",
        "agent": _FakeAgent("20260921_cafebabe"),
        "cwd": "/home/user",
        "transport": types.SimpleNamespace(),
        **extra,
    }
    monkeypatch.setattr(server, "_sessions", {session_key: sess}, raising=False)
    return sess


def _served_home(tmp_path, name):
    """A stored profile home: ``<root>/profiles/<name>`` is authoritative for the
    profile id (hermes_constants.profile_name_for_home)."""
    home = tmp_path / "profiles" / name
    home.mkdir(parents=True)
    return str(home)


def test_binds_the_profile_that_owns_the_session(monkeypatch, tmp_path):
    """A record served for another profile must name THAT profile, never the
    process's own — the whole point once one process serves several."""
    monkeypatch.setattr(server, "_current_profile_name", lambda: "launch-profile")
    _install_session(
        monkeypatch, session_key="skey-served",
        profile_home=_served_home(tmp_path, "team-two"),
    )

    tokens = server._set_session_context("skey-served")
    try:
        assert get_session_env("HERMES_SESSION_PROFILE") == "team-two"
    finally:
        server._clear_session_context(tokens)

    assert get_session_env("HERMES_SESSION_PROFILE") == ""


def test_a_launch_profile_record_names_the_active_profile(monkeypatch):
    """``profile_home`` is None for a session on the launch profile; the active
    home is then the only profile there is."""
    monkeypatch.setattr(server, "_current_profile_name", lambda: "launch-profile")
    _install_session(monkeypatch, session_key="skey-launch", profile_home=None)

    tokens = server._set_session_context("skey-launch")
    try:
        assert get_session_env("HERMES_SESSION_PROFILE") == "launch-profile"
    finally:
        server._clear_session_context(tokens)


def test_an_ephemeral_task_id_still_names_a_profile(monkeypatch):
    """Ephemeral task ids are not in ``_sessions``. They must not bind "" —
    an explicitly empty contextvar is authoritative (no os.environ fallback),
    so a subprocess would see no profile at all."""
    monkeypatch.setattr(server, "_current_profile_name", lambda: "launch-profile")
    monkeypatch.setattr(server, "_sessions", {}, raising=False)

    tokens = server._set_session_context("task-ephemeral")
    try:
        assert get_session_env("HERMES_SESSION_PROFILE") == "launch-profile"
    finally:
        server._clear_session_context(tokens)


def test_the_bound_profile_selects_the_profile_scoped_container_key(monkeypatch, tmp_path):
    """The real reader chain, not just the variable: a persistent-Docker sandbox
    is keyed on the bound profile, so an unbound one let a served profile's turn
    reuse the launch profile's container.

    ``terminal_scope(None)`` is the precondition, not decoration: with a policy
    scope bound, ``_tenv`` reads ONLY that policy and the ``TERMINAL_*`` env this
    test sets would be ignored (tests share one thread context, so a scope an
    earlier test left behind would otherwise decide the backend here)."""
    import tools.terminal_tool as tt
    from tools.terminal_scope import terminal_scope

    monkeypatch.setattr(server, "_current_profile_name", lambda: "launch-profile")
    monkeypatch.setattr(tt, "_ensure_terminal_env_bridged", lambda: None)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    monkeypatch.delenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", raising=False)
    _install_session(
        monkeypatch, session_key="skey-docker",
        profile_home=_served_home(tmp_path, "team-two"),
    )

    tokens = server._set_session_context("skey-docker")
    try:
        with terminal_scope(None):
            assert tt._current_session_profile() == "team-two"
            assert tt._resolve_container_task_id(None) == "profile:team-two"
    finally:
        server._clear_session_context(tokens)
