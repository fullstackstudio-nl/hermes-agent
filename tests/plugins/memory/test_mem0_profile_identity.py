"""Which mem0 identity a profile resolves at start, and what the provider writes down about it.

A named profile that never had an ``agent_id`` of its own used to resolve the built-in ``hermes``,
the same id the default profile runs under, and so read and wrote the default profile's memory
without anyone having configured that. These tests pin what happens to such a profile now: it keeps
the identity it was already using, but explicitly, in its own ``mem0.json``, with a warning -- and
that every identity somebody did configure, sharing included, resolves exactly as before.
"""
import json
import logging
import os
from contextlib import contextmanager

import pytest

from agent import secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.memory.mem0 import Mem0MemoryProvider


class _Backend:
    def search(self, query, *, filters, top_k=10, rerank=False):
        return []

    def close(self):
        pass


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Mem0MemoryProvider, "_create_backend", lambda self: _Backend())
    return root


@contextmanager
def served(home):
    """The multiplexer's view of one profile: its own home and only its own secrets."""
    home_token = set_hermes_home_override(str(home))
    secret_token = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
    secret_scope.set_multiplex_active(True)
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(False)
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def start(home) -> Mem0MemoryProvider:
    with served(home):
        provider = Mem0MemoryProvider()
        provider.initialize("session", user_id="human")
        return provider


def named(root, name, *, mem0=None, env="", provider="mem0", config_yaml=""):
    """A named profile; *config_yaml* None means it has no config.yaml of its own."""
    home = root / "profiles" / name
    home.mkdir(parents=True)
    if config_yaml is not None:
        (home / "config.yaml").write_text(config_yaml or f"memory:\n  provider: {provider}\n", encoding="utf-8")
    (home / ".env").write_text(env, encoding="utf-8")
    if mem0 is not None:
        (home / "mem0.json").write_text(json.dumps(mem0), encoding="utf-8")
    return home


def _mem0_warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING and "mem0" in r.getMessage().lower()]


def test_existing_named_profile_without_an_identity_keeps_it_but_explicitly_and_is_warned(root, caplog):
    home = named(root, "bot")

    with caplog.at_level(logging.WARNING):
        provider = start(home)

    # Same identity as before the upgrade: nothing it could recall yesterday is gone today.
    assert provider._agent_id == "hermes"
    # ...but it is no longer implied. It is written into the profile's own config, marked as the
    # old shared fallback, so an operator can find it and a later start does not decide it again.
    written = json.loads((home / "mem0.json").read_text(encoding="utf-8"))
    assert written["agent_id"] == "hermes"
    assert written["agent_id_source"] == "legacy-default"
    warnings = _mem0_warnings(caplog)
    assert warnings, "pinning another profile's identity must not happen silently"
    message = warnings[0].getMessage()
    assert "bot" in message and "hermes" in message and str(home / "mem0.json") in message


def test_a_pinned_profile_is_warned_again_on_the_next_start(root, caplog):
    from plugins.memory.mem0 import _identity
    home = named(root, "bot", mem0={"agent_id": "hermes", "agent_id_source": "legacy-default"})
    _identity._warned_homes.clear()  # a fresh process

    with caplog.at_level(logging.WARNING):
        assert start(home)._agent_id == "hermes"

    assert _mem0_warnings(caplog)


def test_pinning_keeps_the_rest_of_the_profiles_mem0_config(root):
    home = named(root, "bot", mem0={"mode": "oss", "user_id": "human", "search_agent_ids": ["hermes", "team"]})

    provider = start(home)

    assert provider._agent_id == "hermes"
    assert provider._search_agent_ids == {"hermes", "team"}
    written = json.loads((home / "mem0.json").read_text(encoding="utf-8"))
    assert written["mode"] == "oss" and written["search_agent_ids"] == ["hermes", "team"]


def test_the_default_profile_keeps_the_builtin_identity_and_nothing_is_written(root, caplog):
    with caplog.at_level(logging.WARNING):
        provider = start(root)

    assert provider._agent_id == "hermes"
    assert not (root / "mem0.json").exists()
    assert not _mem0_warnings(caplog)


@pytest.mark.parametrize("config", [
    {"mem0": {"agent_id": "hermes"}},             # chosen on purpose, e.g. through the setup wizard
    {"mem0": {"agent_id": "hermes-bot-1a2b3c4d"}},
    {"env": "MEM0_AGENT_ID=researcher\n"},
])
def test_a_configured_identity_is_used_as_is_and_never_rewritten(root, caplog, config):
    home = named(root, "bot", **config)
    before = (home / "mem0.json").read_text(encoding="utf-8") if (home / "mem0.json").exists() else None

    with caplog.at_level(logging.WARNING):
        provider = start(home)

    expected = config.get("mem0", {}).get("agent_id") or "researcher"
    assert provider._agent_id == expected
    after = (home / "mem0.json").read_text(encoding="utf-8") if (home / "mem0.json").exists() else None
    assert after == before
    assert not _mem0_warnings(caplog)


def test_explicit_sharing_through_search_agent_ids_works_exactly_as_before(root):
    """Two profiles that share a team layer on purpose: each writes its own, both read the team's."""
    alpha = named(root, "alpha", mem0={"agent_id": "alpha", "search_agent_ids": ["alpha", "team"]})
    beta = named(root, "beta", mem0={"agent_id": "beta", "search_agent_ids": ["beta", "team"]})

    a, b = start(alpha), start(beta)

    assert (a._agent_id, b._agent_id) == ("alpha", "beta")
    assert a._search_agent_ids == {"alpha", "team"}
    assert b._search_agent_ids == {"beta", "team"}
    for home in (alpha, beta):
        assert "agent_id_source" not in json.loads((home / "mem0.json").read_text(encoding="utf-8"))


def test_a_profile_that_explicitly_shares_the_default_identity_still_does(root):
    """Sharing the default's own memory is allowed -- when it is written down, not when it is implied."""
    home = named(root, "twin", mem0={"agent_id": "hermes"})

    assert start(home)._agent_id == start(root)._agent_id == "hermes"


# -- The pin only ever adds to a file it can read, parse and owns --------------------------------

_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


def test_a_malformed_mem0_json_is_left_byte_identical(root, caplog):
    """A hand edit with a trailing comma must not be replaced by the pin (and the operator's host,
    user and sharing settings with it)."""
    raw = '{"host": "http://mem0.local", "user_id": "human", "search_agent_ids": ["hermes", "team"],}\n'
    home = named(root, "bot")
    (home / "mem0.json").write_text(raw, encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        assert start(home)._agent_id == "hermes"

    assert (home / "mem0.json").read_text(encoding="utf-8") == raw
    assert any(str(home / "mem0.json") in r.getMessage() and "untouched" in r.getMessage()
               for r in _mem0_warnings(caplog))


@pytest.mark.skipif(_ROOT or os.name != "posix", reason="needs a file this user cannot read")
def test_an_unreadable_mem0_json_is_not_replaced(root, caplog):
    home = named(root, "bot", mem0={"host": "http://mem0.local", "user_id": "human"})
    path = home / "mem0.json"
    before = path.read_bytes()
    path.chmod(0)
    try:
        with caplog.at_level(logging.WARNING):
            assert start(home)._agent_id == "hermes"
    finally:
        path.chmod(0o600)
    assert path.read_bytes() == before
    assert any("untouched" in r.getMessage() for r in _mem0_warnings(caplog))


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership")
def test_a_mem0_json_owned_by_another_user_is_not_replaced(root, monkeypatch, caplog):
    from plugins.memory.mem0 import _identity
    home = named(root, "bot", mem0={"host": "http://mem0.local"})
    before = (home / "mem0.json").read_bytes()
    monkeypatch.setattr(_identity.os, "geteuid", lambda: os.stat(home / "mem0.json").st_uid + 1)

    with caplog.at_level(logging.WARNING):
        assert start(home)._agent_id == "hermes"
        assert start(home)._agent_id == "hermes"

    assert (home / "mem0.json").read_bytes() == before
    # A gateway running as another user than the file's never pins, and says so on every start.
    assert sum("another user" in r.getMessage() for r in _mem0_warnings(caplog)) == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes")
def test_the_pin_keeps_the_files_mode(root):
    home = named(root, "bot", mem0={"mode": "oss"})
    (home / "mem0.json").chmod(0o644)

    start(home)

    assert json.loads((home / "mem0.json").read_text(encoding="utf-8"))["agent_id"] == "hermes"
    assert (home / "mem0.json").stat().st_mode & 0o777 == 0o644


@pytest.mark.skipif(_ROOT or os.name != "posix", reason="needs a directory this user cannot write")
def test_a_read_only_home_runs_as_before_and_says_the_pin_could_not_be_written(root, caplog):
    home = named(root, "bot")
    home.chmod(0o500)
    try:
        with caplog.at_level(logging.WARNING):
            assert start(home)._agent_id == "hermes"
    finally:
        home.chmod(0o700)
    assert not (home / "mem0.json").exists()
    messages = [r.getMessage() for r in _mem0_warnings(caplog)]
    assert any("could not write" in m for m in messages)
    assert any("could not be written down" in m for m in messages)


def test_the_pin_never_overwrites_an_identity_saved_meanwhile():
    """The write re-reads the file in the same step; an agent_id found there stands."""
    from plugins.memory.mem0 import _identity
    assert _identity._pin({"agent_id": "saved-by-setup"}) is None
    assert _identity._pin({"mode": "oss"}) == {"mode": "oss", "agent_id": "hermes", "agent_id_source": "legacy-default"}


def test_the_pin_in_a_single_profile_process(root, monkeypatch):
    """`hermes -p bot ...`: HERMES_HOME is the profile, no multiplexer, no secret scope."""
    home = named(root, "bot")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("MEM0_AGENT_ID", raising=False)

    provider = Mem0MemoryProvider()
    provider.initialize("session", user_id="human")

    assert provider._agent_id == "hermes"
    assert json.loads((home / "mem0.json").read_text(encoding="utf-8"))["agent_id_source"] == "legacy-default"


# -- An operator's own identity after a pin -----------------------------------------------------

def test_an_identity_set_in_env_after_a_pin_wins_and_the_pin_goes(root, caplog):
    home = named(root, "bot", mem0={"mode": "oss", "agent_id": "hermes", "agent_id_source": "legacy-default"},
                 env="MEM0_AGENT_ID=bot-own\n")

    with caplog.at_level(logging.WARNING):
        assert start(home)._agent_id == "bot-own"
        assert start(home)._agent_id == "bot-own"

    assert json.loads((home / "mem0.json").read_text(encoding="utf-8")) == {"mode": "oss"}
    assert not _mem0_warnings(caplog)


def test_saving_an_identity_of_its_own_through_setup_drops_the_pin(root):
    home = named(root, "bot", mem0={"agent_id": "hermes", "agent_id_source": "legacy-default"})
    Mem0MemoryProvider().save_config({"agent_id": "bot-own"}, str(home))
    assert json.loads((home / "mem0.json").read_text(encoding="utf-8")) == {"agent_id": "bot-own"}
    # Saving the form unchanged keeps the pin, and with it the warning.
    home2 = named(root, "bot2", mem0={"agent_id": "hermes", "agent_id_source": "legacy-default"})
    Mem0MemoryProvider().save_config({"agent_id": "hermes", "rerank": "false"}, str(home2))
    assert json.loads((home2 / "mem0.json").read_text(encoding="utf-8"))["agent_id_source"] == "legacy-default"


# -- The dashboard's memory setup form ----------------------------------------------------------

def _schema_default(home):
    with served(home):
        return next(f["default"] for f in Mem0MemoryProvider().get_config_schema() if f["key"] == "agent_id")


def test_the_dashboard_offers_a_named_profile_not_yet_on_mem0_its_own_identity(root):
    assert _schema_default(named(root, "bot", provider="''")).startswith("hermes-bot-")
    assert _schema_default(named(root, "mine", mem0={"agent_id": "hermes-mine-1a2b3c4d"})) == "hermes-mine-1a2b3c4d"
    assert _schema_default(root) == "hermes"


def test_the_dashboard_offers_a_profile_already_running_on_the_fallback_what_it_runs_under(root):
    """A form default is written by every save; offering a new id would switch its memory."""
    assert _schema_default(named(root, "old")) == "hermes"


def _dashboard_save(home, values):
    from hermes_cli.web_routers.memory_providers import _write_memory_provider_config_values
    with served(home):
        _write_memory_provider_config_values("mem0", Mem0MemoryProvider(), values)


def test_a_dashboard_save_of_another_setting_keeps_an_old_profile_pinned(root, caplog):
    home = named(root, "old")  # on mem0, ran under the fallback, no session since the update

    _dashboard_save(home, {"rerank": "true"})

    saved = json.loads((home / "mem0.json").read_text(encoding="utf-8"))
    assert (saved["agent_id"], saved["agent_id_source"], saved["rerank"]) == ("hermes", "legacy-default", "true")
    with caplog.at_level(logging.WARNING):
        assert start(home)._agent_id == "hermes"
    assert _mem0_warnings(caplog)


def test_a_dashboard_save_of_an_identity_of_its_own_unpins_and_moves_the_scope(root):
    home = named(root, "old", mem0={"agent_id": "hermes", "agent_id_source": "legacy-default",
                                     "search_agent_ids": ["hermes", "team"]})

    _dashboard_save(home, {"agent_id": "old-own"})

    saved = json.loads((home / "mem0.json").read_text(encoding="utf-8"))
    assert saved["agent_id"] == "old-own" and "agent_id_source" not in saved
    assert saved["search_agent_ids"] == ["old-own", "team"]
    assert start(home)._search_agent_ids == {"old-own", "team"}


def test_unpinning_through_env_moves_an_explicit_scope_too(root):
    home = named(root, "bot", mem0={"agent_id": "hermes", "agent_id_source": "legacy-default",
                                     "search_agent_ids": ["hermes", "team"]}, env="MEM0_AGENT_ID=bot-own\n")

    provider = start(home)

    assert (provider._agent_id, provider._search_agent_ids) == ("bot-own", {"bot-own", "team"})
    assert json.loads((home / "mem0.json").read_text(encoding="utf-8")) == {"search_agent_ids": ["bot-own", "team"]}


def test_a_process_wide_identity_does_not_unpin(root, monkeypatch):
    """A systemd Environment=MEM0_AGENT_ID in a single-profile process is not the profile's own
    setting: un-pinning on it would pin again at the next multiplexed start."""
    home = named(root, "bot", mem0={"agent_id": "hermes", "agent_id_source": "legacy-default"})
    before = (home / "mem0.json").read_bytes()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("MEM0_AGENT_ID", "from-the-service")

    provider = Mem0MemoryProvider()
    provider.initialize("session", user_id="human")

    assert provider._agent_id == "hermes"
    assert (home / "mem0.json").read_bytes() == before


def test_an_identity_saved_between_the_look_and_the_pin_is_the_one_that_runs(root, monkeypatch, caplog):
    from plugins.memory.mem0 import _identity
    home = named(root, "bot", mem0={"agent_id": "saved-meanwhile"})
    monkeypatch.setattr(_identity, "read_json_or_empty", lambda path: {})  # what the start saw before the save

    with caplog.at_level(logging.WARNING):
        assert _identity.resolve_agent_id({"agent_id": "hermes"}, home) == "saved-meanwhile"

    assert json.loads((home / "mem0.json").read_text(encoding="utf-8")) == {"agent_id": "saved-meanwhile"}
    assert not _mem0_warnings(caplog)


def test_an_explicit_hermes_beside_an_env_identity_is_left_alone(root, caplog):
    """No marker means somebody chose it; mem0.json outranks .env, as it always has."""
    home = named(root, "twin", mem0={"agent_id": "hermes"}, env="MEM0_AGENT_ID=twin-own\n")
    before = (home / "mem0.json").read_bytes()

    with caplog.at_level(logging.WARNING):
        assert start(home)._agent_id == "hermes"

    assert (home / "mem0.json").read_bytes() == before
    assert not _mem0_warnings(caplog)



# -- What a setup save offers and marks, on the safe side ---------------------------------------

@pytest.mark.parametrize("config_yaml", [None, "memory: [\n  provider: mem0\n"], ids=["no-config", "unreadable"])
def test_a_profile_whose_config_cannot_be_read_is_offered_hermes_and_a_save_keeps_it_pinned(root, config_yaml):
    home = named(root, "old", config_yaml=config_yaml)

    assert _schema_default(home) == "hermes"
    _dashboard_save(home, {"rerank": "true"})

    saved = json.loads((home / "mem0.json").read_text(encoding="utf-8"))
    assert (saved["agent_id"], saved["agent_id_source"]) == ("hermes", "legacy-default")


def test_the_form_offers_the_identity_a_service_environment_runs_the_profile_under(root, monkeypatch, caplog):
    """Single-profile process with a systemd Environment=MEM0_AGENT_ID: the default shown, and what a
    save of another setting writes, is that identity -- not hermes, and not marked as the pin."""
    from hermes_cli.web_routers.memory_providers import _write_memory_provider_config_values
    home = named(root, "old")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("MEM0_AGENT_ID", "from-the-service")

    assert next(f["default"] for f in Mem0MemoryProvider().get_config_schema()
                if f["key"] == "agent_id") == "from-the-service"
    _write_memory_provider_config_values("mem0", Mem0MemoryProvider(), {"rerank": "true"})

    saved = json.loads((home / "mem0.json").read_text(encoding="utf-8"))
    assert saved["agent_id"] == "from-the-service" and "agent_id_source" not in saved
    with caplog.at_level(logging.WARNING):
        provider = Mem0MemoryProvider()
        provider.initialize("session", user_id="human")
    assert provider._agent_id == "from-the-service"
    assert not _mem0_warnings(caplog)


def test_typing_hermes_where_its_own_was_offered_is_not_marked(root):
    home = named(root, "plain", provider="''")
    assert _schema_default(home).startswith("hermes-plain-")

    _dashboard_save(home, {"agent_id": "hermes"})

    assert "agent_id_source" not in json.loads((home / "mem0.json").read_text(encoding="utf-8"))


def test_a_new_identity_saved_before_the_first_pin_moves_the_scope_too(root):
    """A legacy profile not started since the update: no agent_id yet, explicit ["hermes", "team"]."""
    home = named(root, "old", mem0={"search_agent_ids": ["hermes", "team", "hermes"]})

    _dashboard_save(home, {"agent_id": "old-own"})

    saved = json.loads((home / "mem0.json").read_text(encoding="utf-8"))
    assert saved["search_agent_ids"] == ["old-own", "team"]
    assert start(home)._search_agent_ids == {"old-own", "team"}



def test_the_launch_profiles_service_identity_is_never_offered_to_another_profile(root, monkeypatch, caplog):
    """A dashboard process running as the default profile, MEM0_AGENT_ID=default-own in its service
    environment, edits named profile X by parameter: home override plus X's secret scope, and in a
    single-profile process a scope miss falls through to os.environ. X must be offered hermes -- what it
    runs under -- not the default's identity, and the save must keep it the pin."""
    from hermes_cli.web_routers.memory_providers import _write_memory_provider_config_values
    home = named(root, "x")
    monkeypatch.setenv("MEM0_AGENT_ID", "default-own")  # HERMES_HOME is the default profile's root

    @contextmanager
    def edited_from_the_dashboard():
        home_token = set_hermes_home_override(str(home))
        secret_token = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
        try:
            assert secret_scope.get_secret("MEM0_AGENT_ID", "") == "default-own"  # the fall-through
            yield
        finally:
            secret_scope.reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    with edited_from_the_dashboard():
        default = next(f["default"] for f in Mem0MemoryProvider().get_config_schema() if f["key"] == "agent_id")
        _write_memory_provider_config_values("mem0", Mem0MemoryProvider(), {"rerank": "true"})

    assert default == "hermes"
    saved = json.loads((home / "mem0.json").read_text(encoding="utf-8"))
    assert (saved["agent_id"], saved["agent_id_source"]) == ("hermes", "legacy-default")
    # The launch profile itself is still offered the identity its service environment gives it.
    assert _identity_default_for(root) == "default-own"


def _identity_default_for(home):
    from plugins.memory.mem0._identity import setup_default_agent_id
    return setup_default_agent_id(home)
