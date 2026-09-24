"""Every way a profile comes into existence gives it a mem0 identity of its own.

The identity is ``agent_id``: the scope a profile's memories are written under and recalled from. A
new profile used to get none, so the mem0 provider fell back to the built-in ``hermes`` -- the
default profile's identity -- or to whatever ``MEM0_AGENT_ID`` a clone or the credential mirror
copied out of another profile's ``.env``. Either way the new bot read and wrote another profile's
memory. Each path below is asserted through the provider itself, so what is checked is the
identity a served turn would actually run under.
"""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from agent import secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.memory.mem0 import Mem0MemoryProvider

DEFAULT_ENV = "MEM0_API_KEY=m0-test\nMEM0_AGENT_ID=hermes\nOPENAI_API_KEY=sk-test\n"
DEFAULT_CONFIG = "model:\n  provider: openrouter\n  default: some/model\nmemory:\n  provider: mem0\n"


class _Backend:
    def search(self, query, *, filters, top_k=10, rerank=False):
        return []

    def close(self):
        pass


@pytest.fixture()
def default_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(DEFAULT_CONFIG, encoding="utf-8")
    (home / ".env").write_text(DEFAULT_ENV, encoding="utf-8")
    (home / "mem0.json").write_text(json.dumps(
        {"user_id": "owner", "agent_id": "hermes", "search_agent_ids": ["hermes", "team"]}), encoding="utf-8")
    monkeypatch.setattr(Mem0MemoryProvider, "_create_backend", lambda self: _Backend())
    return home


@contextmanager
def _served(home):
    home_token = set_hermes_home_override(str(home))
    secret_token = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
    secret_scope.set_multiplex_active(True)
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(False)
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def identity(home) -> str:
    """The agent_id a turn served for *home* runs under."""
    with _served(home):
        provider = Mem0MemoryProvider()
        provider.initialize("session", user_id="human")
        return provider._agent_id


def scope_of(home) -> frozenset:
    with _served(home):
        provider = Mem0MemoryProvider()
        provider.initialize("session", user_id="human")
        return provider._search_agent_ids


def _own(home, default_home):
    agent_id = identity(home)
    assert agent_id != identity(default_home) == "hermes"
    assert agent_id.startswith(f"hermes-{home.name}-")
    return agent_id


class TestCreateProfile:
    def test_a_fresh_profile_has_its_own_identity(self, default_home):
        from hermes_cli.profiles import create_profile
        _own(create_profile("scout", no_alias=True), default_home)

    def test_a_clone_does_not_run_under_the_identity_its_copied_env_names(self, default_home):
        from hermes_cli.profiles import create_profile
        clone = create_profile("scout", clone_config=True, no_alias=True)
        assert "MEM0_AGENT_ID=hermes" in (clone / ".env").read_text(encoding="utf-8")  # copied, and outranked
        _own(clone, default_home)

    def test_a_clone_from_a_named_profile_gets_its_own_not_the_sources(self, default_home):
        from hermes_cli.profiles import create_profile
        source = create_profile("writer", no_alias=True)
        clone = create_profile("editor", clone_from="writer", no_alias=True)
        assert identity(clone) not in {identity(source), "hermes"}

    def test_a_full_clone_keeps_the_sources_sharing_but_not_its_identity(self, default_home):
        from hermes_cli.profiles import create_profile
        clone = create_profile("scout", clone_all=True, no_alias=True)
        agent_id = _own(clone, default_home)
        # The team layer the default reads on purpose is copied as the explicit setting it is;
        # the entry that meant "my own" now means the clone's own.
        assert scope_of(clone) == {agent_id, "team"}
        assert json.loads((clone / "mem0.json").read_text(encoding="utf-8"))["user_id"] == "owner"

    def test_a_clone_carries_the_mem0_settings_but_not_the_identity_or_the_history(self, default_home):
        """``--clone`` copies the active memory provider's own config, so a clone of a mem0 profile arrives
        with its ``mem0.json``. It still runs under an identity of its own, and ``mem0/`` stays behind:
        that directory holds the source's OSS history database -- its memories' past texts, not config."""
        from hermes_cli.profiles import create_profile
        (default_home / "mem0").mkdir()
        (default_home / "mem0" / "history.db").write_bytes(b"the default profile's memory history")
        clone = create_profile("scout", clone_config=True, no_alias=True)
        agent_id = _own(clone, default_home)
        assert scope_of(clone) == {agent_id, "team"}
        assert json.loads((clone / "mem0.json").read_text(encoding="utf-8"))["user_id"] == "owner"
        assert not (clone / "mem0").exists()

    def test_two_new_profiles_never_share_one(self, default_home):
        from hermes_cli.profiles import create_profile
        first = identity(create_profile("one", no_alias=True))
        second = identity(create_profile("two", no_alias=True))
        assert first != second

    def test_a_recreated_name_does_not_inherit_the_deleted_profiles_memory(self, default_home):
        from hermes_cli.profiles import create_profile, delete_profile
        before = identity(create_profile("scout", no_alias=True))
        delete_profile("scout", yes=True)
        assert identity(create_profile("scout", no_alias=True)) != before

    def test_the_identity_file_is_private(self, default_home):
        from hermes_cli.profiles import create_profile
        profile = create_profile("scout", no_alias=True)
        assert (profile / "mem0.json").stat().st_mode & 0o077 == 0


def test_cli_profile_create(default_home):
    from hermes_cli.profile_cmd import _profile_create
    _profile_create(argparse.Namespace(profile_name="scout", no_alias=True))
    _own(default_home / "profiles" / "scout", default_home)


def test_cli_profile_create_clone(default_home):
    from hermes_cli.profile_cmd import _profile_create
    _profile_create(argparse.Namespace(profile_name="scout", clone=True, no_alias=True))
    _own(default_home / "profiles" / "scout", default_home)


@pytest.mark.parametrize("params", [
    {},                                            # what the app sends without a clone source
    {"clone_from": "default"},                     # what it sends with one
    {"model": "some/model", "provider": "openrouter"},
    {"mirror_credentials": False},
])
def test_rpc_profiles_create(default_home, params):
    """The desktop/app twin: a fresh bot also gets the launch profile's .env mirrored in."""
    from tui_gateway import server
    resp = server._methods["profiles.create"]("r1", {"name": "scout", "no_alias": True, **params})
    assert resp.get("error") is None, resp
    _own(default_home / "profiles" / "scout", default_home)


def test_an_imported_profile_gets_its_own_identity(default_home, tmp_path):
    from hermes_cli.profiles import create_profile, export_profile, import_profile
    source = create_profile("writer", no_alias=True)
    archive = export_profile("writer", str(tmp_path / "writer.tar.gz"))
    imported = import_profile(str(archive), name="copy")
    assert identity(imported) not in {identity(source), "hermes"}


def test_an_imported_legacy_profile_does_not_join_the_default(default_home, tmp_path):
    """An archive of a profile that ran on the old shared fallback carries that pin; it is not kept."""
    from hermes_cli.profiles import export_profile, import_profile
    legacy = default_home / "profiles" / "old"
    legacy.mkdir(parents=True)
    (legacy / "config.yaml").write_text("memory:\n  provider: mem0\n", encoding="utf-8")
    pin = {"agent_id": "hermes", "agent_id_source": "legacy-default"}
    (legacy / "mem0.json").write_text(json.dumps(pin), encoding="utf-8")
    archive = export_profile("old", str(tmp_path / "old.tar.gz"))
    imported = import_profile(str(archive), name="copy")
    assert identity(imported) != "hermes"


def test_a_distribution_install_never_uses_an_identity_baked_into_the_distribution(default_home, tmp_path):
    from hermes_cli.profile_distribution import DistributionManifest, install_distribution, update_distribution, write_manifest
    staged = tmp_path / "dist"
    staged.mkdir()
    (staged / "SOUL.md").write_text("I am a template.\n", encoding="utf-8")
    (staged / "config.yaml").write_text("memory:\n  provider: mem0\n", encoding="utf-8")
    (staged / "mem0.json").write_text(json.dumps({"mode": "oss", "agent_id": "template",
                                                  "search_agent_ids": ["template", "team"]}), encoding="utf-8")
    write_manifest(staged, DistributionManifest(name="dist", version="0.1.0"))

    first = install_distribution(str(staged), name="one").target_dir
    second = install_distribution(str(staged), name="two").target_dir
    ids = {identity(first), identity(second)}
    assert len(ids) == 2 and not ids & {"template", "hermes"}
    assert scope_of(first) == {identity(first), "team"}

    # An update re-copies the distribution's files; the profile keeps the identity it has.
    before = identity(first)
    update_distribution("one")
    assert identity(first) == before
    assert scope_of(first) == {before, "team"}


def test_a_profile_that_does_not_use_mem0_still_gets_its_own_identity(default_home):
    """Written whatever the provider, so a profile that turns mem0 on later is already isolated."""
    from hermes_cli.profiles import create_profile
    (default_home / "config.yaml").write_text("model:\n  provider: openrouter\n  default: some/model\n",
                                              encoding="utf-8")
    profile = create_profile("plain", no_alias=True)
    assert "mem0" not in (profile / "config.yaml").read_text(encoding="utf-8")
    assert json.loads((profile / "mem0.json").read_text(encoding="utf-8"))["agent_id"].startswith("hermes-plain-")


def test_a_full_clone_of_a_source_whose_identity_is_in_the_process_env(default_home, monkeypatch):
    """A systemd ``Environment=MEM0_AGENT_ID=...`` source: the clone's scope must name the clone, or
    its mem0 fails closed ("must include agent_id")."""
    from hermes_cli.profiles import create_profile
    (default_home / ".env").write_text("MEM0_API_KEY=m0-test\n", encoding="utf-8")
    (default_home / "mem0.json").write_text(json.dumps({"search_agent_ids": ["from-env", "team"]}), encoding="utf-8")
    monkeypatch.setenv("MEM0_AGENT_ID", "from-env")
    clone = create_profile("scout", clone_all=True, no_alias=True)
    agent_id = identity(clone)
    assert agent_id.startswith("hermes-scout-")
    assert scope_of(clone) == {agent_id, "team"}


def _distribution(tmp_path):
    from hermes_cli.profile_distribution import DistributionManifest, write_manifest
    staged = tmp_path / "dist"
    staged.mkdir()
    (staged / "SOUL.md").write_text("I am a template.\n", encoding="utf-8")
    (staged / "config.yaml").write_text("memory:\n  provider: mem0\n", encoding="utf-8")
    (staged / "mem0.json").write_text(json.dumps({"mode": "oss", "agent_id": "template"}), encoding="utf-8")
    write_manifest(staged, DistributionManifest(name="dist", version="0.1.0"))
    return staged


@pytest.mark.parametrize("again", ["reinstall", "update"])
def test_a_distribution_profile_keeps_an_identity_it_has_in_env(default_home, tmp_path, again):
    from hermes_cli.profile_distribution import install_distribution, update_distribution
    staged = _distribution(tmp_path)
    target = install_distribution(str(staged), name="one").target_dir
    # The operator moved the identity to .env.
    cfg = json.loads((target / "mem0.json").read_text(encoding="utf-8"))
    cfg.pop("agent_id")
    (target / "mem0.json").write_text(json.dumps(cfg), encoding="utf-8")
    (target / ".env").write_text("MEM0_API_KEY=m0-test\nMEM0_AGENT_ID=one-own\n", encoding="utf-8")
    assert identity(target) == "one-own"

    if again == "reinstall":
        install_distribution(str(staged), name="one", force=True)
    else:
        update_distribution("one")

    assert identity(target) == "one-own"


def test_a_reinstall_leaves_a_malformed_mem0_json_of_the_profile_alone(default_home, tmp_path):
    from hermes_cli.profile_distribution import DistributionManifest, install_distribution, write_manifest
    staged = tmp_path / "dist"
    staged.mkdir()
    (staged / "SOUL.md").write_text("I am a template.\n", encoding="utf-8")
    (staged / "config.yaml").write_text("memory:\n  provider: mem0\n", encoding="utf-8")
    write_manifest(staged, DistributionManifest(name="dist", version="0.1.0"))
    target = install_distribution(str(staged), name="one").target_dir
    raw = '{"host": "http://mem0.local", "agent_id": "one-own",}\n'
    (target / "mem0.json").write_text(raw, encoding="utf-8")

    install_distribution(str(staged), name="one", force=True)

    assert (target / "mem0.json").read_text(encoding="utf-8") == raw


def test_a_malformed_mem0_json_copied_into_a_new_profile_is_kept_aside(default_home):
    from hermes_cli.profiles import create_profile
    raw = '{"user_id": "owner",}\n'
    (default_home / "mem0.json").write_text(raw, encoding="utf-8")
    clone = create_profile("scout", clone_all=True, no_alias=True)
    assert (clone / "mem0.json.invalid").read_text(encoding="utf-8") == raw
    assert (default_home / "mem0.json").read_text(encoding="utf-8") == raw
    _own(clone, default_home)
