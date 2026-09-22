"""The mem0 shared memory layer: its name is a setting, and only recall uses it.

Recall is scoped to ``user_id`` and ``agent_id`` at the backend, one query per allowed
agent, so a profile sees its own agent plus one shared layer. The layer's name must be
configurable: a host may want one house layer, a layer per customer or a layer per
department, and a default baked into the code cannot serve all three.

The name is validated exactly like an explicitly configured ``search_agent_ids`` entry,
so a blank or wildcard name fails the profile closed rather than widening recall.

Writes are deliberately NOT part of this: ``_add`` always attaches the writing profile's
own ``agent_id``, so the shared layer is populated by running something whose
``MEM0_AGENT_ID`` *is* that name, never as a side effect of a normal turn.
"""
from __future__ import annotations

import json

import pytest

import plugins.memory.mem0 as mem0

_TENANT = "tenant-one"


def _row(ident, memory, agent_id, score, user_id=_TENANT):
    return {"id": ident, "memory": memory, "agent_id": agent_id, "user_id": user_id, "score": score}


class _FakeBackend:
    """Records what was searched/added and replays a fixed result set.

    It honours ``filters`` the way a real backend does, because recall now pushes both
    ``user_id`` and ``agent_id`` down instead of over-fetching and trimming.
    """

    def __init__(self, results=()):
        self.results = [dict(r) for r in results]
        self.searches: list[dict] = []
        self.added: list[dict] = []

    def search(self, query, filters=None, top_k=10, rerank=False):
        self.searches.append({"query": query, "filters": filters, "top_k": top_k})
        rows = [dict(r) for r in self.results]
        for key, value in (filters or {}).items():
            rows = [r for r in rows if r.get(key) == value]
        return rows[:top_k]

    def add(self, messages, user_id=None, agent_id=None, infer=True, metadata=None):
        self.added.append({"user_id": user_id, "agent_id": agent_id})
        return {"results": []}


@pytest.fixture(autouse=True)
def _no_inherited_secret_scope():
    """``_load_config`` resolves MEM0_* through ``get_secret``, which reads an
    installed scope INSTEAD of os.environ. Tests share one thread context, so a
    scope an earlier test left bound would silently shadow every value these
    tests set. Pin "no scope" — then get_secret falls through to os.environ."""
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    token = set_secret_scope(None)
    try:
        yield
    finally:
        reset_secret_scope(token)


def _provider(monkeypatch, tmp_path, *, file_cfg=None, env=None, backend=None):
    """A provider initialized against an isolated HERMES_HOME, with a fake backend."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("MEM0_AGENT_ID", "MEM0_SHARED_AGENT_ID", "MEM0_USER_ID", "MEM0_HOST", "MEM0_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEM0_USER_ID", _TENANT)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    if file_cfg is not None:
        (tmp_path / "mem0.json").write_text(json.dumps(file_cfg))
    fake = backend if backend is not None else _FakeBackend()
    monkeypatch.setattr(mem0.Mem0MemoryProvider, "_create_backend", lambda self: fake)
    provider = mem0.Mem0MemoryProvider()
    provider.initialize("20260922_sess")
    return provider, fake


class TestSharedLayerName:

    def test_the_default_layer_is_named_shared(self, monkeypatch, tmp_path):
        """No configuration: a profile recalls from its own agent and from ``shared``."""
        provider, _ = _provider(monkeypatch, tmp_path, env={"MEM0_AGENT_ID": "team-two"})

        assert provider._shared_agent_id == "shared"
        assert provider._search_agent_ids == {"team-two", "shared"}

    def test_the_layer_name_is_configurable_per_profile(self, monkeypatch, tmp_path):
        """The env var is read through the profile secret scope like MEM0_AGENT_ID, so under
        multiplexing each profile's own .env names its own layer."""
        provider, _ = _provider(
            monkeypatch, tmp_path,
            env={"MEM0_AGENT_ID": "team-two", "MEM0_SHARED_AGENT_ID": "house-layer"})

        assert provider._shared_agent_id == "house-layer"
        assert provider._search_agent_ids == {"team-two", "house-layer"}

    def test_mem0_json_overrides_the_env_var(self, monkeypatch, tmp_path):
        """Same layering as every other mem0 setting: the file wins over the env default."""
        provider, _ = _provider(
            monkeypatch, tmp_path,
            env={"MEM0_AGENT_ID": "team-two", "MEM0_SHARED_AGENT_ID": "from-env"},
            file_cfg={"shared_agent_id": "from-file"})

        assert provider._search_agent_ids == {"team-two", "from-file"}

    def test_an_explicit_allow_list_is_authoritative(self, monkeypatch, tmp_path):
        """A profile that lists ``search_agent_ids`` says exactly what it may see; the shared
        layer's name is not silently added to it."""
        provider, _ = _provider(
            monkeypatch, tmp_path, env={"MEM0_AGENT_ID": "team-two"},
            file_cfg={"search_agent_ids": ["team-two", "audit"]})

        assert provider._search_agent_ids == {"team-two", "audit"}

    def test_an_empty_allow_list_fails_the_profile_closed(self, monkeypatch, tmp_path):
        """``[]`` used to mean "see every agent under this user_id". It must not: an empty or
        malformed scope is a configuration mistake, and the safe reading of a mistake is no
        recall at all, never everyone's."""
        with pytest.raises(ValueError):
            _provider(monkeypatch, tmp_path, env={"MEM0_AGENT_ID": "team-two"},
                      file_cfg={"search_agent_ids": []})

    @pytest.mark.parametrize("name", [" ", " padded", "*"], ids=["blank", "padded", "wildcard"])
    def test_an_unusable_shared_layer_name_fails_the_profile_closed(self, monkeypatch, tmp_path, name):
        """The configured name reaches the store as an exact id, so it gets the same validation
        as any other entry — a typo must not silently become a scope nobody meant."""
        with pytest.raises(ValueError):
            _provider(monkeypatch, tmp_path,
                      env={"MEM0_AGENT_ID": "team-two", "MEM0_SHARED_AGENT_ID": name})


class TestSharedLayerRecall:

    @pytest.mark.parametrize(
        "shared_name, env",
        [("shared", {}), ("house-layer", {"MEM0_SHARED_AGENT_ID": "house-layer"})],
        ids=["default-name", "configured-name"],
    )
    def test_recall_covers_own_agent_and_the_shared_layer_only(
            self, monkeypatch, tmp_path, shared_name, env):
        backend = _FakeBackend([
            _row("1", "mine", "team-two", 0.9),
            _row("2", "shared fact", shared_name, 0.5),
            _row("3", "another customer's", "team-three", 0.8),
            _row("4", "another tenant's", shared_name, 0.7, user_id="tenant-two"),
        ])
        provider, fake = _provider(
            monkeypatch, tmp_path, env={"MEM0_AGENT_ID": "team-two", **env}, backend=backend)

        found = provider._search("anything", top_k=10)

        assert [r["memory"] for r in found] == ["mine", "shared fact"]
        # One query per allowed agent, each scoped to this tenant at the backend.
        assert {frozenset(call["filters"].items()) for call in fake.searches} == {
            frozenset({"user_id": _TENANT, "agent_id": "team-two"}.items()),
            frozenset({"user_id": _TENANT, "agent_id": shared_name}.items()),
        }

    def test_a_name_that_is_not_configured_is_not_recalled(self, monkeypatch, tmp_path):
        """Renaming the layer must actually narrow recall — otherwise the setting is decoration."""
        backend = _FakeBackend([_row("1", "old layer", "some-other-layer", 0.9)])
        provider, _ = _provider(
            monkeypatch, tmp_path,
            env={"MEM0_AGENT_ID": "team-two", "MEM0_SHARED_AGENT_ID": "house-layer"},
            backend=backend)

        assert provider._search("anything", top_k=10) == []

    def test_a_write_never_lands_in_the_shared_layer(self, monkeypatch, tmp_path):
        """The invariant that makes the layer safe to share: a turn writes under the profile's
        own agent_id, so nothing a profile says leaks into everyone else's recall."""
        provider, fake = _provider(monkeypatch, tmp_path, env={"MEM0_AGENT_ID": "team-two"})

        provider._add([{"role": "user", "content": "remember this"}], infer=False)

        assert fake.added == [{"user_id": _TENANT, "agent_id": "team-two"}]
