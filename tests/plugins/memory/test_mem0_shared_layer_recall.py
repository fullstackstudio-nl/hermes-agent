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


class TestWhatTheModelIsTold:
    """The prompt and the tool descriptions must match the scope the profile actually has.

    A model told only that it "has memory" gets it wrong in both directions: it reports
    something it read from shared memory as a fact the user told it, and it offers to save
    something for the whole team although every write attaches its own agent_id. The wording is
    derived from the resolved scope, so a profile without shared memory is never promised any.
    """

    def test_a_read_only_layer_is_described_as_readable_but_not_writable(
            self, monkeypatch, tmp_path):
        """A layer configured read-only (shared writes off) is still recalled from, and the wording
        has to stop the model offering to put something there."""
        provider, _ = _provider(
            monkeypatch, tmp_path,
            env={"MEM0_AGENT_ID": "team-two", "MEM0_SHARED_WRITES": "false"})

        note = provider.scope_note()

        assert "read but not write" in note
        assert "Everything you store goes into your own memories" in note
        assert "Do not offer to save anything into shared memory" in note
        assert note in provider.system_prompt_block()

    def test_a_profile_without_shared_memory_is_not_promised_any(self, monkeypatch, tmp_path):
        """An explicit scope of just its own agent has no shared memory, so the wording must not
        mention any. This is the case the text is derived for."""
        provider, _ = _provider(
            monkeypatch, tmp_path, env={"MEM0_AGENT_ID": "team-two"},
            file_cfg={"search_agent_ids": ["team-two"]})

        note = provider.scope_note()

        assert "shared" not in note.lower()
        assert "your own memories only" in note
        assert note in provider.system_prompt_block()

    def test_the_shared_identity_is_told_that_others_read_what_it_stores(
            self, monkeypatch, tmp_path):
        """A profile whose own agent_id IS the shared name writes where everyone else reads. It
        has no second scope to warn about, but it does need to know that."""
        provider, _ = _provider(
            monkeypatch, tmp_path,
            env={"MEM0_AGENT_ID": "house-layer", "MEM0_SHARED_AGENT_ID": "house-layer"})

        note = provider.scope_note()

        assert provider._search_agent_ids == {"house-layer"}
        assert "Other profiles recall from that same memory" in note

    def test_no_scope_name_is_ever_put_in_the_prompt(self, monkeypatch, tmp_path):
        """The scope names belong to whoever deployed the gateway. They must not travel into the
        prompt or a tool description just because they are configured."""
        provider, _ = _provider(
            monkeypatch, tmp_path,
            env={"MEM0_AGENT_ID": "team-two", "MEM0_SHARED_AGENT_ID": "a-named-layer"})

        text = provider.system_prompt_block() + "".join(
            s["description"] for s in provider.get_tool_schemas())

        assert "a-named-layer" not in text
        assert "team-two" not in text

    def test_the_tools_say_where_a_write_lands_when_shared_memory_is_read_only(
            self, monkeypatch, tmp_path):
        provider, _ = _provider(
            monkeypatch, tmp_path,
            env={"MEM0_AGENT_ID": "team-two", "MEM0_SHARED_WRITES": "false"})

        described = {s["name"]: s["description"] for s in provider.get_tool_schemas()}

        assert "never into shared memory" in described["mem0_add"]
        assert "read but not write" in described["mem0_search"]
        assert "refused" in described["mem0_update"] and "refused" in described["mem0_delete"]

    def test_the_tools_stay_silent_about_sharing_when_there_is_none(self, monkeypatch, tmp_path):
        provider, _ = _provider(
            monkeypatch, tmp_path, env={"MEM0_AGENT_ID": "team-two"},
            file_cfg={"search_agent_ids": ["team-two"]})

        described = {s["name"]: s["description"] for s in provider.get_tool_schemas()}

        assert not any("shared memory" in d for d in described.values())
        assert sorted(described) == ["mem0_add", "mem0_delete", "mem0_search", "mem0_update"]

    def test_an_uninitialized_provider_promises_nothing(self):
        """It can read nothing at all, so shared memory must not be advertised."""
        provider = mem0.Mem0MemoryProvider()
        provider._user_id = "someone"

        assert "shared" not in provider.scope_note().lower()
        assert not any("shared memory" in s["description"] for s in provider.get_tool_schemas())

    def test_what_a_read_only_layers_tools_claim_is_what_the_code_does(self, monkeypatch, tmp_path):
        """With writes off the descriptions are only honest while a write keeps landing in the
        profile's own scope and a shared row stays unwritable. Tie the wording to both."""
        backend = _FakeBackend([_row("1", "shared fact", "shared", 0.9)])
        backend.rows = {"1": dict(backend.results[0])}
        backend.get = lambda memory_id: backend.rows.get(memory_id)
        provider, fake = _provider(
            monkeypatch, tmp_path, backend=backend,
            env={"MEM0_AGENT_ID": "team-two", "MEM0_SHARED_WRITES": "false"})

        provider._add([{"role": "user", "content": "a fact"}], infer=False)
        assert fake.added == [{"user_id": _TENANT, "agent_id": "team-two"}]

        refused = provider._tool_mutate({"memory_id": "1", "text": "overwrite"})
        assert "error" in json.loads(refused)


class _Store:
    """A backend that keeps its rows, so a write can be read back, mutated and counted."""

    def __init__(self):
        self.rows: dict = {}
        self._next = 0

    def search(self, query, filters=None, top_k=10, rerank=False):
        rows = [dict(r) for r in self.rows.values()]
        for key, value in (filters or {}).items():
            rows = [r for r in rows if r.get(key) == value]
        return rows[:top_k]

    def add(self, messages, user_id=None, agent_id=None, infer=True, metadata=None):
        self._next += 1
        ident = str(self._next)
        self.rows[ident] = {"id": ident, "user_id": user_id, "agent_id": agent_id,
                            "memory": messages[-1]["content"], "score": 1.0,
                            "metadata": dict(metadata or {})}
        return {"results": [dict(self.rows[ident])]}

    def get(self, memory_id):
        row = self.rows.get(memory_id)
        return dict(row) if row else None

    def update(self, memory_id, text):
        self.rows[memory_id]["memory"] = text
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id):
        del self.rows[memory_id]
        return {"result": "Memory deleted.", "memory_id": memory_id}


def _stored(monkeypatch, tmp_path, *, env=None, file_cfg=None):
    """A provider over a real-ish store, with its own agent id set."""
    return _provider(monkeypatch, tmp_path, backend=_Store(),
                     env={"MEM0_AGENT_ID": "team-two", **(env or {})}, file_cfg=file_cfg)


class TestSharedWrites:
    """Storing into shared memory is deliberate, traceable, and only where a layer is configured.

    The whole point of the ownership check the isolation added is that reading a shared fact does not
    license writing over it. Making the layer writable must not weaken that for anything else, so
    these tests pin both the new capability and the boundary it must not cross.
    """

    def test_an_ordinary_save_still_lands_in_the_profiles_own_memory(self, monkeypatch, tmp_path):
        provider, store = _stored(monkeypatch, tmp_path)

        out = json.loads(provider.handle_tool_call("mem0_add", {"content": "a private preference"}))

        assert "error" not in out
        row = next(iter(store.rows.values()))
        assert row["agent_id"] == "team-two"
        assert "written_by" not in row["metadata"]
        assert "shared" not in out["result"].lower()

    def test_the_automatic_turn_sync_never_shares(self, monkeypatch, tmp_path):
        """Sharing has to be chosen for one fact. A conversation must not fill the layer by itself."""
        provider, store = _stored(monkeypatch, tmp_path)

        provider.sync_turn("something the user said", "something answered")
        provider._sync_thread.join(timeout=5)
        assert not provider._sync_thread.is_alive()

        assert [r["agent_id"] for r in store.rows.values()] == ["team-two"]
        assert all("written_by" not in r["metadata"] for r in store.rows.values())

    def test_a_deliberate_shared_save_lands_in_shared_with_its_provenance(self, monkeypatch, tmp_path):
        provider, store = _stored(monkeypatch, tmp_path)

        out = json.loads(provider.handle_tool_call(
            "mem0_add", {"content": "the office is closed on Friday", "shared": True}))

        assert "error" not in out
        row = next(iter(store.rows.values()))
        assert row["agent_id"] == "shared"                 # the configured layer, not its own
        assert row["metadata"]["written_by"] == "team-two"  # who put it there
        assert "shared memory" in out["result"]             # and the turn is told where it went

    def test_a_string_shared_argument_counts(self, monkeypatch, tmp_path):
        """Models pass booleans as strings often enough that silently storing to the wrong scope
        would be the most likely way this goes wrong in production."""
        provider, store = _stored(monkeypatch, tmp_path)

        provider.handle_tool_call("mem0_add", {"content": "a house fact", "shared": "true"})

        assert next(iter(store.rows.values()))["agent_id"] == "shared"

    def test_an_edit_and_a_delete_of_a_shared_entry_both_work(self, monkeypatch, tmp_path):
        provider, store = _stored(monkeypatch, tmp_path)
        provider.handle_tool_call("mem0_add", {"content": "closed on Friday", "shared": True})
        ident = next(iter(store.rows))

        edited = json.loads(provider.handle_tool_call(
            "mem0_update", {"memory_id": ident, "text": "closed on Thursday"}))
        assert "error" not in edited
        assert store.rows[ident]["memory"] == "closed on Thursday"

        removed = json.loads(provider.handle_tool_call("mem0_delete", {"memory_id": ident}))
        assert "error" not in removed
        assert ident not in store.rows

    def test_another_profiles_own_memory_is_still_untouchable(self, monkeypatch, tmp_path):
        """The boundary the isolation drew. Only own memory and the shared layer are writable."""
        provider, store = _stored(monkeypatch, tmp_path)
        store.add([{"content": "another profile's private note"}], user_id=_TENANT, agent_id="team-three")
        ident = next(iter(store.rows))

        for name, args in (("mem0_update", {"memory_id": ident, "text": "overwrite"}),
                           ("mem0_delete", {"memory_id": ident})):
            assert "error" in json.loads(provider.handle_tool_call(name, args))
        assert store.rows[ident]["memory"] == "another profile's private note"

    def test_re_saving_the_same_shared_fact_does_not_duplicate_it(self, monkeypatch, tmp_path):
        """The loop to avoid: recall a shared fact, then store it back, forever. Only an exact
        repeat is caught; that is the shape the loop takes."""
        provider, store = _stored(monkeypatch, tmp_path)
        provider.handle_tool_call("mem0_add", {"content": "closed on Friday", "shared": True})

        again = json.loads(provider.handle_tool_call(
            "mem0_add", {"content": "  Closed   on Friday  ", "shared": True}))

        assert "error" not in again
        assert "Already in shared memory" in again["result"]
        assert len(store.rows) == 1

    def test_the_same_text_in_own_memory_is_not_treated_as_a_duplicate(self, monkeypatch, tmp_path):
        """The check is scoped to the shared layer, so a private note does not block sharing it."""
        provider, store = _stored(monkeypatch, tmp_path)
        provider.handle_tool_call("mem0_add", {"content": "closed on Friday"})

        provider.handle_tool_call("mem0_add", {"content": "closed on Friday", "shared": True})

        assert sorted(r["agent_id"] for r in store.rows.values()) == ["shared", "team-two"]


class TestSharedWritesAbsent:
    """No shared layer, or a layer configured read-only: the capability and its wording disappear."""

    def test_no_shared_layer_means_no_parameter_and_no_wording(self, monkeypatch, tmp_path):
        provider, _ = _stored(monkeypatch, tmp_path, file_cfg={"search_agent_ids": ["team-two"]})

        assert provider._shared_write_target is None
        described = {s["name"]: s for s in provider.get_tool_schemas()}
        assert "shared" not in described["mem0_add"]["parameters"]["properties"]
        assert not any("shared" in s["description"].lower() for s in described.values())
        assert "shared" not in provider.scope_note().lower()

    def test_a_read_only_layer_offers_no_parameter_and_says_so(self, monkeypatch, tmp_path):
        provider, _ = _stored(monkeypatch, tmp_path, env={"MEM0_SHARED_WRITES": "false"})

        assert provider._shared_write_target is None
        assert provider._search_agent_ids == {"team-two", "shared"}  # still readable
        described = {s["name"]: s for s in provider.get_tool_schemas()}
        assert "shared" not in described["mem0_add"]["parameters"]["properties"]
        assert "read but not write" in provider.scope_note()

    def test_asking_to_share_without_a_layer_is_refused_not_misfiled(self, monkeypatch, tmp_path):
        """Defence in depth: a model can invent an argument it was never offered, and that must not
        fall back to writing the fact somewhere else."""
        for cfg in ({"search_agent_ids": ["team-two"]}, None):
            env = {"MEM0_SHARED_WRITES": "false"} if cfg is None else {}
            provider, store = _stored(monkeypatch, tmp_path, env=env, file_cfg=cfg)

            out = json.loads(provider.handle_tool_call(
                "mem0_add", {"content": "meant for everyone", "shared": True}))

            assert "error" in out
            assert store.rows == {}

    def test_the_shared_identity_gets_no_parameter_because_every_save_is_shared(
            self, monkeypatch, tmp_path):
        provider, _ = _stored(
            monkeypatch, tmp_path,
            env={"MEM0_AGENT_ID": "house-layer", "MEM0_SHARED_AGENT_ID": "house-layer"})

        assert provider._shared_write_target is None
        described = {s["name"]: s for s in provider.get_tool_schemas()}
        assert "shared" not in described["mem0_add"]["parameters"]["properties"]
        assert "Other profiles recall from that same memory" in provider.scope_note()


class TestSharedWriteWording:
    """The prompt must describe the capability the profile actually has, in each configuration."""

    def test_a_writable_layer_is_described_as_deliberate_and_visible_to_everyone(
            self, monkeypatch, tmp_path):
        provider, _ = _stored(monkeypatch, tmp_path)

        note = provider.scope_note()

        assert "unless you pass shared on mem0_add" in note
        assert "every profile sharing this memory can read it" in note
        assert "never put anything private to this user there" in note
        assert "correct or remove entries in shared memory" in note
        assert note in provider.system_prompt_block()

    def test_no_scope_name_reaches_the_prompt_or_the_parameter(self, monkeypatch, tmp_path):
        provider, _ = _stored(
            monkeypatch, tmp_path, env={"MEM0_SHARED_AGENT_ID": "a-named-layer"})

        text = provider.system_prompt_block() + json.dumps(provider.get_tool_schemas())

        assert "a-named-layer" not in text
        assert "team-two" not in text

    def test_what_the_tools_promise_is_what_the_code_does(self, monkeypatch, tmp_path):
        """Tie every claim in the writable wording to observed behaviour, so widening or narrowing
        the write path fails a description instead of leaving it a polite lie."""
        provider, store = _stored(monkeypatch, tmp_path)
        described = {s["name"]: s["description"] for s in provider.get_tool_schemas()}
        assert "unless you pass shared" in described["mem0_add"]
        assert "entries in shared memory" in described["mem0_update"]

        # "goes into your own memories unless you pass shared"
        provider.handle_tool_call("mem0_add", {"content": "mine"})
        provider.handle_tool_call("mem0_add", {"content": "ours", "shared": True})
        by_scope = {r["memory"]: r["agent_id"] for r in store.rows.values()}
        assert by_scope == {"mine": "team-two", "ours": "shared"}

        # "you may also correct or remove entries in shared memory"
        shared_id = next(i for i, r in store.rows.items() if r["agent_id"] == "shared")
        assert "error" not in json.loads(provider.handle_tool_call(
            "mem0_update", {"memory_id": shared_id, "text": "ours, corrected"}))

        # "anything else is refused"
        store.add([{"content": "theirs"}], user_id=_TENANT, agent_id="team-three")
        foreign = next(i for i, r in store.rows.items() if r["agent_id"] == "team-three")
        assert "error" in json.loads(provider.handle_tool_call("mem0_delete", {"memory_id": foreign}))
