"""Tenant read/write contract through real provider config and multiplex scopes."""
import json
from contextlib import contextmanager

import pytest

from agent import secret_scope
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from plugins.memory.mem0 import Mem0MemoryProvider
from plugins.memory.mem0._backend import Mem0Backend


@contextmanager
def scope(home):
    home_token = set_hermes_home_override(home)
    secret_token = secret_scope.set_secret_scope({})
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


class Store(Mem0Backend):
    def __init__(self):
        self.rows = {}
        self.calls = []

    def search(self, query, *, filters, top_k=10, rerank=False):
        self.calls.append(filters)
        matches = [r.copy() for r in self.rows.values()
                   if all(r.get(k) == v for k, v in filters.items())]
        # A malformed/misbehaving backend must not escape response validation.
        return matches[:top_k] + [{"id": "injected", "memory": "foreign",
                                  "user_id": "other", "agent_id": filters.get("agent_id")}]

    def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
        ident = str(len(self.rows))
        self.rows[ident] = dict(id=ident, user_id=user_id, agent_id=agent_id,
                               memory=messages[0]["content"], score=1)
        return {"results": [self.rows[ident]]}

    def get(self, memory_id):
        return self.rows.get(memory_id)

    def _update(self, memory_id, text):
        self.rows[memory_id]["memory"] = text

    def _delete(self, memory_id):
        del self.rows[memory_id]


def test_profile_reads_writes_prefetch_and_sync_stay_scoped_a_b_a(tmp_path, monkeypatch):
    backend = Store()
    monkeypatch.setattr(Mem0MemoryProvider, "_create_backend", lambda self: backend)
    for agent in ["a", "b", "shared", "foreign"]:
        backend.add([{"content": f"{agent} knowledge"}], user_id="human", agent_id=agent)
    backend.add([{"content": "other user"}], user_id="other", agent_id="a")
    homes = {}
    for agent in ["a", "b"]:
        home = tmp_path / agent
        home.mkdir()
        (home / "mem0.json").write_text(json.dumps(dict(mode="oss", user_id="human",
            agent_id=agent, search_agent_ids=[agent, "shared"])))
        homes[agent] = home
    secret_scope.set_multiplex_active(True)
    try:
        for agent in ["a", "b", "a"]:
            with scope(homes[agent]):
                p = Mem0MemoryProvider()
                p.initialize("test")
                found = p._search("knowledge", top_k=20)
                assert {r["agent_id"] for r in found} == {agent, "shared"}
                assert all(r["user_id"] == "human" for r in found)
                body = p.prefetch("knowledge")
                assert f"{agent} knowledge" in body and "shared knowledge" in body
                assert "foreign" not in body and "other user" not in body
                # Even known UUIDs do not confer write access to another tenant.
                for row in list(backend.rows.values()):
                    if row["agent_id"] != agent or row["user_id"] != "human":
                        before = row.copy()
                        for name in ["mem0_update", "mem0_delete"]:
                            result = p.handle_tool_call(name, {"memory_id": row["id"], "text": "overwrite"})
                            assert "error" in json.loads(result)
                            assert backend.rows[row["id"]] == before
                add = p.handle_tool_call("mem0_add", {"content": "new private fact"})
                assert "error" not in json.loads(add)
                own = next(r for r in backend.rows.values() if r["memory"] == "new private fact")
                assert (own["user_id"], own["agent_id"]) == ("human", agent)
                assert "error" not in json.loads(p.handle_tool_call("mem0_update", {"memory_id": own["id"], "text": "updated"}))
                assert "error" not in json.loads(p.handle_tool_call("mem0_delete", {"memory_id": own["id"]}))
                p.sync_turn("automatic private fact", "ack")
                p._sync_thread.join(timeout=5)
                assert not p._sync_thread.is_alive()
                newest = list(backend.rows.values())[-1]
                assert (newest["user_id"], newest["agent_id"]) == ("human", agent)
                p.shutdown()
        assert all(set(call) == {"user_id", "agent_id"} for call in backend.calls)
    finally:
        secret_scope.set_multiplex_active(False)


def test_bad_scope_fails_closed_and_missing_scope_is_own_plus_shared_layer(tmp_path, monkeypatch):
    backend = Store()
    monkeypatch.setattr(Mem0MemoryProvider, "_create_backend", lambda self: backend)
    config = dict(mode="oss", user_id="human", agent_id="a")
    with scope(tmp_path):
        for invalid in [None, [], "a", [""], [" "], ["*"], [1], {}, ["foreign"]]:
            (tmp_path / "mem0.json").write_text(json.dumps({**config, "search_agent_ids": invalid}))
            p = Mem0MemoryProvider()
            with pytest.raises(ValueError):
                p.initialize("test")
            assert p._search("anything") == []
            assert "error" in json.loads(p.handle_tool_call("mem0_add", {"content": "never written"}))
        (tmp_path / "mem0.json").write_text(json.dumps(config))
        p = Mem0MemoryProvider()
        p.initialize("test")
        backend.add([{"content": "shared"}], user_id="human", agent_id="shared")
        backend.add([{"content": "own"}], user_id="human", agent_id="a")
        backend.add([{"content": "another tenant"}], user_id="human", agent_id="foreign")
        # Widening the default to the shared layer must not widen it to anything else.
        assert {r["memory"] for r in p._search("anything")} == {"own", "shared"}
        p.shutdown()
