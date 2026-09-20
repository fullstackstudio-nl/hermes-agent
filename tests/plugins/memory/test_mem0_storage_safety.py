"""Profile-local history and non-destructive embedding mismatch contracts."""
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from plugins.memory.mem0._backend import OSSBackend


def test_backend_passes_distinct_durable_history_paths_a_b_a(tmp_path, monkeypatch):
    paths = []

    class Memory:
        @classmethod
        def from_config(cls, config):
            path = config["history_db_path"]
            paths.append(path)
            with sqlite3.connect(path) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS events (value TEXT)")
                conn.execute("INSERT INTO events VALUES (?)", (path,))
            return SimpleNamespace()

    monkeypatch.setitem(sys.modules, "mem0", SimpleNamespace(Memory=Memory))
    cfg = dict(vector_store={"provider": "qdrant", "config": {}},
               llm={"provider": "ollama"}, embedder={"provider": "ollama"})
    for name in ["a", "b", "a"]:
        token = set_hermes_home_override(tmp_path / name)
        try:
            OSSBackend(cfg)
        finally:
            reset_hermes_home_override(token)
    assert paths[0] == paths[2] and paths[0] != paths[1]
    for path, count in [(paths[0], 2), (paths[1], 1)]:
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == count
        assert (tmp_path / ("a" if count == 2 else "b") / "mem0").stat().st_mode & 0o777 == 0o700


def test_dimension_mismatch_never_deletes_existing_qdrant_collection(monkeypatch):
    collections = {"existing": SimpleNamespace(size=768)}
    deletions = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def collection_exists(self, name):
            return name in collections

        def get_collection(self, name):
            return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=collections[name])))

        def delete_collection(self, name):
            deletions.append(name)
            del collections[name]

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=Client))
    cfg = {"url": "http://127.0.0.1:6333", "collection_name": "existing"}
    with pytest.raises(ValueError, match="refusing destructive"):
        OSSBackend._recreate_collection_if_dims_changed("qdrant", cfg, 1536)
    assert not deletions and collections["existing"].size == 768
    OSSBackend._recreate_collection_if_dims_changed("qdrant", cfg, 768)
    assert not deletions
