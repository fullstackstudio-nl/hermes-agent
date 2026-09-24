"""Mem0 memory plugin — MemoryProvider interface.

Server-side fact extraction and semantic search via the Mem0 Platform API (cloud), a
self-hosted Mem0 server (MEM0_HOST, HTTP), or OSS Memory. Secrets live in $HERMES_HOME/.env
(MEM0_API_KEY, MEM0_HOST); settings in $HERMES_HOME/mem0.json via `hermes memory setup`:
mode ("platform"|"oss"), host, user_id (canonical id across gateways; unset → gateway-native
id), agent_id (a named profile's own, written when the profile is made; see _identity). MEM0_* env vars
remain a fallback.
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider, spawn_context_thread
from agent.secret_scope import get_secret
from . import _identity
from tools.registry import tool_error
from utils import atomic_json_write, read_json_or_empty

logger = logging.getLogger(__name__)

# Circuit breaker: after _BREAKER_THRESHOLD consecutive failures, pause API
# calls for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD, _BREAKER_COOLDOWN_SECS, _PREFETCH_WAIT_SECS = 5, 120, 3
_CLIENT_ERROR_TYPES = ("MemoryNotFoundError", "ValidationError")
# Placeholder user_id. initialize() treats it as "no operator-configured user_id"
# so legacy mem0.json files written by the wizard don't override gateway-native ids.
_DEFAULT_USER_ID = "hermes-user"

# Name of the memory layer every profile may recall from besides its own. A turn reaches it in two
# ways and no more: recall reads it, and ``mem0_add`` writes to it when the call explicitly asks to
# share (see ``_shared_write_target``). An automatic turn sync never does, so nothing lands there as
# a side effect of a conversation. Deployments name their own layer with
# MEM0_SHARED_AGENT_ID (per-profile .env) or "shared_agent_id" in mem0.json — a shared house layer,
# one layer per customer, one per department. A profile that lists "search_agent_ids" explicitly
# says exactly what it may see and this name is not added to it.
_DEFAULT_SHARED_AGENT_ID = "shared"

# sync_turn sends the whole turn to the backend for fact extraction. OSS embedding
# models often have small context windows (bge-small-zh-v1.5: 512 tokens ≈ 500 chars;
# jina-embeddings-v3: 8192), and oversized turns make backend.add() raise — Ollama
# answers HTTP 500, hosted APIs return INPUT_TOKEN_LIMIT_EXCEEDED — which _try only
# logs, silently dropping the turn's memory extraction. Cap each message up front.
# The default fits a 512-token embedder (measured: 450 OK, 600 -> HTTP 500 on
# bge-small-zh-v1.5:f16); ``sync_max_chars`` in mem0.json raises it for larger windows.
_SYNC_MSG_MAX_CHARS = 450


# Sentence ends recognized when trimming a synced message. Deliberately unordered:
# the LAST boundary of ANY kind wins, so one CJK stop early in a mixed-script turn
# cannot outrank a Latin stop near the end of the window. ``".\n"`` is not listed —
# its index can never exceed the bare ``"."`` it starts with.
_SYNC_SENTENCE_ENDS = ("。", "！", "？", ".", "!", "?")


def _truncate_for_sync(text: str, max_len: int = _SYNC_MSG_MAX_CHARS) -> str:
    """Cap a synced message at its last sentence boundary within ``max_len``.

    Short messages pass through unchanged; long ones keep the last complete
    sentence inside the window so fact extraction still sees coherent statements,
    with a hard cut as fallback when no boundary exists (or one only appears in
    the first third of the window, which usually means unsegmented input).
    """
    if len(text) <= max_len:
        return text
    window = text[:max_len]
    cut = max(window.rfind(sep) for sep in _SYNC_SENTENCE_ENDS)
    if cut > max_len // 3:
        return text[:cut + 1]
    return text[:max_len]


def _is_client_error(exc: Exception) -> bool:
    """True for user-caused errors (bad ID, not found) that should NOT trip circuit breaker."""
    err_str = str(exc).lower()
    return type(exc).__name__ in _CLIENT_ERROR_TYPES or any(s in err_str for s in ("404", "not found", "valid uuid"))


def _load_config() -> dict:
    """Env vars provide defaults; $HERMES_HOME/mem0.json overrides individual keys.
    Layering avoids a silent failure when the JSON file exists but lacks fields
    like ``api_key`` that the user set in ``.env``."""
    from hermes_constants import get_hermes_home
    # Identity (user/agent id), host and mode are .env values like the key: read them through the
    # profile scope too, or a secondary profile's memories land in the default profile's account.
    # A scope-less multiplex caller raises here on purpose — that is a spawn-site bug, and
    # swallowing it would silently route the turn's memories to the default profile.
    config = {"mode": get_secret("MEM0_MODE", "") or "platform", "host": get_secret("MEM0_HOST", "") or "",
              "agent_id": get_secret("MEM0_AGENT_ID", "") or _identity.LEGACY_AGENT_ID,
              # Read through the profile scope like agent_id: under multiplexing the name of a
              # profile's shared layer lives in that profile's own .env, not the process env.
              "shared_agent_id": get_secret("MEM0_SHARED_AGENT_ID", "") or _DEFAULT_SHARED_AGENT_ID,
              # Whether a turn may deliberately store into the shared layer. Configuring a layer is
              # already the opt-in, so this defaults on; MEM0_SHARED_WRITES=false keeps a curated
              # layer that profiles may read and only an operator fills.
              "shared_writes": get_secret("MEM0_SHARED_WRITES", ""),
              "oss": {}}
    if user_id := get_secret("MEM0_USER_ID", ""):  # only when explicitly configured, so initialize() can fall back to the gateway-native id
        config["user_id"] = user_id
    file_cfg = read_json_or_empty(get_hermes_home() / "mem0.json")
    config.update({k: v for k, v in file_cfg.items() if k == "search_agent_ids" or (v is not None and v != "")})
    # MEM0_API_KEY authenticates the Platform and self-hosted HTTP backends; pure OSS mode builds its
    # backend from the local ``oss`` config and has no platform credential to resolve, so a profile
    # scope WITHOUT the key must still load an OSS config (#99121 as it stands today: the caller is
    # scoped, the scope is just empty). Decided after mem0.json overrode the env defaults because
    # the file may be what selects ``oss``. Scope-less callers already raised above.
    if config.get("mode", "platform") == "oss":
        config.setdefault("api_key", "")
    elif not config.get("api_key"):
        config["api_key"] = get_secret("MEM0_API_KEY", "")
    return config


def _schema(name: str, description: str, properties: dict[str, tuple[str, str]], required: list[str]) -> dict:
    props = {k: {"type": t, "description": d} for k, (t, d) in properties.items()}
    return {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required}}


TOOL_SCHEMAS = [
    _schema("mem0_search", "Search the user's memories by meaning; returns facts ranked by relevance. Use this before answering any question that may depend on what you know about the user (preferences, facts, history, people, projects, past decisions). For multi-part or multi-hop questions, call it several times — vary the wording and run follow-up searches on what earlier results reveal; one search is rarely enough.",
            {"query": ("string", "What to search for."), "top_k": ("integer", "Max results (default: 10, max: 50)."), "rerank": ("boolean", "Rerank results for relevance (default: false, platform mode only).")}, ["query"]),
    _schema("mem0_add", "Store a durable fact about the user, verbatim (no LLM extraction). Call this the moment the user states a lasting preference, correction, decision, or personal detail worth recalling on future turns — don't wait to be asked to remember. Skip transient chit-chat and facts you've already stored.",
            {"content": ("string", "The fact to store.")}, ["content"]),
    _schema("mem0_update", "Replace the text of an existing memory by its ID (take the ID from a mem0_search result). Use when a stored fact has changed or was wrong — correct it in place instead of adding a duplicate.",
            {"memory_id": ("string", "Memory UUID to update."), "text": ("string", "New text content.")}, ["memory_id", "text"]),
    _schema("mem0_delete", "Delete a memory by its ID (take the ID from a mem0_search result). Use when a stored fact is obsolete or the user asks you to forget it; prefer mem0_update if the fact merely changed.",
            {"memory_id": ("string", "Memory UUID to delete.")}, ["memory_id"]),
]

_PROMPT_BODY = (
    "You have persistent memory of this user from past conversations. You should call mem0_search before answering anything that could depend on prior context (the user's preferences, facts, history, people, projects, or earlier decisions) — do not rely on the chat window alone, and do not assume you have no memory.\n"
    "For multi-part or multi-hop questions, run several searches with different wording/angles and follow-up searches on what the first results surface; one search is rarely enough. Keep searching until you have every fact the question needs before you answer.\n"
    "Tools: mem0_search to find memories, mem0_add to store facts, mem0_update and mem0_delete to manage by ID."
)

# Recall can cover more than one scope while a write only ever lands in the profile's own
# (``_add`` attaches ``self._agent_id``, and ``_tool_mutate`` refuses a row owned by anyone
# else). A model told only that it "has memory" states that wrongly in both directions: it
# says "I remember" about a fact it read from another scope, and it offers to save something
# "for the team" although it cannot. These notes say which is which, and they are added only
# when a shared scope is actually configured -- see ``_shared_scopes``. No scope is ever named
# here: a name belongs to whoever deployed the gateway, and this text goes into the prompt.
_SHARED_SCOPE_TOOL_NOTES = {
    "mem0_search": " Results can come from your own memories or from shared memory you can read but not write.",
    "mem0_add": " The fact goes into your own memories, never into shared memory.",
    "mem0_update": " Only your own memories can be edited; one recalled from shared memory is refused.",
    "mem0_delete": " Only your own memories can be deleted; one recalled from shared memory is refused.",
}


# Same, for a profile that may also store into the shared layer. Sharing is a parameter on the add
# tool rather than a tool of its own, so the model cannot reach it without choosing it for one fact.
_SHARED_WRITE_TOOL_NOTES = {
    "mem0_search": " Results can come from your own memories or from shared memory.",
    "mem0_add": (" It goes into your own memories unless you pass shared, which stores that one fact"
                 " where every profile sharing this memory can read it."),
    "mem0_update": " You may edit your own memories and entries in shared memory; anything else is refused.",
    "mem0_delete": " You may delete your own memories and entries in shared memory; anything else is refused.",
}
_SHARED_ADD_PARAM = {
    "shared": {
        "type": "boolean",
        "description": ("Store this in shared memory instead of your own, where every profile that"
                        " shares it can read it. Default false. Only for something meant for"
                        " everyone, never for anything private to this user."),
    },
}


def _as_bool(value) -> bool:
    """Tool arguments arrive as a bool or as the string a model typed."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _shared_scopes(agent_id: str, search_agent_ids) -> list[str]:
    """The scopes recall may read besides the profile's own. Empty means own memories only.

    An uninitialized provider has an empty scope and so reads nothing, which lands here as
    own-only -- the safe answer, because promising shared memory it cannot reach would be a lie.
    """
    return sorted(set(search_agent_ids) - {agent_id})


class Mem0MemoryProvider(MemoryProvider):
    """Mem0 memory with server-side extraction and semantic search (platform, self-hosted or OSS)."""

    def __init__(self):
        self._config = self._backend = self._sync_thread = self._prefetch_thread = None
        self._mode, self._api_key, self._host, self._user_id, self._agent_id = "platform", "", "", _DEFAULT_USER_ID, "hermes"
        self._rerank_default, self._channel = False, "cli"  # channel = gateway name (cli/telegram/discord/...)
        self._search_agent_ids = frozenset()  # uninitialized providers cannot read memory
        self._shared_agent_id = _DEFAULT_SHARED_AGENT_ID
        self._shared_write_target = None  # nor write it
        self._sync_max_chars = _SYNC_MSG_MAX_CHARS
        self._prefetch_query = self._prefetch_result = ""
        self._prefetch_done = self._atexit_registered = False
        self._consecutive_failures, self._breaker_open_until = 0, 0.0  # circuit breaker state
        self._breaker_lock, self._sync_lock, self._prefetch_lock = threading.Lock(), threading.Lock(), threading.Lock()

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        if cfg.get("mode", "platform") == "oss":
            return bool(cfg.get("oss", {}).get("vector_store"))
        return bool(cfg.get("api_key") or cfg.get("host"))  # platform needs a key; self-hosted a host (key optional with AUTH_DISABLED)

    def save_config(self, values, hermes_home):
        """Merge-write config to $HERMES_HOME/mem0.json."""
        config_path = Path(hermes_home) / "mem0.json"
        before = read_json_or_empty(config_path)
        merged = _identity.settle_saved_identity(hermes_home, before, {**before, **values})
        atomic_json_write(config_path, merged, mode=0o600)

    def get_config_schema(self):
        from hermes_constants import get_hermes_home
        api_key_required = _load_config().get("mode", "platform") != "oss"
        return [
            {"key": "api_key", "description": "Mem0 Platform API key", "secret": True, "required": api_key_required, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "host", "description": "Self-hosted Mem0 server URL (leave blank for cloud)", "required": False, "env_var": "MEM0_HOST"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            # The dashboard form's default: never the default profile's identity for a named profile.
            {"key": "agent_id", "description": "Agent identifier", "default": _identity.setup_default_agent_id(get_hermes_home())},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "false", "choices": ["true", "false"]},
        ]

    def post_setup(self, hermes_home: str, config: dict) -> None:
        from ._setup import post_setup
        post_setup(hermes_home, config)

    def _oss_hint(self, template: str, default: str = "vector store") -> str:
        """OSS-only hint; ``{vs}`` is the configured vector-store provider. "" in other modes."""
        return template.format(vs=self._config.get("oss", {}).get("vector_store", {}).get("provider", default)) if self._mode == "oss" else ""

    def _create_backend(self):
        # Lazy-install the mem0 SDK before the backend imports it (honors security.allow_lazy_installs);
        # on failure the backend import raises the canonical error, captured below.
        with suppress(Exception):
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("memory.mem0", prompt=False)
        try:
            from . import _backend
            if self._mode == "oss":
                return _backend.OSSBackend(self._config.get("oss", {}))
            return _backend.SelfHostedBackend(self._api_key, self._host) if self._host else _backend.PlatformBackend(self._api_key)
        except Exception as e:
            logger.error("Mem0 backend failed to initialize (%s mode): %s", self._mode, e)
            self._init_error = str(e)
            return None

    def _is_breaker_open(self) -> bool:
        """True while the breaker is tripped; an expired cooldown resets the failure count."""
        with self._breaker_lock:
            if self._consecutive_failures >= _BREAKER_THRESHOLD and time.monotonic() < self._breaker_open_until:
                return True
            if self._consecutive_failures >= _BREAKER_THRESHOLD:
                self._consecutive_failures = 0
            return False

    def _format_error(self, prefix: str, exc: Exception) -> str:
        msg = f"{prefix}: {exc}"
        if any(s in str(exc).lower() for s in ("connection", "refused", "timeout")):
            msg += self._oss_hint(" (check that {vs} is running)")
        return msg

    def _record_success(self):
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self):
        with self._breaker_lock:
            self._consecutive_failures = count = self._consecutive_failures + 1
            if count >= _BREAKER_THRESHOLD:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
        if count >= _BREAKER_THRESHOLD:
            hint = self._oss_hint(" Check that your {vs} vector store is running and reachable.", "unknown")
            logger.warning("Mem0 circuit breaker tripped after %d consecutive failures. Pausing API calls for %ds.%s", count, _BREAKER_COOLDOWN_SECS, hint)

    def _try(self, call, log, msg: str):
        """Background-path wrapper: run ``call`` under the breaker; on error log ``msg`` and return None."""
        try:
            result = call()
        except Exception as e:
            self._record_failure()
            log(msg, e)
            return None
        self._record_success()
        return result

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = cfg = _load_config()
        self._mode, self._api_key, self._host = cfg.get("mode", "platform"), cfg.get("api_key", ""), cfg.get("host", "")
        # The "hermes" _load_config falls back to is the default profile's identity: a named profile
        # never runs under it unannounced (_identity.resolve_agent_id).
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
        self._agent_id = _identity.resolve_agent_id(cfg, home, env_agent_id=get_secret("MEM0_AGENT_ID", "") or "")
        # user_id precedence: operator-configured (env/mem0.json) > gateway-native id (kwargs) > _DEFAULT_USER_ID.
        # The literal placeholder counts as unset so wizard users still get gateway-native ids.
        configured = cfg.get("user_id")
        self._user_id = (None if configured == _DEFAULT_USER_ID else configured) or kwargs.get("user_id") or _DEFAULT_USER_ID
        # Persisted rerank preference: default for mem0_search when the model omits ``rerank``. Platform-only.
        _rr = cfg.get("rerank", False)
        self._rerank_default = _rr.lower() in ("true", "1", "yes") if isinstance(_rr, str) else bool(_rr)
        self._channel = kwargs.get("platform") or "cli"
        self._sync_max_chars = int(cfg.get("sync_max_chars") or _SYNC_MSG_MAX_CHARS)
        # Missing scope means own memories plus the shared layer; null/empty/malformed never means all.
        # The shared layer's name is a setting (see _DEFAULT_SHARED_AGENT_ID) rather than a default
        # allow-list of one, because a profile per customer or per department still needs one layer
        # it may read. It goes through the same validation as an explicit list: a blank, padded or
        # wildcard name fails the profile closed instead of widening recall.
        self._backend = None
        self._search_agent_ids = frozenset()
        self._shared_agent_id = str(cfg.get("shared_agent_id") or _DEFAULT_SHARED_AGENT_ID)
        ids = cfg.get("search_agent_ids", [self._agent_id, self._shared_agent_id])
        if (not isinstance(ids, list) or not ids or
                any(not isinstance(x, str) or not x.strip() or x != x.strip() or x == "*" for x in ids) or
                not isinstance(self._agent_id, str) or not self._agent_id.strip() or
                self._agent_id not in ids or not isinstance(self._user_id, str) or not self._user_id.strip()):
            raise ValueError("Mem0 search_agent_ids must be a nonempty list of exact IDs including agent_id")
        self._search_agent_ids = frozenset(ids)
        # Writing into the shared layer is deliberate (a parameter on mem0_add) and possible only
        # where that layer is also readable, so a write can never land somewhere recall would not
        # show it back. A profile that IS the shared identity has no separate target: every save it
        # makes is already the shared one.
        _sw = cfg.get("shared_writes", True)
        if isinstance(_sw, str):
            _sw = _sw.strip().lower() not in ("false", "0", "no", "off") if _sw.strip() else True
        self._shared_write_target = (
            self._shared_agent_id
            if bool(_sw) and self._shared_agent_id in self._search_agent_ids
            and self._shared_agent_id != self._agent_id else None)
        self._backend = self._create_backend()
        if self._backend and not self._atexit_registered:
            atexit.register(self._shutdown_backend)
            self._atexit_registered = True

    def _search(self, query: str, top_k: int = 10, rerank: bool = False, backend=None) -> list:
        # Scalar queries work with Mem0 V3 validation; fetch per scope avoids a
        # noisy foreign profile crowding all allowed results out of a postfilter.
        results = {}
        for agent_id in sorted(self._search_agent_ids):
            rows = (backend or self._backend).search(
                query, filters={"user_id": self._user_id, "agent_id": agent_id},
                top_k=top_k, rerank=rerank,
            )
            for row in rows:
                if (isinstance(row, dict) and row.get("user_id") == self._user_id
                        and row.get("agent_id") == agent_id and row.get("id")):
                    results[row["id"]] = row
        return sorted(results.values(), key=lambda r: float(r.get("score") or 0), reverse=True)[:top_k]

    def _add(self, messages: list, infer: bool, *, shared: bool = False):
        metadata = {"channel": self._channel} if self._channel else {}
        agent_id = self._agent_id
        if shared:
            # Provenance is the price of a writable shared layer: an entry everyone reads must say
            # which profile put it there, so a reader can weigh it and a bad one can be traced back.
            # ``written_by`` is the writing profile's own agent id, the same identity recall scopes on.
            agent_id = self._shared_write_target
            metadata["written_by"] = self._agent_id
        return self._backend.add(messages, user_id=self._user_id, agent_id=agent_id, infer=infer, metadata=metadata)

    def _shared_duplicate(self, content: str) -> bool:
        """True when this exact text already sits in shared memory.

        Guards the loop where a model recalls a shared fact and deliberately stores it straight back.
        Only an exact match after folding whitespace and case counts: judging near-duplicates is the
        backend's job, and refusing a deliberate write on a guess is worse than keeping a duplicate.
        Any backend trouble fails open for the same reason -- a transient error must not silently
        swallow something the model meant to share.
        """
        wanted = " ".join(content.split()).casefold()
        try:
            rows = self._backend.search(
                content, filters={"user_id": self._user_id, "agent_id": self._shared_write_target},
                top_k=10, rerank=False)
        except Exception:
            return False
        return any(" ".join(str(r.get("memory", "")).split()).casefold() == wanted
                   for r in rows if isinstance(r, dict))

    def scope_note(self) -> str:
        """What recall covers and where a write lands, in the words the model reads.

        Derived from the resolved scope rather than fixed, so a profile with no shared memory
        is not told it has any, and a profile that IS the shared identity -- its own agent_id
        is the shared name, so its writes are what the others read -- is told that instead.
        """
        if not _shared_scopes(self._agent_id, self._search_agent_ids):
            note = "Recall covers your own memories only, and that is also where everything you store goes."
            if self._agent_id == self._shared_agent_id:
                note += " Other profiles recall from that same memory, so treat what you store as shared."
            return note
        if self._shared_write_target:
            return (
                "Recall covers your own memories and shared memory. What you store goes into your own "
                "memories unless you pass shared on mem0_add, which puts that one fact where every "
                "profile sharing this memory can read it — so never put anything private to this user "
                "there. You may also correct or remove entries in shared memory. Do not present what "
                "you recall from shared memory as something this user told you."
            )
        return (
            "Recall covers your own memories and shared memory you can read but not write. "
            "Everything you store goes into your own memories, and only those can be updated or "
            "deleted. Do not offer to save anything into shared memory, and do not present what "
            "you recall from it as something this user told you."
        )

    def system_prompt_block(self) -> str:
        # Mirror _create_backend precedence (oss > host > platform). Rerank is a Mem0 Platform feature only.
        mode_label = "OSS (self-hosted)" if self._mode == "oss" else "self-hosted (HTTP API)" if self._host else "platform (cloud API)"
        rerank_note = " Rerank is available on search." if (self._mode == "platform" and not self._host) else ""
        return f"# Mem0 Memory\nActive. Mode: {mode_label}. User: {self._user_id}.\n{_PROMPT_BODY}{rerank_note}\n{self.scope_note()}"

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._start_prefetch(message)

    def _consume_prefetch_result(self, query: str) -> str | None:
        """Pop the finished prefetch body for ``query`` (None if absent or still running)."""
        with self._prefetch_lock:
            if self._prefetch_query != query or not self._prefetch_done:
                return None
            result, self._prefetch_result, self._prefetch_done = self._prefetch_result, "", False
            return result

    def _start_prefetch(self, query: str) -> None:
        backend = self._backend
        if not query or backend is None or self._is_breaker_open():
            return

        def _run():
            results = self._try(lambda: self._search(query, backend=backend), logger.debug, "Mem0 prefetch failed: %s")
            lines = [r.get("memory", "") for r in (results or []) if r.get("memory")]
            body = "## Mem0 Memory\n" + "\n".join(f"- {l}" for l in lines) if lines else ""
            with self._prefetch_lock:
                if self._prefetch_query == query:
                    self._prefetch_result, self._prefetch_done = body, True

        with self._prefetch_lock:
            # Same query already answered or still in flight: don't restart it.
            if self._prefetch_query == query and (self._prefetch_done or (self._prefetch_thread and self._prefetch_thread.is_alive())):
                return
            self._prefetch_query, self._prefetch_result, self._prefetch_done = query, "", False
            self._prefetch_thread = t = spawn_context_thread(_run, name="mem0-prefetch")
        t.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall memories for the CURRENT question with a short hot-path wait."""
        if (cached := self._consume_prefetch_result(query)) is not None:
            return cached
        self._start_prefetch(query)
        with self._prefetch_lock:
            thread = self._prefetch_thread if self._prefetch_query == query else None
        if thread:
            thread.join(timeout=_PREFETCH_WAIT_SECS)
        return self._consume_prefetch_result(query) or ""  # slow backend: skip injection; mem0_search remains the backstop

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for server-side fact extraction (non-blocking)."""
        if self._backend is None or self._is_breaker_open():
            return

        def _sync():
            if self._backend is not None:
                messages = [
                    {"role": "user", "content": _truncate_for_sync(user_content, self._sync_max_chars)},
                    {"role": "assistant", "content": _truncate_for_sync(assistant_content, self._sync_max_chars)},
                ]
                self._try(lambda: self._add(messages, infer=True), logger.warning, "Mem0 sync failed: %s")

        with self._sync_lock:
            prev = self._sync_thread
            if prev and prev.is_alive():
                prev.join(timeout=5.0)
                if prev.is_alive():  # still busy after the wait: skip to avoid duplicate ingestion
                    return
            self._sync_thread = spawn_context_thread(_sync, name="mem0-sync")
            self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if not _shared_scopes(self._agent_id, self._search_agent_ids):
            return list(TOOL_SCHEMAS)  # own memories only: nothing to distinguish, say nothing
        writable = bool(self._shared_write_target)
        notes = _SHARED_WRITE_TOOL_NOTES if writable else _SHARED_SCOPE_TOOL_NOTES
        schemas = []
        for base in TOOL_SCHEMAS:
            schema = {**base, "description": base["description"] + notes.get(base["name"], "")}
            if writable and base["name"] == "mem0_add":
                # The parameter exists only where there is somewhere to put it: a profile with a
                # read-only or absent shared layer is never offered a knob it cannot use.
                params = {**schema["parameters"]}
                params["properties"] = {**params["properties"], **_SHARED_ADD_PARAM}
                schema["parameters"] = params
            schemas.append(schema)
        return schemas

    # -- tool handlers: (required params, error label, body, client-error policy) ---
    # Client errors (bad ID / not found) never trip the breaker, except for mem0_add
    # where they count as failures; update/delete answer them with "Memory not found".

    def _tool_search(self, args: dict) -> str:
        top_k = max(1, min(int(args.get("top_k", 10)), 50))
        rerank_raw = args.get("rerank", self._rerank_default)
        rerank = rerank_raw.lower() not in ("false", "0", "no") if isinstance(rerank_raw, str) else bool(rerank_raw)
        results = self._search(args["query"], top_k, rerank)
        if not results:
            return json.dumps({"result": "No relevant memories found."})
        items = [{"id": r.get("id"), "memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
        return json.dumps({"results": items, "count": len(items)})

    def _tool_add(self, args: dict) -> str:
        # Only an explicit parameter shares. The automatic turn sync never passes it, so nothing
        # reaches the shared layer as a side effect of a conversation.
        shared = _as_bool(args.get("shared"))
        if shared and not self._shared_write_target:
            return tool_error("This profile has no shared memory to store into")
        if shared and self._shared_duplicate(args["content"]):
            return json.dumps({"result": "Already in shared memory; nothing stored."})
        result = self._add([{"role": "user", "content": args["content"]}], infer=False, shared=shared)
        event_id = result.get("event_id") if isinstance(result, dict) else None
        # Cloud add is async (server-side extraction); OSS and self-hosted store synchronously.
        msg = "Fact stored." if (self._mode == "oss" or self._host) else "Fact queued for storage."
        if shared:  # say where it went, so the turn and its transcript both show it was shared
            msg = f"{msg} In shared memory: every profile that shares it can read this."
        return json.dumps({"result": msg, "event_id": event_id})

    def _tool_mutate(self, args: dict, *, delete: bool = False) -> str:
        # Search grants read access to shared facts, never write access. Check the
        # current stored payload, not a prior search result supplied by the model.
        row = self._backend.get(args["memory_id"])
        # A profile that may write shared memory may also correct or remove what is in it, whoever
        # wrote it -- ``written_by`` is what makes that traceable. Every other scope stays owner-only.
        allowed = {self._agent_id}
        if self._shared_write_target:
            allowed.add(self._shared_write_target)
        if (not isinstance(row, dict) or row.get("user_id") != self._user_id
                or row.get("agent_id") not in allowed):
            return tool_error("Memory not found or not owned by this profile")
        result = (self._backend.delete(args["memory_id"]) if delete else
                  self._backend.update(args["memory_id"], args["text"]))
        return json.dumps(result)

    _TOOL_HANDLERS = {
        "mem0_search": (("query",), "Search failed", _tool_search, "skip"),
        "mem0_add": (("content",), "Failed to store", _tool_add, "count"),
        "mem0_update": (("memory_id", "text"), "Update failed", lambda self, a: self._tool_mutate(a), "not_found"),
        "mem0_delete": (("memory_id",), "Delete failed", lambda self, a: self._tool_mutate(a, delete=True), "not_found"),
    }

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._backend is None:
            err = getattr(self, "_init_error", "unknown error")
            return json.dumps({"error": f"Mem0 backend not initialized: {err}.{self._oss_hint(' Check that {vs} is running and reachable.')}"})
        if self._is_breaker_open():
            return json.dumps({"error": f"Mem0 temporarily unavailable (multiple consecutive failures). Will retry automatically.{self._oss_hint(' Check that your {vs} is running.')}"})
        if tool_name not in self._TOOL_HANDLERS:
            return tool_error(f"Unknown tool: {tool_name}")
        required, label, body, on_client_error = self._TOOL_HANDLERS[tool_name]
        if missing := next((k for k in required if not args.get(k, "")), None):
            return tool_error(f"Missing required parameter: {missing}")
        try:
            result = body(self, args)
        except Exception as e:
            client = _is_client_error(e)
            if client and on_client_error == "not_found":
                return tool_error(f"Memory not found: {args['memory_id']}")
            if not client or on_client_error == "count":
                self._record_failure()
            return tool_error(self._format_error(label, e))
        self._record_success()
        return result

    def _shutdown_backend(self):
        with suppress(Exception):
            if self._backend:
                self._backend.close()
                self._backend = None

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        self._shutdown_backend()


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

ADD_SCHEMA = {
    "name": "mem0_add",
    "description": (
        "Store a durable fact about the user, verbatim (no LLM extraction). "
        "Call this the moment the user states a lasting preference, correction, "
        "decision, or personal detail worth recalling on future turns — don't "
        "wait to be asked to remember. Skip transient chit-chat and facts you've "
        "already stored."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to store."},
        },
        "required": ["content"],
    },
}

DELETE_SCHEMA = {
    "name": "mem0_delete",
    "description": (
        "Delete a memory by its ID (take the ID from a mem0_search "
        "result). Use when a stored fact is obsolete or the user asks you to "
        "forget it; prefer mem0_update if the fact merely changed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to delete."},
        },
        "required": ["memory_id"],
    },
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search the user's memories by meaning; returns facts ranked by "
        "relevance. Use this before answering any question that may depend on "
        "what you know about the user (preferences, facts, history, people, "
        "projects, past decisions). For multi-part or multi-hop questions, "
        "call it several times — vary the wording and run follow-up searches "
        "on what earlier results reveal; one search is rarely enough."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
            "rerank": {"type": "boolean", "description": "Rerank results for relevance (default: false, platform mode only)."},
        },
        "required": ["query"],
    },
}

UPDATE_SCHEMA = {
    "name": "mem0_update",
    "description": (
        "Replace the text of an existing memory by its ID (take the ID from a "
        "mem0_search result). Use when a stored fact has changed "
        "or was wrong — correct it in place instead of adding a duplicate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory UUID to update."},
            "text": {"type": "string", "description": "New text content."},
        },
        "required": ["memory_id", "text"],
    },
}
# ---- END PLUGIN-COMPAT ----
