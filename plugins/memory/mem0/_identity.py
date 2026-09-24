"""A profile's own mem0 identity.

``agent_id`` is the scope a profile's memories are written under and, by default, recalled from. The
built-in ``hermes`` belongs to the default profile (or a single-home install). Every named profile
gets one of its own the moment it exists -- created, cloned, imported or installed from a
distribution, and whether or not it uses mem0 yet -- derived from its name plus a random suffix, so
a later profile of the same name, after a rename or a delete, never inherits the memories of the one
before. Sharing stays possible and stays explicit: ``search_agent_ids`` / ``shared_agent_id``, or an
``agent_id`` somebody writes down.

A named profile that already existed before this has no identity of its own and has been running
under ``hermes``. Handing it a new one would drop everything it can recall today out of recall, so
its first start writes the identity it already uses into its own ``mem0.json`` instead, marked as the
old fallback, and warns until an operator decides. That write only ever adds to a file this process
can read, parse and owns; anything else is left exactly as it is.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from pathlib import Path
from typing import Callable, Optional, Tuple

from agent.secret_scope import get_secret
from utils import atomic_json_write, read_json_or_empty

logger = logging.getLogger(__name__)

# The identity every profile without one of its own resolved to, the default profile included.
LEGACY_AGENT_ID = "hermes"
# Marks an ``agent_id`` the provider wrote down for a profile that had none, rather than one somebody
# chose. Kept in the profile's ``mem0.json`` so an operator can find those profiles and decide.
SOURCE_KEY = "agent_id_source"
LEGACY_SOURCE = "legacy-default"

_warned_homes: set[str] = set()  # once per process and profile, not once per session


def new_agent_id(profile_name: str) -> str:
    return f"hermes-{profile_name}-{secrets.token_hex(4)}"


def configured_agent_id(home) -> Optional[str]:
    """The ``agent_id`` *home*'s own files name -- ``mem0.json`` over ``.env``, the precedence the
    provider applies -- or None."""
    return current_identity(home)[0]


def current_identity(home) -> Tuple[Optional[str], Optional[str]]:
    """``(agent_id, agent_id_source)`` as *home*'s own files state it; the source only when the
    ``agent_id`` comes from ``mem0.json``."""
    home = Path(home)
    data = read_json_or_empty(home / "mem0.json")
    agent_id = data.get("agent_id")
    if isinstance(agent_id, str) and agent_id:
        return agent_id, data.get(SOURCE_KEY)
    from agent.secret_scope import load_env_file
    agent_id = load_env_file(home / ".env").get("MEM0_AGENT_ID")
    return (agent_id if isinstance(agent_id, str) and agent_id else None), None


def profile_name_of(home) -> Optional[str]:
    """The named profile *home* is, or None for the default profile and a single-home install."""
    from hermes_constants import PROFILE_ID_RE, get_default_hermes_root
    try:
        rel = Path(home).resolve().relative_to((get_default_hermes_root() / "profiles").resolve())
    except (OSError, ValueError):
        return None
    return rel.parts[0] if len(rel.parts) == 1 and PROFILE_ID_RE.match(rel.parts[0]) else None


def setup_default_agent_id(home) -> str:
    """The ``agent_id`` setup offers -- the CLI wizard and the dashboard form alike, whose every save
    writes it -- which is always the identity the profile would run under now, so a save that only
    meant to change another setting never switches its memory: the one its own files name, else a
    ``MEM0_AGENT_ID`` the process supplies (a service environment), else, for a named profile, a new
    one of its own only when its config loads and names another memory provider. A named profile on
    mem0, or whose config cannot be read, is offered ``hermes`` -- what it has been running under --
    and ``settle_saved_identity`` marks a save of it as the pin. The default profile gets ``hermes``."""
    if current := configured_agent_id(home) or _process_agent_id(home):
        return current
    return new_agent_id(profile_name_of(home)) if _may_get_a_new_identity(home) else LEGACY_AGENT_ID


def _may_get_a_new_identity(home) -> bool:
    """A named profile whose config loads and selects another memory provider (or none): it has not
    been running on mem0, so there is nothing it could lose."""
    return bool(profile_name_of(home)) and _mem0_state(home) is False


def settle_saved_identity(home, before: dict, merged: dict) -> dict:
    """The mem0 settings a setup save writes over *before* (the file as it was), with the pin kept
    straight: saving ``hermes`` for a named profile that had no identity writes the pin marker, so
    the profile stays pinned and warned about rather than turning into explicit sharing; saving an
    identity of its own over the pin drops the marker and moves the ``hermes`` entry of an explicit
    ``search_agent_ids`` to the new identity, or the profile would fail closed."""
    merged = dict(merged)
    new_id = merged.get("agent_id")
    # Ran under the fallback until now: pinned, or a named profile nothing gave an identity.
    was_fallback = _is_pin(before) or (bool(profile_name_of(home)) and not before.get("agent_id")
                                       and not _own_env_agent_id(home) and not _process_agent_id(home))
    if new_id != LEGACY_AGENT_ID:
        merged.pop(SOURCE_KEY, None)
        if new_id and was_fallback:
            _swap_scope(merged, LEGACY_AGENT_ID, new_id)
    elif was_fallback and not _may_get_a_new_identity(home):
        # ``hermes`` is what setup offered, so saving it is not a choice to share: keep it the pin.
        # A profile that was offered an identity of its own and typed ``hermes`` chose it.
        merged[SOURCE_KEY] = LEGACY_SOURCE
    return merged


def _mem0_state(home) -> Optional[bool]:
    """Whether *home*'s effective config selects mem0: True or False when its own ``config.yaml``
    loads, None when there is none or it cannot be read -- which callers treat like mem0, the side
    on which nothing it may have stored is dropped out of recall."""
    path = Path(home) / "config.yaml"
    try:
        if not path.is_file():
            return None
        from hermes_cli.config import read_user_config_raw
        read_user_config_raw(path)  # parse-health only: the effective loader forgives a broken file
        from hermes_cli.memory_provider_migration import configured_provider
        return configured_provider(Path(home)) == "mem0"
    except Exception:
        logger.debug("mem0: could not read the memory provider of %s", home, exc_info=True)
        return None


def _process_agent_id(home) -> str:
    """``MEM0_AGENT_ID`` from the process -- a service ``Environment=`` -- when *home* is the profile
    this process runs as, and "" for any other profile.

    The process environment belongs to the process's own profile. A dashboard edits other profiles by
    parameter under a home override and that profile's secret scope, and in a single-profile process
    a scope miss falls through to ``os.environ``: without this check a named profile would be offered,
    and saved with, the launch profile's identity. The comparison is with the process home, not
    ``get_hermes_home()``, because under that override the active home is the edited profile."""
    from hermes_constants import get_process_hermes_home
    try:
        if Path(home).resolve() != get_process_hermes_home().resolve():
            return ""
    except OSError:
        return ""
    try:
        value = get_secret("MEM0_AGENT_ID", "")
    except Exception:  # no scope under multiplexing: nothing process-wide applies to this profile
        return ""
    return value if isinstance(value, str) else ""


def _own_env_agent_id(home) -> str:
    """``MEM0_AGENT_ID`` from *home*'s own ``.env``, never the process environment."""
    from agent.secret_scope import load_env_file
    value = load_env_file(Path(home) / ".env").get("MEM0_AGENT_ID")
    return value if isinstance(value, str) else ""


def _is_pin(data: dict) -> bool:
    return data.get("agent_id") == LEGACY_AGENT_ID and data.get(SOURCE_KEY) == LEGACY_SOURCE


def _swap_scope(cfg: dict, old: str, new: str) -> None:
    ids = cfg.get("search_agent_ids")
    if isinstance(ids, list):
        swapped = [new if entry == old else entry for entry in ids]
        cfg["search_agent_ids"] = [e for i, e in enumerate(swapped) if e not in swapped[:i]]


def give_own_identity(profile_dir, profile_name: str, *, keep: Optional[str] = None,
                      keep_source: Optional[str] = None, fresh: bool = True) -> Optional[str]:
    """Write a new profile's own ``agent_id`` into its ``mem0.json`` and return it.

    Whatever ``mem0.json`` or ``.env`` the profile was copied with (a clone, an import, a
    distribution) names another profile's identity, and ``mem0.json`` outranks ``.env``, so the one
    written here is the one that runs. *keep* is this same profile's identity from before a re-copy
    (a distribution reinstall or update) and is kept, with its *keep_source* marker. An explicit
    ``search_agent_ids`` list that came with a full copy is kept as the sharing setting it is, with
    the entry that meant the source's own scope now meaning this profile's. Every other mem0
    setting is left as copied.

    A ``mem0.json`` that cannot be read as an object is never overwritten. In a profile being made
    (*fresh*) it is a copy: it is kept beside as ``mem0.json.invalid`` and the new file starts empty,
    because the profile needs an identity of its own. In an existing profile (a distribution
    reinstall or update) it is left alone with a warning and None is returned; the provider's start
    then warns about it too.
    """
    path = Path(profile_dir) / "mem0.json"
    try:
        cfg = _read_strict(path) or {}
    except (OSError, ValueError) as exc:
        if not fresh:
            logger.warning("mem0: %s cannot be read as a JSON object (%s); left untouched, so profile %r "
                           "gets no identity of its own until it is fixed", path, exc, profile_name)
            return None
        aside = path.with_name("mem0.json.invalid")
        os.replace(path, aside)
        logger.warning("mem0: the %s copied into new profile %r cannot be read as a JSON object (%s); "
                       "kept as %s and started a new one", path.name, profile_name, exc, aside)
        cfg = {}
    # "The source's own": what its files name, else what a single-profile process env named for it.
    inherited = configured_agent_id(profile_dir) or os.environ.get("MEM0_AGENT_ID") or LEGACY_AGENT_ID
    agent_id = keep or new_agent_id(profile_name)
    ids = cfg.get("search_agent_ids")
    if isinstance(ids, list):
        cfg["search_agent_ids"] = [agent_id if entry == inherited else entry for entry in ids]
    cfg["agent_id"] = agent_id
    cfg.pop(SOURCE_KEY, None)
    if keep and keep_source:
        cfg[SOURCE_KEY] = keep_source
    atomic_json_write(path, cfg, mode=0o600)
    return agent_id


def resolve_agent_id(cfg: dict, home, *, env_agent_id: str = "") -> str:
    """The ``agent_id`` a start of *home* runs under.

    *cfg* is the loaded mem0 config (``mem0.json`` over ``.env``) and *env_agent_id* what
    ``MEM0_AGENT_ID`` resolves to on its own. A configured identity is used as is, except the
    provider's own pin: it only stood in for an identity nobody had set, so a ``MEM0_AGENT_ID`` set
    since replaces it and the pin is removed. Without any, the default profile keeps the built-in
    identity it always had and nothing is written. A named profile without one predates
    per-profile identities (every path that makes a profile now writes one), so it keeps the
    identity it has been running under -- written into its own ``mem0.json`` where that is safe,
    marked, and warned about -- rather than silently switching and losing recall of what it stored.
    """
    home = Path(home)
    name = profile_name_of(home)
    file_id = read_json_or_empty(home / "mem0.json").get("agent_id")
    pinned = file_id == LEGACY_AGENT_ID and cfg.get(SOURCE_KEY) == LEGACY_SOURCE
    # Only the profile's own .env un-pins: a process-wide MEM0_AGENT_ID (a systemd Environment=) would
    # un-pin in a single-profile process and the next multiplexed start would pin again.
    own_env = _own_env_agent_id(home) if pinned else ""
    if own_env:
        _swap_scope(cfg, LEGACY_AGENT_ID, own_env)  # this start, even if the file cannot be written
        written, _ = _rewrite(home, name, _drop_pin(own_env), "keeps its pin in mem0.json")
        if written:
            logger.info("mem0: profile %r now runs under MEM0_AGENT_ID %r from its .env; removed the "
                        "fallback pinned in %s", name, own_env, home / "mem0.json")
        return own_env
    if file_id or env_agent_id:
        if pinned and name:
            _warn_legacy(home, name, written=True)
        return cfg.get("agent_id") or env_agent_id
    if name is None:
        return LEGACY_AGENT_ID
    written, on_disk = _rewrite(home, name, _pin,
                                f"runs under {LEGACY_AGENT_ID!r} this start without it being written down")
    saved = (on_disk or {}).get("agent_id")
    if saved and not _is_pin(on_disk):  # saved meanwhile (a setup or dashboard save): that one runs
        return saved
    _warn_legacy(home, name, written=written or bool(saved))
    return LEGACY_AGENT_ID


def _pin(data: dict) -> Optional[dict]:
    if data.get("agent_id"):  # set meanwhile (a setup or dashboard save): theirs stands
        return None
    return {**data, "agent_id": LEGACY_AGENT_ID, SOURCE_KEY: LEGACY_SOURCE}


def _drop_pin(new_id: str) -> Callable[[dict], Optional[dict]]:
    def change(data: dict) -> Optional[dict]:
        if not _is_pin(data):
            return None
        out = {k: v for k, v in data.items() if k not in ("agent_id", SOURCE_KEY)}
        _swap_scope(out, LEGACY_AGENT_ID, new_id)
        return out
    return change


def _read_strict(path: Path) -> Optional[dict]:
    """The object in *path*; None when there is no file. Raises OSError when it cannot be read and
    ValueError when it is not a JSON object -- the two cases ``read_json_or_empty`` folds into {}."""
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    return data


def _rewrite(home: Path, name: Optional[str], change: Callable[[dict], Optional[dict]],
             otherwise: str) -> Tuple[bool, Optional[dict]]:
    """Apply *change* to *home*'s ``mem0.json`` in one read-then-write step, only where that cannot
    lose anything: the file is absent, or it reads and parses as an object and belongs to this
    process's user. The read is fresh, so a value saved since the caller looked is seen by *change*,
    which returns the new content or None to leave the file alone. An existing file keeps its mode.
    Anything else is logged with *otherwise* and nothing is written. Returns whether the file was
    written and what it holds now (None when it could not be read)."""
    path = home / "mem0.json"
    if not home.is_dir():
        return False, None
    try:
        data = _read_strict(path)
        euid = getattr(os, "geteuid", None)  # POSIX only; elsewhere ownership is not checked
        owner_ok = data is None or euid is None or path.stat().st_uid == euid()
    except (OSError, ValueError) as exc:
        logger.warning("mem0: %s cannot be read as a JSON object (%s). It is left untouched; profile %r %s. "
                       "Fix or remove the file.", path, exc, name, otherwise)
        return False, None
    if not owner_ok:
        logger.warning("mem0: %s belongs to another user. It is left untouched; profile %r %s.", path, name, otherwise)
        return False, data
    new = change(dict(data or {}))
    if new is None:
        return False, data
    try:
        mode = (path.stat().st_mode & 0o777) if data is not None else 0o600
        atomic_json_write(path, new, mode=mode)
    except OSError as exc:
        logger.warning("mem0: could not write %s (%s); profile %r %s.", path, exc, name, otherwise)
        return False, data
    return True, new


def _warn_legacy(home: Path, name: str, *, written: bool) -> None:
    key = str(home)
    if key in _warned_homes:
        return
    _warned_homes.add(key)
    path = home / "mem0.json"
    where = f"This is kept as it was and written down in {path}." if written else (
        f"This is kept as it was; it could not be written down in {path} (see above).")
    logger.warning(
        "mem0: profile %r has no memory identity of its own. It runs under %r, the identity the default "
        "profile uses unless that names another, so the two read and write the same memories. %s "
        "To give the profile its own memory, set \"agent_id\" in %s to a new name (for example %r) -- or "
        "MEM0_AGENT_ID in its .env -- and remove %r; to keep sharing on purpose, remove %r.",
        name, LEGACY_AGENT_ID, where, path, new_agent_id(name), SOURCE_KEY, SOURCE_KEY)
