"""Contract test: a session two people share must not assert either of them.

``auth_user_id`` names the login the session record was CREATED under. Nothing
re-stamps it when a second window attaches — ``_note_foreign_login`` only logs,
and the transport slot becomes a FanoutTransport, which names no login at all.
So on a shared session the stamp is not an answer to "who is acting now": it is
whoever opened the conversation, and every reader of HERMES_SESSION_USER_ID /
_USER_NAME (kanban attribution, send_message, cron job args, the
terminal_tool_background watcher fields, per-person authorisation limits) takes
it as one anyway.

Empty is a state all of those already handle — an ungated gateway binds it on
every turn. Wrong is not. So the ambiguous session binds nothing.

One person with two windows is NOT ambiguous and must keep working.
"""
import types

import pytest

from gateway.session_context import (
    _UNSET,
    _VAR_MAP,
    get_session_env,
)
from tui_gateway.transport import FanoutTransport
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


def _peer(provider, user_id, user_name=None):
    """A live client peer carrying a server-minted WS identity."""
    identity = {"provider": provider, "user_id": user_id}
    if user_name is not None:
        identity["user_name"] = user_name
    return types.SimpleNamespace(auth_identity=identity)


def _install_session(monkeypatch, *, session_key, transport=None, **extra):
    sess = {
        "session_key": session_key,
        "source": "desktop",
        "agent": _FakeAgent("20260921_cafebabe"),
        "cwd": "/home/user",
        **extra,
    }
    if transport is not None:
        sess["transport"] = transport
    monkeypatch.setattr(server, "_sessions", {session_key: sess}, raising=False)
    return sess


def _bound(session_key):
    tokens = server._set_session_context(session_key)
    try:
        return (
            get_session_env("HERMES_SESSION_USER_ID"),
            get_session_env("HERMES_SESSION_USER_ID_ALT"),
            get_session_env("HERMES_SESSION_USER_NAME"),
        )
    finally:
        server._clear_session_context(tokens)


def test_two_signed_in_members_on_one_session_bind_nothing(monkeypatch):
    """The reported failure. Two people are attached; the stamp names the one
    who opened the conversation, and a turn the other submitted would be
    attributed to them. Neither may be asserted."""
    creator, joiner = _peer("oidc", "user-a", "Robin"), _peer("oidc", "user-b", "Sam")
    _install_session(
        monkeypatch, session_key="skey-shared",
        transport=FanoutTransport(creator, joiner),
        auth_user_id="oidc:user-a", auth_user_name="Robin",
    )

    assert _bound("skey-shared") == ("", "", "")


def test_a_second_login_that_has_left_still_binds_nothing(monkeypatch):
    """Sticky: the second person leaving does not turn the creator's stamp back
    into proof of who acted, and the slot alone can no longer tell us apart."""
    creator = _peer("oidc", "user-a", "Robin")
    sess = _install_session(
        monkeypatch, session_key="skey-was-shared", transport=creator,
        auth_user_id="oidc:user-a", auth_user_name="Robin",
    )
    server._attach_session_transport(sess, _peer("oidc", "user-b", "Sam"))
    assert sess.get("auth_user_shared") is True
    server._detach_session_transport(sess, sess["transport"].transports()[1])

    assert _bound("skey-was-shared") == ("", "", "")


def test_one_person_in_two_windows_still_binds(monkeypatch):
    """A second window of the SAME login is not two people: the pop-out, the
    HUD and the phone all attach as the same person and attribution must keep
    working. This is the case the fanout arm used to break on."""
    first, second = _peer("oidc", "user-a", "Robin"), _peer("oidc", "user-a", "Robin")
    sess = _install_session(
        monkeypatch, session_key="skey-two-windows", transport=first,
        auth_user_id="oidc:user-a", auth_user_name="Robin",
    )
    server._attach_session_transport(sess, second)
    assert isinstance(sess["transport"], FanoutTransport)
    assert not sess.get("auth_user_shared")

    assert _bound("skey-two-windows") == ("oidc:user-a", "", "Robin")


def test_a_slot_that_names_no_login_still_binds_the_stamp(monkeypatch):
    """A parked session, and the compute-host child whose only peer is the host
    pipe, have no competing person attached — the stamp is the one identity in
    play and stays bindable. Fail closed means "when two could have acted", not
    "whenever no socket corroborates"."""
    _install_session(
        monkeypatch, session_key="skey-relayed", transport=types.SimpleNamespace(),
        auth_user_id="oidc:user-a", auth_user_name="Robin",
    )

    assert _bound("skey-relayed") == ("oidc:user-a", "", "Robin")


def test_a_foreign_login_attaching_marks_the_record(monkeypatch, caplog):
    """The mark is set where the foreign login is already noticed, so no attach
    path can share a session without recording that it is shared."""
    sess = _install_session(
        monkeypatch, session_key="skey-mark", transport=_peer("oidc", "user-a"),
        auth_user_id="oidc:user-a", auth_user_name="Robin",
    )

    with caplog.at_level("WARNING"):
        server._attach_session_transport(sess, _peer("basic", "user-b"))

    assert sess["auth_user_shared"] is True
    assert "a client logged in as" in caplog.text


def test_an_unattributable_slot_login_binds_nothing(monkeypatch):
    """Defence in depth for a peer that reached the slot without passing the
    attach path: a login attached that the stamp does not name is unresolved,
    whatever set the slot."""
    _install_session(
        monkeypatch, session_key="skey-unmarked",
        transport=FanoutTransport(_peer("oidc", "user-b", "Sam")),
        auth_user_id="oidc:user-a", auth_user_name="Robin",
    )

    assert _bound("skey-unmarked") == ("", "", "")
