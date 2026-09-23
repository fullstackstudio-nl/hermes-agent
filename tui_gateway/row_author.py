"""Who wrote a turn, written onto the row it persists.

A client cannot tell who sent a message it did not send itself. ``message.start`` is contracted
with no payload, and two clients on one session share a ``FanoutTransport``, so the socket says
nothing about the other person's turn; the row has no author column and no wire field either, so a
reload leaves nothing to read. Every client therefore treats an unmarked user row as its own, and a
colleague's sentence is painted as the reader's -- a false statement, not a gap, and one no client
can repair for itself, because a locally inferred marker does not survive a re-hydration.

``display_metadata`` is the one per-row field that crosses the SQLite boundary without a schema or
a wire-contract change: it is free-form TEXT, and both the REST read and the WS history projection
already forward it whole. So the author rides there, under ``"author"``.

A leaf module on purpose. The gateway's split modules have their bodies re-created against
``server.py``'s globals (``method_ctx.bind_module``), and tests re-create individual turn helpers
against a namespace they build by hand (``method_ctx.rebind``) -- so a bare global reference added
to one of those bodies breaks every such test. Callers import from here inside the function that
needs it, which is the idiom those bodies already use for everything else.
"""

#: Advertised through ``gateway.capabilities``. A module constant beside the code that does the
#: writing, not a config flag -- the same rule ``PER_SESSION_EXCLUSIVE_SUBMIT`` states for itself:
#: it holds because :func:`with_row_author` below is what every submitted turn goes through, so it
#: cannot drift from the behaviour without this file changing.
#:
#: It answers the one question no per-row read can: whether THIS gateway attributes messages at
#: all. A client needs that BEFORE it draws anything, so it can lay a transcript out with room for
#: a sender's name instead of having rows shift as authored ones arrive -- and so a client talking
#: to a gateway that does not attribute keeps behaving exactly as it did before any of this
#: existed. Because it is answered by the RUNNING process, a gateway carrying this code on disk but
#: not yet restarted answers without the key at all, which is the honest answer: it is not stamping
#: anything yet.
PER_MESSAGE_AUTHOR = True


def row_author(auth_user: tuple[str | None, str] | None) -> dict | None:
    """``{"id": "<provider>:<user id>", "name": ...}`` naming who wrote a turn's user row, or None
    where the gateway cannot prove it.

    ``auth_user`` is the identity of the connection that SUBMITTED the turn, minted at the WS
    upgrade from a verified ticket and carried into the turn by ``prompt.submit`` -- never a value
    any RPC parameter can reach. None where that connection names no login (stdio, the legacy
    token, the PTY child's server-internal credential), and the unattributed-turn sentinel a crash
    continuation, a wake-up, a cron run or a bot delivery binds needs no special case here: it
    names no login by construction, so the check below already refuses it.

    A session record's own ``auth_user_id`` is deliberately NOT a fallback. It names the login the
    conversation was created under, which is not evidence that its owner typed THIS message, and a
    row is read back long after every live signal is gone.

    ``name`` is omitted rather than empty, because a reader falls back to its own directory or to
    the id, where an empty string would read as a person who has no name."""
    if not auth_user:
        return None
    user_id, user_name = auth_user
    if not user_id:
        return None
    return {"id": user_id, **({"name": user_name} if user_name else {})}


def with_row_author(display_metadata: dict | None, auth_user) -> dict | None:
    """``display_metadata`` with this turn's author merged in under ``"author"``, or unchanged when
    there is nothing to assert.

    Every key already on the dict is kept as it is -- the gateway's own ``title_preview``,
    ``notification_category``, a timeline shape -- and a row whose only metadata is the author
    still gets one."""
    author = row_author(auth_user)
    if author is None:
        return display_metadata
    return {**(display_metadata or {}), "author": author}
