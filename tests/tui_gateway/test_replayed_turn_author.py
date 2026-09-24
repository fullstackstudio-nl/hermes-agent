"""Contract test: a turn the gateway replays or starts by itself names only the person who typed it.

A user row's ``display_metadata.author`` is the one statement a shared chat makes about who sent
what, and it is stored for good. So a row is authored only by the identity of the connection that
submitted THAT text; when the gateway cannot tell, the row names nobody. Two places used to take
the author from the turn that happened to be running instead:

- a leftover steer. A steer that arrives after the final answer has no tool result to ride on, so
  the agent hands it back and the gateway queues it as the next turn. It was queued under the
  identity of the turn it arrived in, so a colleague's words were written down as the turn owner's.
  The sender is now handed to the agent with the text and drained with it; text more than one
  person may have contributed to is queued with no identity at all.
- a turn nobody typed. The ``/goal`` continuation (and every other turn the gateway starts on its
  own) still runs scoped as the person whose work it continues -- memory, tools and permissions --
  but that is not authorship, and the row carries no author.

The same pairing stamps the rows a steer or redirect lands as MID-turn, which named nobody before --
and an unmarked row is drawn as the reader's own, so everyone saw a colleague's interjection as theirs.
"""
import threading
import time

import pytest

from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
from agent.conversation_loop import _apply_active_turn_redirect
from agent.interrupt_control import InterruptControlMixin
from agent.turn_finalizer import hand_back_leftover_steer
from agent.turn_iteration_prep import apply_pending_redirect, requeue_unapplied_redirect
from hermes_state import SessionDB
from tui_gateway.transport import bind_transport, reset_transport
import tui_gateway.server as server

ROBIN = ("oidc:user-a", "Robin")
SAM = ("oidc:user-b", "Sam")
AUTHOR_ROBIN = {"id": "oidc:user-a", "name": "Robin"}
AUTHOR_SAM = {"id": "oidc:user-b", "name": "Sam"}
CASEY = ("oidc:user-c", "Casey")
PAT = ("oidc:user-d", "Pat")
AUTHOR_PAT = {"id": "oidc:user-d", "name": "Pat"}


def _hand_back(agent, result):
    """The finalizer's own leftover handoff (``turn_finalizer.hand_back_leftover_steer``)."""
    hand_back_leftover_steer(agent, result)


class _Peer:
    """A live client connection carrying a server-minted WS identity."""

    def __init__(self, user):
        self.auth_identity = (
            {"provider": "oidc", "user_id": user[0].split(":", 1)[1], "user_name": user[1]} if user else None)

    def write(self, obj):
        return True

    def close(self):
        return None


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None, name=None):
        self._run = lambda: target(*args, **(kwargs or {}))

    def start(self):
        self._run()

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


class _Agent(InterruptControlMixin):
    """The real steer / redirect slots (``InterruptControlMixin``), with a scripted turn.

    Each ``run_conversation`` records what the turn would write on its user row and who the turn is
    scoped to, runs the next scripted step (what happens while the turn is live), and hands back a
    leftover steer exactly as ``turn_finalizer`` does: whatever is still in the slot at the end."""

    def __init__(self, session_key):
        self.session_id = session_key
        self._session_messages = []
        self._last_flushed_db_idx = 0
        self._db_flush_scan_prefix = []
        self._pending_steer = None
        self._pending_steer_lock = threading.Lock()
        self._pending_redirect = None
        self._pending_redirect_lock = threading.Lock()
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_interrupt_reason = None
        self._execution_thread_id = None
        self._model_request_active = threading.Event()
        self._supports_active_turn_redirect = False
        self._current_streamed_assistant_text = ""
        self.turns = []
        self.rows = []
        self.script = []
        self.session = None

    def _strip_think_blocks(self, text):
        return text

    def clear_interrupt(self, *_a, **_k):
        self._interrupt_requested = False
        return True

    def interrupt(self, *_a, **_k):
        self._interrupt_requested = True
        return True

    def run_conversation(self, message, conversation_history=None, stream_callback=None,
                         persist_user_display_metadata=None, **_kw):
        self.turns.append({
            "text": message,
            "author": (persist_user_display_metadata or {}).get("author"),
            "replayed_by": (persist_user_display_metadata or {}).get("replayed_by"),
            "scope": server._acting_auth_user(self.session),
        })
        if self.script:
            self.script.pop(0)()
        result = {"final_response": "done"}
        _hand_back(self, result)
        return result

    def deliver_mid_turn(self):
        """What the turn loop does between tool batches: the post-batch steer drain and the redirect
        apply, both the agent's own code. The rows they write are kept for inspection."""
        messages = [{"role": "assistant", "content": "", "tool_calls": []},
                    {"role": "tool", "tool_call_id": "t1", "content": "ok"}]
        apply_pending_steer_to_tool_results(self, messages, 1)
        apply_pending_redirect(self, messages, _apply_active_turn_redirect)
        self.rows += [m for m in messages[2:] if m.get("role") == "user"]


@pytest.fixture()
def room(tmp_path, monkeypatch):
    """One session two signed-in people share, wired for real ``prompt.submit`` / ``session.steer``
    calls against a real store. Every turn runs inline, so a follow-up is observable at once."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("room", source="desktop")
    robin, sam = _Peer(ROBIN), _Peer(SAM)
    agent = _Agent("room")
    session = {
        "agent": agent, "attached_images": [], "cols": 80, "cwd": str(tmp_path),
        "history": [], "history_lock": threading.Lock(), "history_version": 0,
        "inflight_turn": None, "running": False, "session_key": "room",
        "show_reasoning": False, "slash_worker": None, "source": "desktop",
        "tool_progress_mode": "all", "transport": robin,
        "auth_user_id": ROBIN[0], "auth_user_name": ROBIN[1],
    }
    agent.session = session
    monkeypatch.setattr(server, "_db", db, raising=False)
    monkeypatch.setattr(server, "_sessions", {"sid": session}, raising=False)
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "render_message", lambda *_a: "")
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    server._attach_session_transport(session, sam)
    assert session["auth_user_shared"] is True

    def call(peer, method, **params):
        token = bind_transport(peer)
        try:
            return server._methods[method]("rid", {"session_id": "sid", **params})
        finally:
            reset_transport(token)

    agent.db = db
    yield agent, call, robin, sam
    db.close()


def _replayed(agent, text):
    return [turn for turn in agent.turns if turn["text"] == text]


# ---------------------------------------------------------------------------
# A steer the agent hands back is authored by whoever steered
# ---------------------------------------------------------------------------

def test_a_colleagues_leftover_steer_is_never_stamped_as_the_turn_owner(room):
    """Robin's turn is writing its final answer when Sam steers. No tool result is left to take it,
    so it comes back as ``pending_steer`` and runs as the next turn -- and that turn's row is Sam's."""
    agent, call, robin, sam = room
    agent.script = [lambda: call(sam, "session.steer", text="use the staging data instead")]

    assert call(robin, "prompt.submit", text="summarise the report")["result"]["status"] == "streaming"

    assert agent.turns[0]["author"] == AUTHOR_ROBIN
    [steer_turn] = _replayed(agent, "use the staging data instead")
    assert steer_turn["author"] == AUTHOR_SAM
    assert steer_turn["scope"] == SAM


def test_the_owners_own_leftover_steer_stays_theirs(room):
    agent, call, robin, _sam = room
    agent.script = [lambda: call(robin, "session.steer", text="and keep it short")]

    call(robin, "prompt.submit", text="summarise the report")

    [steer_turn] = _replayed(agent, "and keep it short")
    assert steer_turn["author"] == AUTHOR_ROBIN


def test_a_leftover_two_people_steered_into_names_nobody(room):
    """The agent joins every pending steer into one text, so a leftover both of them contributed to
    has no single author. It runs unattributed rather than under either name."""
    agent, call, robin, sam = room

    def both_steer():
        call(robin, "session.steer", text="keep it short")
        call(sam, "session.steer", text="and cite the source")

    agent.script = [both_steer]
    call(robin, "prompt.submit", text="summarise the report")

    [steer_turn] = _replayed(agent, "keep it short\nand cite the source")
    assert steer_turn["author"] is None


def test_a_steer_accepted_while_idle_keeps_its_steerer_into_the_next_turn(room):
    """``session.steer`` on an idle session parks the text in the agent until the next turn -- which
    may be somebody else's. The turn it surfaces in does not make it that person's."""
    agent, call, robin, sam = room
    assert call(sam, "session.steer", text="check the totals too")["result"]["status"] == "queued"

    call(robin, "prompt.submit", text="summarise the report")

    [steer_turn] = _replayed(agent, "check the totals too")
    assert steer_turn["author"] == AUTHOR_SAM


def test_a_steer_typed_as_a_busy_message_keeps_its_submitter(room, monkeypatch):
    """``busy_input_mode: steer`` turns a mid-turn ``prompt.submit`` into a steer; same rule."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "steer")
    agent, call, robin, sam = room
    agent.script = [lambda: call(sam, "prompt.submit", text="use the staging data instead")]

    call(robin, "prompt.submit", text="summarise the report")

    [steer_turn] = _replayed(agent, "use the staging data instead")
    assert steer_turn["author"] == AUTHOR_SAM


def test_a_redirect_the_agent_hands_back_as_a_steer_keeps_its_submitter(room, monkeypatch):
    """``busy_input_mode: interrupt`` redirects. A redirect the turn could not apply (the restart
    cap) is handed back through the steer slot, and must not pick up the turn owner on the way."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    agent, call, robin, sam = room
    agent._supports_active_turn_redirect = True
    agent._model_request_active.set()

    def redirect_then_hit_the_cap():
        assert call(sam, "prompt.submit", text="stop, use last year")["result"]["status"] == "redirected"
        requeue_unapplied_redirect(agent)  # turn_iteration_prep's restart-cap handoff

    agent.script = [redirect_then_hit_the_cap]
    call(robin, "prompt.submit", text="summarise the report")

    [steer_turn] = _replayed(agent, "stop, use last year")
    assert steer_turn["author"] == AUTHOR_SAM


@pytest.mark.parametrize("mode", ["queue", "interrupt"])
def test_a_queued_or_interrupting_message_keeps_its_submitter(room, monkeypatch, mode):
    """``queue``, and ``interrupt`` where the agent cannot redirect, queue the message as its own
    turn with its own sender."""
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: mode)
    agent, call, robin, sam = room
    agent.script = [lambda: call(sam, "prompt.submit", text="then draft the email")]

    call(robin, "prompt.submit", text="summarise the report")

    [queued_turn] = _replayed(agent, "then draft the email")
    assert queued_turn["author"] == AUTHOR_SAM
    assert queued_turn["scope"] == SAM


# ---------------------------------------------------------------------------
# A turn nobody typed is scoped as a person but authored by nobody
# ---------------------------------------------------------------------------

def test_the_goal_continuation_runs_as_the_human_but_is_not_their_message(room, monkeypatch):
    """Nobody typed "[Continuing toward your standing goal] ...". The continuation keeps working for
    the person whose goal it is -- that is its scope -- but its row names no author."""
    agent, call, robin, _sam = room
    monkeypatch.setattr(
        server, "_goal_followup_after_turn",
        lambda _sid, _session, _result, _status, _raw: (
            "[Continuing toward your standing goal] keep going" if len(agent.turns) == 1 else None))

    call(robin, "prompt.submit", text="work through the backlog")

    [continuation] = _replayed(agent, "[Continuing toward your standing goal] keep going")
    assert continuation["author"] is None
    assert continuation["scope"] == ROBIN


def test_a_turn_identity_alone_never_authors_a_row(room):
    """Every turn the gateway starts for itself -- a continuation, an isolated child's turn whose
    frame carries only the scope -- enters with an identity for scoping and no author. Only a caller
    that holds the submitter of that exact text names one."""
    agent, _call, _robin, _sam = room
    session = agent.session

    server._run_prompt_submit("rid", "sid", session, "scoped only", turn_auth_user=ROBIN)
    server._run_prompt_submit("rid", "sid", session, "typed by sam", turn_auth_user=SAM, row_auth_user=SAM)

    assert _replayed(agent, "scoped only")[0]["author"] is None
    assert _replayed(agent, "scoped only")[0]["scope"] == ROBIN
    assert _replayed(agent, "typed by sam")[0]["author"] == AUTHOR_SAM


def test_an_isolated_turn_drained_from_the_queue_is_authored_by_the_envelope_only(room, monkeypatch):
    """A queued prompt on a session whose turns run in the compute-host child is authored on the
    parent, from the envelope's own submitter. An envelope that names nobody is sent with no author,
    whatever identity the frame then resolves for scoping."""
    agent, _call, robin, _sam = room
    session = agent.session
    sent = []
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_submit_prompt_to_compute_host",
                        lambda _rid, _sid, _session, text, **kw: sent.append((text, kw)) or {"result": {}})

    server._enqueue_prompt(session, "from sam", robin, turn_auth_user=SAM)
    assert server._drain_queued_prompt("rid", "sid", session) is True
    session["running"] = False
    server._enqueue_prompt(session, "from nobody", robin)
    assert server._drain_queued_prompt("rid", "sid", session) is True

    authors = {text: (kw.get("display_metadata") or {}).get("author") for text, kw in sent}
    assert authors == {"from sam": AUTHOR_SAM, "from nobody": None}


# ---------------------------------------------------------------------------
# A steer or redirect that lands MID-turn is its sender's row
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("steerers, expected", [
    (["sam"], AUTHOR_SAM),
    (["robin"], AUTHOR_ROBIN),
    (["robin", "sam"], None),
    (["nobody", "robin"], None),
], ids=["colleague", "own", "two_people", "a_connection_naming_no_login"])
def test_a_mid_turn_steer_row_names_exactly_who_sent_it(room, steerers, expected):
    """Under the default ``interrupt`` mode a colleague's mid-turn message lands as a steer row in the
    owner's turn. It carries its sender -- or nobody, when the joined text is not one person's."""
    agent, call, robin, sam = room
    peers = {"robin": robin, "sam": sam, "nobody": _Peer(None)}

    def steer_then_run_a_tool_batch():
        for who in steerers:
            call(peers[who], "session.steer", text=f"note from {who}")
        agent.deliver_mid_turn()

    agent.script = [steer_then_run_a_tool_batch]
    call(robin, "prompt.submit", text="summarise the report")

    [row] = agent.rows
    assert row["display_kind"] == "steer"
    assert (row.get("display_metadata") or {}).get("author") == expected
    assert len(agent.turns) == 1  # delivered mid-turn, nothing handed back


@pytest.mark.parametrize("who, expected", [("sam", AUTHOR_SAM), ("robin", AUTHOR_ROBIN)])
def test_a_mid_turn_redirect_row_names_who_sent_it(room, monkeypatch, who, expected):
    monkeypatch.setattr(server, "_load_busy_input_mode", lambda: "interrupt")
    agent, call, robin, sam = room
    agent._supports_active_turn_redirect = True
    agent._model_request_active.set()
    peer = {"robin": robin, "sam": sam}[who]

    def redirect_then_apply():
        assert call(peer, "prompt.submit", text="stop, use last year")["result"]["status"] == "redirected"
        agent.deliver_mid_turn()

    agent.script = [redirect_then_apply]
    call(robin, "prompt.submit", text="summarise the report")

    [row] = agent.rows
    assert row["content"] == "stop, use last year"
    assert (row.get("display_metadata") or {}).get("author") == expected


# ---------------------------------------------------------------------------
# Real threads: another turn starts and ends inside this turn's clean-up
# ---------------------------------------------------------------------------

def _wait(predicate, what, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


@pytest.fixture()
def live_room(tmp_path, monkeypatch):
    """The shared session with REAL turn threads. A turn releases the session (``running = False``)
    before its clean-up hands a leftover steer on, so a second turn can run to completion in between."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("room", source="desktop")
    peers = {name: _Peer(user) for name, user in
             {"robin": ROBIN, "sam": SAM, "casey": CASEY, "pat": PAT}.items()}
    agent = _Agent("room")
    session = {
        "agent": agent, "attached_images": [], "cols": 80, "cwd": str(tmp_path),
        "history": [], "history_lock": threading.Lock(), "history_version": 0,
        "inflight_turn": None, "running": False, "session_key": "room",
        "show_reasoning": False, "slash_worker": None, "source": "desktop",
        "tool_progress_mode": "all", "transport": peers["robin"],
        "auth_user_id": ROBIN[0], "auth_user_name": ROBIN[1],
    }
    agent.session = session
    monkeypatch.setattr(server, "_db", db, raising=False)
    monkeypatch.setattr(server, "_sessions", {"sid": session}, raising=False)
    monkeypatch.setattr(server, "_emit", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "render_message", lambda *_a: "")
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    for name in ("sam", "casey", "pat"):
        server._attach_session_transport(session, peers[name])

    def call(name, method, **params):
        token = bind_transport(peers[name])
        try:
            return server._methods[method]("rid", {"session_id": "sid", **params})
        finally:
            reset_transport(token)

    yield agent, session, call, monkeypatch
    _wait(lambda: not session.get("running"), "the session to settle")
    db.close()


def test_a_turn_that_runs_inside_anothers_clean_up_cannot_hand_it_a_later_steerer(live_room):
    """Robin's turn T1 ends with Sam's steer X left over. T1 has released the session but not yet
    queued X when Casey's turn T2 starts and finishes; then Pat steers the idle session. Whatever T1
    does next, X is Sam's words and must never be stored under Pat's name."""
    agent, session, call, monkeypatch = live_room
    t1_released, t1_may_continue = threading.Event(), threading.Event()
    settled = []

    def settled_info(_sid, _session, _agent):
        settled.append(threading.current_thread().name)
        if len(settled) == 1:  # T1's clean-up, after running=False and before its post-turn step
            t1_released.set()
            assert t1_may_continue.wait(10)

    monkeypatch.setattr(server, "_emit_settled_session_info", settled_info)
    agent.script = [lambda: call("sam", "session.steer", text="X: use the staging data"), lambda: None]

    assert call("robin", "prompt.submit", text="summarise the report")["result"]["status"] == "streaming"
    assert t1_released.wait(10)

    assert call("casey", "prompt.submit", text="quick question")["result"]["status"] == "streaming"
    _wait(lambda: len(settled) == 2 and not session.get("running"), "T2 to finish inside T1's clean-up")
    assert call("pat", "session.steer", text="Y: and check the totals")["result"]["status"] == "queued"

    t1_may_continue.set()
    _wait(lambda: len(_replayed(agent, "X: use the staging data")) == 1
          and len(_replayed(agent, "Y: and check the totals")) == 1, "both steers to run")

    [x_turn] = _replayed(agent, "X: use the staging data")
    [y_turn] = _replayed(agent, "Y: and check the totals")
    assert x_turn["author"] != AUTHOR_PAT
    assert x_turn["author"] == AUTHOR_SAM
    assert y_turn["author"] == AUTHOR_PAT


# ---------------------------------------------------------------------------
# Retry, regenerate and edit replay a stored row's words: they keep that row's author
# ---------------------------------------------------------------------------

def _stored_exchange(agent, question, author):
    """A stored question and its reply, with the warm history in step with the store."""
    db, session = agent.db, agent.session
    db.append_message("room", "user", "good morning", display_metadata={"author": AUTHOR_ROBIN})
    db.append_message("room", "assistant", "morning")
    db.append_message("room", "user", question, display_metadata={"author": author} if author else None)
    db.append_message("room", "assistant", "an answer")
    session["history"] = db.get_messages_as_conversation("room", include_row_ids=True)


def _stored_authors(agent):
    return [(row["content"], (row.get("display_metadata") or {}).get("author"), row.get("active", 1))
            for row in agent.db.get_messages("room", include_inactive=True) if row["role"] == "user"]


def _retry(call, peer):
    """Press Regenerate as ``peer``, then act on the directive exactly as every client does
    (apps/shared parseCommandDispatch, the app's dispatchSlash): ``send`` / ``skill`` resubmit
    ``message`` from the presser's own connection."""
    directive = call(peer, "command.dispatch", name="retry", arg="")["result"]
    if directive["type"] in ("send", "skill"):
        call(peer, "prompt.submit", text=directive["message"])
    return directive


@pytest.mark.parametrize("original, replayed_by", [(AUTHOR_ROBIN, AUTHOR_SAM), (None, AUTHOR_SAM)],
                         ids=["robins_question", "an_unattributed_question"])
def test_a_retry_is_written_by_its_author_and_run_by_the_presser(room, original, replayed_by):
    """Sam presses Regenerate under Robin's question. The words stay Robin's (or nobody's, as they
    were), the row says Sam replayed them, and the turn ACTS as Sam: Robin sending them once is not
    consent to run them again on somebody else's action. Nothing stores Robin's words as Sam's, and the
    reply is an ``exec`` line -- nothing a client resubmits."""
    agent, call, _robin, sam = room
    _stored_exchange(agent, "what about Q3?", original)

    directive = _retry(call, sam)

    assert all(author != AUTHOR_SAM for _text, author, _active in _stored_authors(agent))
    [retried] = _replayed(agent, "what about Q3?")
    assert (retried["author"], retried["replayed_by"], retried["scope"]) == (original, replayed_by, SAM)
    assert directive["type"] == "exec"
    assert [(text, author) for text, author, active in _stored_authors(agent) if active] == [
        ("good morning", AUTHOR_ROBIN), ("what about Q3?", original)]


def test_retrying_your_own_turn_is_simply_yours(room):
    agent, call, robin, _sam = room
    _stored_exchange(agent, "what about Q3?", AUTHOR_ROBIN)

    _retry(call, robin)

    [retried] = _replayed(agent, "what about Q3?")
    assert (retried["author"], retried["replayed_by"], retried["scope"]) == (AUTHOR_ROBIN, None, ROBIN)


def test_a_retry_that_finds_the_session_busy_keeps_the_split_through_the_queue(room):
    """The retry's own busy check and its submit are two steps; a turn can start in between. The
    queued replay still runs as Sam and is written as Robin's."""
    from tui_gateway.row_author import ReplayedTurn

    agent, call, _robin, sam = room
    session = agent.session
    session["running"] = True
    queued = call(sam, "prompt.submit", text="what about Q3?", _replayed_turn=ReplayedTurn(AUTHOR_ROBIN, SAM))
    assert queued["result"]["status"] == "queued"

    session["running"] = False
    assert server._drain_queued_prompt("rid", "sid", session) is True

    [retried] = _replayed(agent, "what about Q3?")
    assert (retried["author"], retried["replayed_by"], retried["scope"]) == (AUTHOR_ROBIN, AUTHOR_SAM, SAM)


@pytest.mark.parametrize("text, original, expected, replayed_by", [
    ("what about Q3?", AUTHOR_ROBIN, AUTHOR_ROBIN, AUTHOR_SAM),
    ("what about Q3?", AUTHOR_SAM, AUTHOR_SAM, None),
    ("what about Q4?", AUTHOR_ROBIN, None, None),
    ("what about Q4?", AUTHOR_SAM, AUTHOR_SAM, None),
], ids=["regenerate_robins_words", "regenerate_own_words", "sam_edits_robins_row", "sam_edits_own_row"])
def test_a_truncating_resubmit_is_written_by_whoever_wrote_the_words_and_run_by_the_sender(
        room, text, original, expected, replayed_by):
    """The desktop's regenerate and edit resubmit through ``prompt.submit`` with a truncation target.
    The row's own words sent again stay its author's, marked as replayed by the sender; new words over
    one's own row are one's own; new words over somebody else's row cannot be told apart from a
    re-spelled regenerate, so they name nobody. The turn always runs as the sender."""
    agent, call, _robin, sam = room
    _stored_exchange(agent, "what about Q3?", original)
    row_id = next(row["_row_id"] for row in agent.session["history"] if row["content"] == "what about Q3?")

    response = call(sam, "prompt.submit", text=text, truncate_before_row_id=row_id, confirm_truncate=True)

    assert response.get("result", {}).get("status") == "streaming", response
    [turn] = _replayed(agent, text)
    assert (turn["author"], turn["replayed_by"], turn["scope"]) == (expected, replayed_by, SAM)
    active = [(t, a) for t, a, is_active in _stored_authors(agent) if is_active]
    assert active == [("good morning", AUTHOR_ROBIN), (text, expected)]


# ---------------------------------------------------------------------------
# Undo in a shared chat takes back only your own words
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["command", "rpc"])
@pytest.mark.parametrize("original, presser, refused", [
    (AUTHOR_ROBIN, "sam", True), (AUTHOR_SAM, "sam", False), (None, "sam", False),
], ids=["a_colleagues_row", "own_row", "an_unattributed_row"])
def test_undo_takes_back_only_your_own_words(room, method, original, presser, refused):
    """``/undo`` deletes the row and drops its words into the presser's composer, where one tap of Send
    would store them as the presser's. A colleague's row is refused; one's own and an unattributed row
    (a gateway that stamps nobody) behave as before."""
    agent, call, robin, sam = room
    _stored_exchange(agent, "what about Q3?", original)
    peer = {"robin": robin, "sam": sam}[presser]

    if method == "command":
        response = call(peer, "command.dispatch", name="undo", arg="")
    else:
        response = call(peer, "session.undo")

    active = [text for text, _author, is_active in _stored_authors(agent) if is_active]
    if refused:
        assert "only undo your own" in response["error"]["message"]
        assert active == ["good morning", "what about Q3?"]
    else:
        assert "error" not in response, response
        assert active == ["good morning"]
