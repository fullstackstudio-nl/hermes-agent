"""A steer or redirect keeps its sender beside its text, from ``steer()`` to the row it lands as.

The agent joins pending steers into one string and drains it on its own schedule -- after a tool
batch, before the next API call, handed back after the final answer, or put back when no tool result
can take it. A surface that knows who sent the text (the TUI gateway) passes ``author=``; the slot keeps
it with the text under the slot's lock, so every drain returns the pair together and a row is authored
only when every word of its text came from that one sender. Text written into the slot any other way
names nobody.
"""
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.agent_runtime_helpers import _requeue_pending_steer
from agent.interrupt_control import InterruptControlMixin
from agent.turn_finalizer import hand_back_leftover_steer
from agent.turn_iteration_prep import inject_pending_steer, requeue_unapplied_redirect

ROBIN = {"id": "oidc:user-a", "name": "Robin"}
SAM = {"id": "oidc:user-b", "name": "Sam"}


class _Slots(InterruptControlMixin):
    def __init__(self):
        self._pending_steer = None
        self._pending_steer_lock = threading.Lock()
        self._pending_redirect = None
        self._pending_redirect_lock = threading.Lock()
        self._interrupt_requested = False
        self._interrupt_message = None
        self._execution_thread_id = None
        self._model_request_active = threading.Event()
        self._model_request_active.set()


def _tool_turn():
    return [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "tool_call_id": "t1", "content": "ok"}]


@pytest.mark.parametrize("steers, expected", [
    ([("keep it short", SAM)], SAM),
    ([("keep it short", ROBIN), ("cite it", ROBIN)], ROBIN),
    ([("keep it short", ROBIN), ("cite it", SAM)], None),
    ([("keep it short", ROBIN), ("cite it", None)], None),
], ids=["one_sender", "same_sender_twice", "two_senders", "one_unknown"])
def test_every_drain_pairs_the_text_with_its_one_sender_or_nobody(steers, expected):
    # Before the next API call.
    agent = _Slots()
    for text, author in steers:
        agent.steer(text, author=author)
    messages = _tool_turn()
    inject_pending_steer(agent, messages)
    assert messages[-1]["display_kind"] == "steer"
    assert (messages[-1].get("display_metadata") or {}).get("author") == expected

    # Handed back after the final answer.
    agent = _Slots()
    for text, author in steers:
        agent.steer(text, author=author)
    result = {}
    hand_back_leftover_steer(agent, result)
    assert result["pending_steer"] == "\n".join(text for text, _ in steers)
    assert result.get("pending_steer_author") == expected


def test_putting_a_steer_back_keeps_its_sender_and_a_later_one_cannot_take_it_over():
    agent = _Slots()
    agent.steer("keep it short", author=SAM)
    inject_pending_steer(agent, [{"role": "user", "content": "go"}])  # no tool result: put back
    assert agent._pending_steer == "keep it short"
    result = {}
    hand_back_leftover_steer(agent, result)
    assert result["pending_steer_author"] == SAM

    # A steer arriving while the drained text is out of the slot joins it; the pair is then two people's.
    agent.steer("cite it", author=ROBIN)
    _requeue_pending_steer(agent, "keep it short", SAM)
    result = {}
    hand_back_leftover_steer(agent, result)
    assert result["pending_steer"] == "cite it\nkeep it short"
    assert "pending_steer_author" not in result


def test_a_redirect_handed_back_at_the_restart_cap_keeps_its_sender():
    agent = _Slots()
    assert agent.redirect("use last year", author=SAM)
    requeue_unapplied_redirect(agent)
    result = {}
    hand_back_leftover_steer(agent, result)
    assert (result["pending_steer"], result["pending_steer_author"]) == ("use last year", SAM)


def test_text_the_slot_did_not_record_names_nobody():
    """A slot written without ``steer()`` -- or rewritten after -- has no author to give."""
    agent = _Slots()
    agent.steer("keep it short", author=SAM)
    agent._pending_steer = "something else entirely"
    result = {}
    hand_back_leftover_steer(agent, result)
    assert "pending_steer_author" not in result


def test_the_authored_steer_row_is_stored_with_its_author():
    """End to end through a real agent and store: the row the post-batch drain appends is flushed
    with its ``display_metadata.author``."""
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        db = SessionDB(db_path=Path(tmp) / "state.db")
        sid = "20260924_130000_steer"
        db.create_session(sid, "desktop", model="test/model")
        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, session_db=db, session_id=sid,
                        skip_context_files=True, skip_memory=True)
        assert agent.steer("keep it short", author=SAM)
        messages = _tool_turn()
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
        agent._flush_messages_to_session_db(messages)

        rows = [r for r in db.get_messages_as_conversation(sid) if r.get("display_kind") == "steer"]
        assert [(r.get("display_metadata") or {}).get("author") for r in rows] == [SAM]
        db.close()


def _real_agent():
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        return AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                       quiet_mode=True, skip_context_files=True, skip_memory=True)


@pytest.mark.parametrize("stop", [
    lambda agent: agent.clear_interrupt(preserve_redirect=False, hard_cancel=True),
    lambda agent: agent.interrupt(hard_cancel=True),
], ids=["clear_interrupt", "hard_interrupt"])
def test_a_cleared_slot_forgets_its_sender(stop):
    """The real ``clear_interrupt()`` and hard ``interrupt()`` drop the sender with the text. Written
    back later as exactly the same text by any other path, it must not get the old sender back."""
    agent = _real_agent()
    agent._model_request_active.set()
    assert agent.redirect("use last year", author=SAM)
    if not agent._pending_steer:
        agent.steer("keep it short", author=SAM)

    stop(agent)
    assert agent._pending_redirect is None
    agent._pending_redirect = "use last year"
    assert agent._drain_pending_redirect_entry() == ("use last year", None)
    if agent._pending_steer is None:  # only the hard clear drops a pending steer
        agent._pending_steer = "keep it short"
        assert agent._drain_pending_steer_entry() == ("keep it short", None)


def test_an_agent_whose_steer_takes_no_author_gets_the_bare_text():
    """A subclass that overrides ``steer(self, text)`` is detected on the bound method: the gateway
    and ``redirect()`` hand it the text alone instead of raising."""
    from tui_gateway.row_author import deliver_correction

    class _Legacy(_Slots):
        def __init__(self):
            super().__init__()
            self.seen = []

        def steer(self, text):
            self.seen.append(text)
            return True

    agent = _Legacy()
    agent._executing_tools = True  # redirect() degrades to steer() during a tool batch
    assert agent.redirect("use last year", author=SAM)
    assert deliver_correction(agent, "steer", "keep it short", ("oidc:user-b", "Sam"))
    assert agent.seen == ["use last year", "keep it short"]
