"""A user row keeps the author it was written with through compaction.

The gateway stamps ``display_metadata.author`` on the user row a signed-in person submitted, so a
shared chat can say who sent what. Compaction rewrites the stored transcript -- in place
(``archive_and_compact``), by rotating to a child session (``publish_compression_child``), and
``replace_messages`` for a rewrite -- and the carried-forward rows are written afresh each time. A
row that loses its author there reads, after the next reload, as nobody's; one that picks up
another row's author reads as the wrong person. Both are checked against a real compressor and a
real store: every stored user row that names an author names the one its own text was sent with.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROBIN = {"id": "oidc:user-a", "name": "Robin"}
SAM = {"id": "oidc:user-b", "name": "Sam"}


def _transcript(turns=30):
    """Alternating senders, each question unique so a row can be traced back to who sent it."""
    messages, authors = [], {}
    for i in range(turns):
        text = f"question {i} " + "x" * 400
        author = ROBIN if i % 2 == 0 else SAM
        authors[text] = author
        messages.append({"role": "user", "content": text, "display_metadata": {"author": author}})
        messages.append({"role": "assistant", "content": f"answer {i} " + "y" * 400})
    return messages, authors


def _agent(db, session_id, *, in_place):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, session_db=db, session_id=session_id,
            skip_context_files=True, skip_memory=True)
    agent.compression_in_place = in_place
    return agent


def _live_text(row):
    from agent.context_compressor import user_originated_turn_view

    view = user_originated_turn_view(row)
    return view.get("content") if view else None


def _stored_authors(db, session_id, authors):
    """``(text, stored author, author it was sent with)`` for every stored user row that is one of
    the original questions (the summary row is not)."""
    out = []
    for row in db.get_messages_as_conversation(session_id):
        if row.get("role") != "user":
            continue
        text = _live_text(row)
        if text in authors:
            out.append((text, (row.get("display_metadata") or {}).get("author"), authors[text]))
    return out


@pytest.mark.parametrize("in_place", [True, False], ids=["in_place", "rotation"])
def test_compaction_keeps_each_carried_rows_own_author(in_place):
    from agent.conversation_compression import compress_context
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "state.db")
        sid = "20260924_120000_author"
        db.create_session(sid, "desktop", model="test/model")
        messages, authors = _transcript()
        for message in messages:
            db.append_message(session_id=sid, role=message["role"], content=message["content"],
                              display_metadata=message.get("display_metadata"))
        agent = _agent(db, sid, in_place=in_place)
        summary = MagicMock()
        summary.choices[0].message.content = "## Active Task\nkeep going"
        with patch("agent.context_compressor.call_llm", return_value=summary), \
                patch("agent.context_compressor.get_model_context_length", return_value=8000):
            agent.context_compressor.context_length = 8000
            compressed, _system = compress_context(
                agent, [dict(m) for m in messages], approx_tokens=100_000, system_message="sys")

        assert len(compressed) < len(messages), "compaction must actually have dropped turns"
        stored = _stored_authors(db, agent.session_id, authors)
        assert stored, "the carried tail must still hold original questions"
        assert [(text, author) for text, author, _sent in stored] == [
            (text, sent) for text, _author, sent in stored]
        db.close()


def test_replace_messages_keeps_each_rows_own_author():
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmp:
        db = SessionDB(db_path=Path(tmp) / "state.db")
        sid = "20260924_120500_author"
        db.create_session(sid, "desktop", model="test/model")
        messages, authors = _transcript(turns=4)
        db.replace_messages(sid, messages)
        assert [author for _t, author, _s in _stored_authors(db, sid, authors)] == [ROBIN, SAM, ROBIN, SAM]
        # A rewrite that keeps a prefix and archives the rest (rewind / edit) keeps the kept rows' authors.
        db.replace_messages(sid, messages[:4], archive_dropped=True)
        assert [author for _t, author, _s in _stored_authors(db, sid, authors)] == [ROBIN, SAM]
        db.close()
