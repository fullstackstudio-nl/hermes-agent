"""Internal metadata attached to durable conversation messages."""

from __future__ import annotations

from time import time as wall_time
from typing import Any, Mapping, MutableMapping, Optional, TypeVar


# These fields describe Hermes' durable record, not provider-visible message
# content. They must not influence context-pressure decisions.
PERSISTENCE_ONLY_MESSAGE_FIELDS = frozenset({"timestamp"})

_Message = TypeVar("_Message", bound=MutableMapping[str, Any])


def stamp_message_timestamp(
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Attach a creation timestamp without replacing source-provided time.

    Gateway adapters can supply the platform event time; all other callers use
    the local wall clock. Returns the same mapping for use at append sites.
    """
    if message.get("timestamp") is None:
        message["timestamp"] = wall_time() if timestamp is None else timestamp
    return message


def append_message(
    messages: list[Any],
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Stamp and append one live transcript message."""
    messages.append(stamp_message_timestamp(message, timestamp=timestamp))
    return message


def keep_shared_author(joined: MutableMapping[str, Any], other: Any) -> None:
    """After ``other``'s words were joined into ``joined``, drop ``display_metadata["author"]`` from
    ``joined`` unless ``other`` names the very same author.

    A row's author (stamped by a surface that knows who sent the text) names the one person every word
    of the row came from. Joining two user rows for role alternation keeps the first row's metadata, so
    without this the first sender's name would be stored on the second sender's words -- a false
    statement once the joined row is written back. Unknown on either side means nobody. The dict is
    replaced, never mutated, because it may be shared with the row it was copied from."""
    metadata = joined.get("display_metadata")
    if isinstance(metadata, str):
        # A row read back raw carries its metadata as JSON text; judge what it says, not its type.
        import json
        try:
            metadata = json.loads(metadata)
        except ValueError:
            return
    if not isinstance(metadata, dict) or "author" not in metadata:
        return
    other_metadata = other.get("display_metadata") if isinstance(other, Mapping) else None
    if isinstance(other_metadata, dict) and other_metadata.get("author") == metadata["author"]:
        return
    rest = {key: value for key, value in metadata.items() if key != "author"}
    if rest:
        joined["display_metadata"] = rest
    else:
        joined.pop("display_metadata", None)
