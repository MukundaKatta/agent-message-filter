"""Filter and slice LLM message history by role, content, or index.

Messages are plain dicts with at least a ``"role"`` key and a ``"content"``
key.  The content may be a string or a list of content blocks (Anthropic
multi-modal style).

All standalone functions return new lists and never mutate the input.
:class:`MessageFilter` wraps the same functions as chainable methods.

Example::

    from agent_message_filter import MessageFilter

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "bye"},
    ]

    result = (
        MessageFilter(messages)
        .exclude_role("system")
        .keep_last_n(2)
        .to_list()
    )
    # [{"role": "assistant", ...}, {"role": "user", "content": "bye"}]
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

# A message is a dict with at least "role"; "content" is optional (e.g.
# tool_use blocks may omit it).
Message = dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content_as_text(message: Message) -> str:
    """Extract a plain-text representation of a message's content."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "") or block.get("content", "")
                if text:
                    parts.append(str(text))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Standalone filter functions
# ---------------------------------------------------------------------------


def filter_by_role(messages: list[Message], role: str) -> list[Message]:
    """Return only messages whose ``role`` equals *role*."""
    return [m for m in messages if m.get("role") == role]


def exclude_role(messages: list[Message], role: str) -> list[Message]:
    """Return messages excluding those whose ``role`` equals *role*."""
    return [m for m in messages if m.get("role") != role]


def filter_by_content(
    messages: list[Message],
    pattern: str,
    *,
    case_sensitive: bool = False,
    regex: bool = False,
) -> list[Message]:
    """Return messages whose text content matches *pattern*.

    Args:
        messages:       Input message list.
        pattern:        Substring or regex pattern to search for.
        case_sensitive: If ``False`` (default), match is case-insensitive.
        regex:          If ``True``, treat *pattern* as a regex; otherwise
                        use plain substring search.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    results: list[Message] = []
    for m in messages:
        text = _content_as_text(m)
        if regex:
            if re.search(pattern, text, flags):
                results.append(m)
        else:
            if case_sensitive:
                if pattern in text:
                    results.append(m)
            else:
                if pattern.lower() in text.lower():
                    results.append(m)
    return results


def exclude_content(
    messages: list[Message],
    pattern: str,
    *,
    case_sensitive: bool = False,
    regex: bool = False,
) -> list[Message]:
    """Return messages whose text content does NOT match *pattern*."""
    matched = set(
        id(m)
        for m in filter_by_content(
            messages, pattern, case_sensitive=case_sensitive, regex=regex
        )
    )
    return [m for m in messages if id(m) not in matched]


def keep_last_n(messages: list[Message], n: int) -> list[Message]:
    """Return the last *n* messages.  Returns all if ``n >= len(messages)``."""
    if n <= 0:
        return []
    return messages[-n:]


def keep_first_n(messages: list[Message], n: int) -> list[Message]:
    """Return the first *n* messages.  Returns all if ``n >= len(messages)``."""
    if n <= 0:
        return []
    return messages[:n]


def drop_system(messages: list[Message]) -> list[Message]:
    """Convenience alias for ``exclude_role(messages, "system")``."""
    return exclude_role(messages, "system")


def only_roles(messages: list[Message], roles: list[str]) -> list[Message]:
    """Return messages whose role is in *roles*."""
    role_set = set(roles)
    return [m for m in messages if m.get("role") in role_set]


def filter_by_predicate(
    messages: list[Message],
    predicate: Callable[[Message], bool],
) -> list[Message]:
    """Return messages for which *predicate(message)* returns ``True``."""
    return [m for m in messages if predicate(m)]


def deduplicate(messages: list[Message]) -> list[Message]:
    """Remove consecutive duplicate messages (same role + content).

    Non-consecutive duplicates are kept so that conversation structure is
    preserved (e.g. same user question asked twice is intentional).
    """
    result: list[Message] = []
    for m in messages:
        if not result:
            result.append(m)
            continue
        prev = result[-1]
        if prev.get("role") == m.get("role") and _content_as_text(
            prev
        ) == _content_as_text(m):
            continue
        result.append(m)
    return result


def slice_messages(
    messages: list[Message], start: int = 0, stop: int | None = None
) -> list[Message]:
    """Return ``messages[start:stop]``."""
    return messages[start:stop]


# ---------------------------------------------------------------------------
# MessageFilter — chainable wrapper
# ---------------------------------------------------------------------------


class MessageFilter:
    """Chainable message filter.

    Each method returns a new :class:`MessageFilter` (the original is
    unmodified).  Call :meth:`to_list` to get the final list.

    Example::

        result = (
            MessageFilter(messages)
            .exclude_role("system")
            .filter_by_content("hello")
            .keep_last_n(5)
            .to_list()
        )
    """

    def __init__(self, messages: list[Message]) -> None:
        self._messages: list[Message] = list(messages)

    # ------------------------------------------------------------------
    # Chainable methods
    # ------------------------------------------------------------------

    def filter_by_role(self, role: str) -> MessageFilter:
        """Keep only messages with the given role."""
        return MessageFilter(filter_by_role(self._messages, role))

    def exclude_role(self, role: str) -> MessageFilter:
        """Remove messages with the given role."""
        return MessageFilter(exclude_role(self._messages, role))

    def filter_by_content(
        self,
        pattern: str,
        *,
        case_sensitive: bool = False,
        regex: bool = False,
    ) -> MessageFilter:
        """Keep only messages whose content matches *pattern*."""
        return MessageFilter(
            filter_by_content(
                self._messages,
                pattern,
                case_sensitive=case_sensitive,
                regex=regex,
            )
        )

    def exclude_content(
        self,
        pattern: str,
        *,
        case_sensitive: bool = False,
        regex: bool = False,
    ) -> MessageFilter:
        """Remove messages whose content matches *pattern*."""
        return MessageFilter(
            exclude_content(
                self._messages,
                pattern,
                case_sensitive=case_sensitive,
                regex=regex,
            )
        )

    def keep_last_n(self, n: int) -> MessageFilter:
        """Keep only the last *n* messages."""
        return MessageFilter(keep_last_n(self._messages, n))

    def keep_first_n(self, n: int) -> MessageFilter:
        """Keep only the first *n* messages."""
        return MessageFilter(keep_first_n(self._messages, n))

    def drop_system(self) -> MessageFilter:
        """Remove system messages."""
        return MessageFilter(drop_system(self._messages))

    def only_roles(self, roles: list[str]) -> MessageFilter:
        """Keep only messages whose role is in *roles*."""
        return MessageFilter(only_roles(self._messages, roles))

    def filter_by_predicate(
        self, predicate: Callable[[Message], bool]
    ) -> MessageFilter:
        """Keep only messages for which *predicate* returns ``True``."""
        return MessageFilter(filter_by_predicate(self._messages, predicate))

    def deduplicate(self) -> MessageFilter:
        """Remove consecutive duplicate messages."""
        return MessageFilter(deduplicate(self._messages))

    def slice(self, start: int = 0, stop: int | None = None) -> MessageFilter:
        """Slice the message list."""
        return MessageFilter(slice_messages(self._messages, start, stop))

    # ------------------------------------------------------------------
    # Terminal methods
    # ------------------------------------------------------------------

    def to_list(self) -> list[Message]:
        """Return the filtered messages as a plain list."""
        return list(self._messages)

    def count(self) -> int:
        """Return the number of messages after filtering."""
        return len(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"MessageFilter(count={len(self._messages)})"
