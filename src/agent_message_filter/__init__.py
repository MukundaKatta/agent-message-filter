"""Filter and slice LLM message history by role, content, or index."""

from __future__ import annotations

from .core import MessageFilter, filter_by_content, filter_by_role

__all__ = [
    "MessageFilter",
    "filter_by_content",
    "filter_by_role",
]
