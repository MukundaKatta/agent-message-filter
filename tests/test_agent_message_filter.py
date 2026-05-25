"""Tests for agent_message_filter."""

from __future__ import annotations

import pytest

from agent_message_filter import MessageFilter, filter_by_content, filter_by_role
from agent_message_filter.core import (
    deduplicate,
    drop_system,
    exclude_content,
    exclude_role,
    filter_by_predicate,
    keep_first_n,
    keep_last_n,
    only_roles,
    slice_messages,
)

SYS = {"role": "system", "content": "You are helpful."}
USER1 = {"role": "user", "content": "Hello world"}
ASST1 = {"role": "assistant", "content": "Hi there"}
USER2 = {"role": "user", "content": "Goodbye"}
ASST2 = {"role": "assistant", "content": "Bye"}

MESSAGES = [SYS, USER1, ASST1, USER2, ASST2]


# ---------------------------------------------------------------------------
# filter_by_role
# ---------------------------------------------------------------------------


def test_filter_by_role_user():
    result = filter_by_role(MESSAGES, "user")
    assert result == [USER1, USER2]


def test_filter_by_role_assistant():
    result = filter_by_role(MESSAGES, "assistant")
    assert result == [ASST1, ASST2]


def test_filter_by_role_system():
    result = filter_by_role(MESSAGES, "system")
    assert result == [SYS]


def test_filter_by_role_no_match():
    result = filter_by_role(MESSAGES, "tool")
    assert result == []


def test_filter_by_role_empty_input():
    assert filter_by_role([], "user") == []


# ---------------------------------------------------------------------------
# exclude_role
# ---------------------------------------------------------------------------


def test_exclude_role_system():
    result = exclude_role(MESSAGES, "system")
    assert SYS not in result
    assert len(result) == 4


def test_exclude_role_user():
    result = exclude_role(MESSAGES, "user")
    assert USER1 not in result
    assert USER2 not in result


def test_exclude_role_no_match():
    result = exclude_role(MESSAGES, "tool")
    assert result == MESSAGES


# ---------------------------------------------------------------------------
# filter_by_content
# ---------------------------------------------------------------------------


def test_filter_by_content_substring():
    result = filter_by_content(MESSAGES, "hello")
    assert USER1 in result


def test_filter_by_content_case_insensitive():
    result = filter_by_content(MESSAGES, "HELLO")
    assert USER1 in result


def test_filter_by_content_case_sensitive_no_match():
    result = filter_by_content(MESSAGES, "HELLO", case_sensitive=True)
    assert result == []


def test_filter_by_content_case_sensitive_match():
    result = filter_by_content(MESSAGES, "Hello", case_sensitive=True)
    assert USER1 in result


def test_filter_by_content_regex():
    result = filter_by_content(MESSAGES, r"^Hi", regex=True)
    assert ASST1 in result


def test_filter_by_content_regex_no_match():
    result = filter_by_content(MESSAGES, r"^\d+", regex=True)
    assert result == []


def test_filter_by_content_empty():
    result = filter_by_content([], "hello")
    assert result == []


# ---------------------------------------------------------------------------
# exclude_content
# ---------------------------------------------------------------------------


def test_exclude_content():
    result = exclude_content(MESSAGES, "hello")
    assert USER1 not in result
    assert len(result) == 4


def test_exclude_content_no_match():
    result = exclude_content(MESSAGES, "zzz")
    assert result == MESSAGES


# ---------------------------------------------------------------------------
# keep_last_n / keep_first_n
# ---------------------------------------------------------------------------


def test_keep_last_n_basic():
    result = keep_last_n(MESSAGES, 2)
    assert result == [USER2, ASST2]


def test_keep_last_n_more_than_len():
    result = keep_last_n(MESSAGES, 100)
    assert result == MESSAGES


def test_keep_last_n_zero():
    assert keep_last_n(MESSAGES, 0) == []


def test_keep_last_n_negative():
    assert keep_last_n(MESSAGES, -1) == []


def test_keep_first_n_basic():
    result = keep_first_n(MESSAGES, 2)
    assert result == [SYS, USER1]


def test_keep_first_n_more_than_len():
    result = keep_first_n(MESSAGES, 100)
    assert result == MESSAGES


def test_keep_first_n_zero():
    assert keep_first_n(MESSAGES, 0) == []


# ---------------------------------------------------------------------------
# drop_system
# ---------------------------------------------------------------------------


def test_drop_system():
    result = drop_system(MESSAGES)
    assert SYS not in result
    assert len(result) == 4


def test_drop_system_no_system():
    msgs = [USER1, ASST1]
    assert drop_system(msgs) == msgs


# ---------------------------------------------------------------------------
# only_roles
# ---------------------------------------------------------------------------


def test_only_roles():
    result = only_roles(MESSAGES, ["user", "assistant"])
    assert SYS not in result
    assert len(result) == 4


def test_only_roles_empty_set():
    assert only_roles(MESSAGES, []) == []


# ---------------------------------------------------------------------------
# filter_by_predicate
# ---------------------------------------------------------------------------


def test_filter_by_predicate():
    result = filter_by_predicate(
        MESSAGES, lambda m: len(m.get("content", "")) > 15
    )
    assert SYS in result  # "You are helpful." = 17 chars
    assert USER1 not in result  # "Hello world" = 11 chars, not > 15


def test_filter_by_predicate_none():
    result = filter_by_predicate(MESSAGES, lambda m: False)
    assert result == []


def test_filter_by_predicate_all():
    result = filter_by_predicate(MESSAGES, lambda m: True)
    assert result == MESSAGES


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------


def test_deduplicate_removes_consecutive_same():
    dup = [USER1, USER1, ASST1]
    result = deduplicate(dup)
    assert result == [USER1, ASST1]


def test_deduplicate_keeps_non_consecutive():
    msgs = [USER1, ASST1, USER1]
    result = deduplicate(msgs)
    assert result == msgs  # not consecutive


def test_deduplicate_no_dups():
    assert deduplicate(MESSAGES) == MESSAGES


def test_deduplicate_empty():
    assert deduplicate([]) == []


def test_deduplicate_all_same():
    result = deduplicate([USER1, USER1, USER1])
    assert result == [USER1]


# ---------------------------------------------------------------------------
# slice_messages
# ---------------------------------------------------------------------------


def test_slice_messages():
    result = slice_messages(MESSAGES, 1, 3)
    assert result == [USER1, ASST1]


def test_slice_messages_no_stop():
    result = slice_messages(MESSAGES, 2)
    assert result == [ASST1, USER2, ASST2]


def test_slice_messages_default():
    assert slice_messages(MESSAGES) == MESSAGES


# ---------------------------------------------------------------------------
# multi-modal content (list of blocks)
# ---------------------------------------------------------------------------


def test_multimodal_content_filter():
    mm_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image", "source": "..."},
        ],
    }
    result = filter_by_content([mm_msg], "what is")
    assert mm_msg in result


def test_multimodal_content_no_match():
    mm_msg = {
        "role": "user",
        "content": [{"type": "image", "source": "..."}],
    }
    result = filter_by_content([mm_msg], "hello")
    assert result == []


# ---------------------------------------------------------------------------
# MessageFilter chaining
# ---------------------------------------------------------------------------


def test_message_filter_chain():
    result = (
        MessageFilter(MESSAGES)
        .exclude_role("system")
        .keep_last_n(2)
        .to_list()
    )
    assert result == [USER2, ASST2]


def test_message_filter_filter_by_role():
    result = MessageFilter(MESSAGES).filter_by_role("user").to_list()
    assert result == [USER1, USER2]


def test_message_filter_drop_system():
    result = MessageFilter(MESSAGES).drop_system().to_list()
    assert SYS not in result


def test_message_filter_only_roles():
    result = (
        MessageFilter(MESSAGES).only_roles(["user", "assistant"]).to_list()
    )
    assert len(result) == 4


def test_message_filter_content():
    result = (
        MessageFilter(MESSAGES).filter_by_content("hello").to_list()
    )
    assert result == [USER1]


def test_message_filter_exclude_content():
    result = (
        MessageFilter(MESSAGES).exclude_content("hello").to_list()
    )
    assert USER1 not in result


def test_message_filter_predicate():
    result = (
        MessageFilter(MESSAGES)
        .filter_by_predicate(lambda m: m["role"] == "system")
        .to_list()
    )
    assert result == [SYS]


def test_message_filter_deduplicate():
    msgs = [USER1, USER1, ASST1]
    result = MessageFilter(msgs).deduplicate().to_list()
    assert result == [USER1, ASST1]


def test_message_filter_slice():
    result = MessageFilter(MESSAGES).slice(1, 3).to_list()
    assert result == [USER1, ASST1]


def test_message_filter_keep_first_n():
    result = MessageFilter(MESSAGES).keep_first_n(1).to_list()
    assert result == [SYS]


def test_message_filter_count():
    mf = MessageFilter(MESSAGES).filter_by_role("user")
    assert mf.count() == 2
    assert len(mf) == 2


def test_message_filter_repr():
    r = repr(MessageFilter(MESSAGES))
    assert "5" in r


def test_message_filter_does_not_mutate_original():
    original = list(MESSAGES)
    mf = MessageFilter(MESSAGES)
    mf.filter_by_role("user").to_list()
    assert list(MESSAGES) == original


def test_message_filter_chain_independence():
    base = MessageFilter(MESSAGES)
    a = base.filter_by_role("user")
    b = base.filter_by_role("assistant")
    assert a.count() == 2
    assert b.count() == 2


def test_message_filter_to_list_copy():
    mf = MessageFilter(MESSAGES)
    lst = mf.to_list()
    lst.append({"role": "extra", "content": ""})
    assert mf.count() == len(MESSAGES)


# ---------------------------------------------------------------------------
# public API re-exports
# ---------------------------------------------------------------------------


def test_public_filter_by_role():
    assert filter_by_role(MESSAGES, "user") == [USER1, USER2]


def test_public_filter_by_content():
    assert filter_by_content(MESSAGES, "bye") == [USER2, ASST2]


def test_message_filter_regex():
    result = (
        MessageFilter(MESSAGES)
        .filter_by_content(r"^(Hello|Hi)", regex=True)
        .to_list()
    )
    assert USER1 in result
    assert ASST1 in result


def test_exclude_content_regex():
    result = (
        MessageFilter(MESSAGES)
        .exclude_content(r"bye", regex=True, case_sensitive=False)
        .to_list()
    )
    assert USER2 not in result
    assert ASST2 not in result


def test_no_content_key():
    msg = {"role": "tool", "tool_use_id": "abc"}
    result = filter_by_content([msg], "anything")
    assert result == []
    result2 = filter_by_role([msg], "tool")
    assert result2 == [msg]


def test_only_roles_with_system():
    result = only_roles(MESSAGES, ["system"])
    assert result == [SYS]


def test_filter_predicate_on_role_length():
    result = filter_by_predicate(MESSAGES, lambda m: len(m["role"]) > 4)
    # "system"=6, "assistant"=9 match; "user"=4 does not
    roles = {m["role"] for m in result}
    assert "user" not in roles
    assert "system" in roles
    assert "assistant" in roles


def test_keep_last_n_one():
    result = keep_last_n(MESSAGES, 1)
    assert result == [ASST2]


def test_keep_first_n_one():
    result = keep_first_n(MESSAGES, 1)
    assert result == [SYS]


def test_slice_negative_start():
    result = slice_messages(MESSAGES, -2)
    assert result == [USER2, ASST2]


def test_dedup_different_role_same_content():
    msg_a = {"role": "user", "content": "same"}
    msg_b = {"role": "assistant", "content": "same"}
    result = deduplicate([msg_a, msg_b])
    assert result == [msg_a, msg_b]


def test_message_filter_slice_beyond():
    result = MessageFilter(MESSAGES).slice(3, 100).to_list()
    assert result == [USER2, ASST2]


def test_filter_by_content_empty_pattern():
    result = filter_by_content(MESSAGES, "")
    # empty string is in every string
    assert result == MESSAGES


def test_only_roles_multiple():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "tool", "content": "c"},
        {"role": "system", "content": "d"},
    ]
    result = only_roles(msgs, ["user", "tool"])
    assert len(result) == 2
    assert {m["role"] for m in result} == {"user", "tool"}


def test_chained_excludes():
    result = (
        MessageFilter(MESSAGES)
        .exclude_role("system")
        .exclude_role("user")
        .to_list()
    )
    assert all(m["role"] == "assistant" for m in result)


def test_message_filter_error_on_bad_predicate():
    with pytest.raises(Exception):
        MessageFilter(MESSAGES).filter_by_predicate(
            lambda m: m["nonexistent_key"]  # KeyError
        ).to_list()
