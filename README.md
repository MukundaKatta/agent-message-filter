# agent-message-filter

Filter and slice LLM message history by role, content, or index. Zero dependencies.

## Install

```bash
pip install agent-message-filter
```

## Usage

```python
from agent_message_filter import MessageFilter

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello world"},
    {"role": "assistant", "content": "Hi there"},
    {"role": "user", "content": "Goodbye"},
    {"role": "assistant", "content": "Bye"},
]

# Chainable API
result = (
    MessageFilter(messages)
    .exclude_role("system")
    .filter_by_content("hello")
    .to_list()
)
# [{"role": "user", "content": "Hello world"}]
```

## Standalone functions

```python
from agent_message_filter import filter_by_role, filter_by_content
from agent_message_filter.core import (
    exclude_role, exclude_content,
    keep_last_n, keep_first_n, drop_system,
    only_roles, filter_by_predicate,
    deduplicate, slice_messages,
)

# Role filtering
users = filter_by_role(messages, "user")
no_sys = exclude_role(messages, "system")
no_sys = drop_system(messages)           # convenience alias
ua_only = only_roles(messages, ["user", "assistant"])

# Content filtering
matches = filter_by_content(messages, "hello")                   # substring
regex_m = filter_by_content(messages, r"^Hi", regex=True)        # regex
exact  = filter_by_content(messages, "Hello", case_sensitive=True)

# Slicing
last2  = keep_last_n(messages, 2)
first1 = keep_first_n(messages, 1)
window = slice_messages(messages, 1, 4)

# Custom predicate
long_msgs = filter_by_predicate(messages, lambda m: len(m.get("content","")) > 20)

# Dedup consecutive identical messages
clean = deduplicate(messages)
```

## MessageFilter methods

| Method | Description |
|--------|-------------|
| `.filter_by_role(role)` | Keep messages with this role |
| `.exclude_role(role)` | Remove messages with this role |
| `.filter_by_content(pat, *, case_sensitive, regex)` | Keep by content match |
| `.exclude_content(pat, *, case_sensitive, regex)` | Remove by content match |
| `.keep_last_n(n)` | Keep the last n messages |
| `.keep_first_n(n)` | Keep the first n messages |
| `.drop_system()` | Remove system messages |
| `.only_roles(roles)` | Keep only messages in this role set |
| `.filter_by_predicate(fn)` | Keep where `fn(msg)` is truthy |
| `.deduplicate()` | Remove consecutive identical messages |
| `.slice(start, stop)` | Standard list slice |
| `.to_list()` | Return final list |
| `.count()` | Number of messages after filtering |

Supports multi-modal content (list of blocks) — text is extracted for content matching.

## License

MIT
