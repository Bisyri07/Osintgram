import json

from src.web.llm_client import (
    LLMFunctionCall,
    LLMToolCall,
    _to_ollama_messages,
)


def test_tool_calls_are_flattened_to_plain_dicts():
    """Regression: ollama's pydantic `Message` rejects our `LLMToolCall`
    dataclasses ("Input should be a valid dictionary or instance of
    ToolCall"), so assistant tool calls must be sent as plain dicts."""
    call = LLMToolCall(
        id="call_get_hashtags",
        function=LLMFunctionCall(name="get_hashtags", arguments={"limit_posts": "31"}),
    )
    messages = [
        {"role": "user", "content": "hashtags?"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
    ]

    result = _to_ollama_messages(messages)
    sent = result[1]["tool_calls"]

    assert isinstance(sent[0], dict)
    assert sent == [{"function": {"name": "get_hashtags", "arguments": {"limit_posts": "31"}}}]
    # Only the single `function` field ollama's ToolCall declares is emitted.
    assert set(sent[0]) == {"function"}
    # The payload is sent over HTTP, so it must survive JSON round-tripping.
    assert json.loads(json.dumps(sent))[0]["function"]["name"] == "get_hashtags"


def test_messages_without_tool_calls_pass_through_untouched():
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hello"},
    ]

    result = _to_ollama_messages(messages)

    assert result[0] is messages[0]
    assert result[1] is messages[1]


def test_dict_tool_calls_are_normalized_too():
    messages = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "name": "get_profile", "arguments": {"username": "x"}}]},
    ]

    result = _to_ollama_messages(messages)

    assert result[0]["tool_calls"] == [
        {"function": {"name": "get_profile", "arguments": {"username": "x"}}}
    ]


def test_missing_arguments_fall_back_to_empty_dict():
    call = LLMToolCall(id="c1", function=LLMFunctionCall(name="get_info", arguments={}))
    messages = [{"role": "assistant", "content": "", "tool_calls": [call]}]

    result = _to_ollama_messages(messages)

    assert result[0]["tool_calls"][0]["function"] == {"name": "get_info", "arguments": {}}
