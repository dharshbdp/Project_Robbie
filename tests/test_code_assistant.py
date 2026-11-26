"""Unit tests for Python Code Snippet Assistant (Phase 10 Code Space)."""

import pytest
import re
from robbie.core.conversation.code_assistant import PythonCodeAssistant


def test_is_code_request_detection():
    queries = [
        "write a python snippet",
        "write a python script",
        "give me a python snippet",
        "show me a python code snippet",
        "write hello world in python",
        "how to read a file in python",
        "python code to reverse a string",
        "how to parse json in python",
        "write a python function to sort a list",
        "can you write a python snippet?",
    ]
    for q in queries:
        assert PythonCodeAssistant.is_code_request(q) is True, f"Failed to detect: {q}"


def test_non_code_requests_rejected():
    non_code = [
        "what is the time?",
        "/system",
        "/history",
        "what is RAM?",
        "how are you today?",
        "what are the new games that released?",
    ]
    for q in non_code:
        assert PythonCodeAssistant.is_code_request(q) is False, f"False positive: {q}"


def test_hello_world_generation():
    snippet = PythonCodeAssistant.generate_snippet("write hello world in python")
    assert "```python" in snippet
    assert "Hello, World!" in snippet
    assert "Send to Editor" in snippet


def test_file_read_generation():
    snippet = PythonCodeAssistant.generate_snippet("how to read a file in python")
    assert "```python" in snippet
    assert "read_file_contents" in snippet
    assert "pathlib" in snippet


def test_reverse_string_generation():
    snippet = PythonCodeAssistant.generate_snippet("python code to reverse a string")
    assert "```python" in snippet
    assert "[::-1]" in snippet


def test_general_snippet_generation():
    snippet = PythonCodeAssistant.generate_snippet("write a python snippet")
    assert "```python" in snippet
    assert "TaskTracker" in snippet
    assert "Send to Editor" in snippet
