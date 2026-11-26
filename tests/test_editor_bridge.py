"""Verify the real editor protocol, checkpoint choice, and shared session state."""

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bridge(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("editor_bridge_test", ROOT / "editor/bridge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("ROBBIE_CHECKPOINT", raising=False)
    monkeypatch.setenv("ROBBIE_MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setenv("ROBBIE_CHAT_HISTORY_PATH", str(tmp_path / "chat_history.json"))
    return module


def test_latest_best_checkpoint_and_explicit_override(bridge, monkeypatch, tmp_path):
    older = tmp_path / "answers_first/best_answers.pt"
    newer = tmp_path / "answers_second/best_answers.pt"
    for path, timestamp in ((older, 100), (newer, 200)):
        path.parent.mkdir()
        path.touch()
        os.utime(path, (timestamp, timestamp))
    assert bridge.select_checkpoint(tmp_path) == newer.resolve()
    chatbot = tmp_path / "chat_run/best_chat.pt"
    chatbot.parent.mkdir()
    chatbot.touch()
    os.utime(chatbot, (300, 300))
    assert bridge.select_checkpoint(tmp_path) == chatbot.resolve()
    monkeypatch.setenv("ROBBIE_CHECKPOINT", str(older))
    assert bridge.select_checkpoint(tmp_path) == older.resolve()
    monkeypatch.setenv("ROBBIE_CHECKPOINT", str(tmp_path / "missing.pt"))
    with pytest.raises(FileNotFoundError, match="Configured"):
        bridge.select_checkpoint(tmp_path)


def test_missing_checkpoint_does_not_create_random_model(bridge, tmp_path):
    with pytest.raises(FileNotFoundError, match="No trained"):
        bridge.select_checkpoint(tmp_path)


def test_protocol_keeps_logs_out_of_json_and_recovers(bridge, monkeypatch, capsys):
    def noisy(payload):
        print("Diagnostic log, not a protocol response")
        return {"reply": payload["message"]}
    monkeypatch.setitem(bridge.HANDLERS, "test", noisy)
    monkeypatch.setattr(sys, "stdin", io.StringIO(
        json.dumps({"id": 1, "command": "test", "payload": {"message": "hello"}}) + "\n"
        + "invalid json\n"
        + json.dumps({"id": 3, "command": "test", "payload": {"message": "again"}}) + "\n"
    ))
    bridge.serve()
    captured = capsys.readouterr()
    responses = [json.loads(line) for line in captured.out.splitlines()]
    assert len(responses) == 3
    assert responses[0] == {"id": 1, "result": {"reply": "hello"}}
    assert "error" in responses[1]["result"]
    assert responses[2]["result"]["reply"] == "again"
    assert "Diagnostic log" in captured.err


def test_editor_memory_and_chat_context_stay_in_sync(bridge):
    from types import SimpleNamespace
    from robbie.core.conversation.context import ConversationContext
    from robbie.core.tokenizer.char_tokenizer import CharTokenizer
    tok = CharTokenizer()
    tok.train(["User: Robbie: \nhello"])
    context = ConversationContext(tok, persistent_memory=bridge.get_memory())
    bridge._cached_engine = SimpleNamespace(context=context)
    bridge.handle_memory({"action": "set", "key": "preferred_editor", "value": "Robbie"})
    assert context.slots["preferred_editor"] == "Robbie"
    context.add_user_message("hello")
    bridge.handle_reset({})
    assert context.turn_count == 0
    assert context.slots["preferred_editor"] == "Robbie"
    bridge.handle_memory({"action": "clear"})
    assert "preferred_editor" not in context.slots


def test_scratchpad_exit_does_not_stop_bridge(bridge):
    result = bridge.handle_execute({"code": "print('scratchpad output'); raise SystemExit(2)"})
    assert not result["success"]
    assert "scratchpad output" in result["stdout"]
    assert bridge.handle_reset({})["turns"] == 0


def test_real_worker_keeps_model_and_conversation_between_requests(tmp_path):
    # A fresh memory file keeps this integration test away from user preferences.
    environment = {
        **os.environ,
        "ROBBIE_MEMORY_PATH": str(tmp_path / "worker-memory.json"),
        "ROBBIE_CHAT_HISTORY_PATH": str(tmp_path / "worker-history.json"),
        "PYTHONIOENCODING": "utf-8",
    }
    requests = [
        ("status", {}), ("chat", {"message": "What is RAM?"}),
        ("chat", {"message": "whats the time now?"}), ("reset", {}), ("status", {}),
    ]
    payload = "".join(json.dumps({"id": i, "command": command, "payload": data}) + "\n"
                      for i, (command, data) in enumerate(requests))
    result = subprocess.run([sys.executable, str(ROOT / "editor/bridge.py"), "--serve"],
                            input=payload, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=90, cwd=tmp_path, env=environment)
    assert result.returncode == 0, result.stderr
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(responses) == len(requests), result.stdout
    status = responses[0]["result"]
    assert status["loaded"] and status["checkpoint"].endswith(".pt")
    assert responses[1]["result"]["turns"] == 2
    assert responses[2]["result"]["turns"] == 4
    assert "UTC" in responses[2]["result"]["reply"]
    assert responses[3]["result"]["turns"] == 0
    assert responses[4]["result"]["checkpoint"] == status["checkpoint"]


def test_bridge_chat_history_persistence_and_deletion(bridge, tmp_path):
    history_file = tmp_path / "custom_chat_history.json"
    from robbie.core.conversation.context import ConversationContext
    from robbie.core.tokenizer.char_tokenizer import CharTokenizer
    tok = CharTokenizer()
    tok.train(["User: Robbie: testing chat persistence"])
    ctx = ConversationContext(tok, history_file=history_file)
    ctx.add_user_message("Hello from user")
    ctx.add_assistant_message("Hello from Robbie")

    assert history_file.is_file()
    with open(history_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["turn_count"] == 2
    assert data["messages"][0]["content"] == "Hello from user"

    # Test rehydration across sessions
    ctx2 = ConversationContext(tok, history_file=history_file, auto_load=True)
    assert ctx2.turn_count == 2
    assert ctx2.messages[0].content == "Hello from user"

    # Test deletion
    assert ctx2.delete_history() is True
    assert not history_file.exists()
    assert ctx2.turn_count == 0

