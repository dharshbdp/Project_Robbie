"""Verify the clean curriculum, exact inference format, and staged-run control."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from robbie.core.conversation.formatter import ConversationFormatter
from robbie.core.conversation.messages import Message, Role
from robbie.core.conversation.model_chat import ModelChat
from robbie.core.data.chat_data import collate_answers, encode_conversation
from robbie.core.tokenizer.char_tokenizer import CharTokenizer
from robbie.core.training.generation_quality import assess_samples


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tokenizer():
    tok = CharTokenizer()
    tok.train(["User: Robbie: abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ?.!'\n"])
    return tok


def test_history_eos_matches_between_training_and_actual_chat():
    tok = tokenizer()
    turns = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
             {"role": "user", "content": "Why?"}, {"role": "assistant", "content": "Because."}]
    samples = encode_conversation(turns, tok, 256, assistant_eos=True)
    x, y = collate_answers([samples[1]], tok.pad_token_id)

    class Model:
        config = SimpleNamespace(block_size=256)
        prompts = []

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def generate(self, ids, **kwargs):
            self.prompts.append(ids[0].tolist())
            return torch.cat([ids, torch.tensor([tok.encode(" Hello") + [tok.eos_token_id]])], dim=1)

    model = Model()
    chat = ModelChat(model, tok, max_tokens=32, assistant_eos=True)
    chat.reply("Hi")
    chat.reply("Why?")
    start = samples[1]["answer_start"]
    assert model.prompts[-1] == x[0, :start + 1].tolist()
    assert model.prompts[-1].count(tok.eos_token_id) == 1
    assert y[y != -100].tolist() == tok.encode(" Because.") + [tok.eos_token_id]
    assert chat.last_generation["stopped_on_eos"]


def test_legacy_encoding_is_unchanged():
    tok = tokenizer()
    messages = [Message(Role.USER, "Hi"), Message(Role.ASSISTANT, "Hello"), Message(Role.USER, "Why?")]
    formatter = ConversationFormatter()
    assert formatter.encode_dialogue(messages, tok) == tok.encode(formatter.format_dialogue(messages))
    assert tok.eos_token_id not in formatter.encode_dialogue(messages, tok)


def test_curriculum_groups_topics_instead_of_all_hello_starters():
    pytest.importorskip("pyarrow")
    module = script("prepare_clean_chat")
    first = [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello"},
             {"role": "user", "content": "I like drawing."}, {"role": "assistant", "content": "That's nice."}]
    second = first[:2] + [{"role": "user", "content": "I enjoy reading."}, {"role": "assistant", "content": "Me too."}]
    assert module.has_starter(first)
    assert module.conversation_group(first) != module.conversation_group(second)
    assert module.conversation_group(first) == module.conversation_group(first[2:])


def test_filter_normalizes_punctuation_but_does_not_truncate_turns():
    pytest.importorskip("pyarrow")
    module = script("prepare_clean_chat")
    tok = tokenizer()
    turns = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "I\u2019m here."}]
    assert module.normalize_and_filter(turns, tok)[1]["content"] == "I'm here."
    turns[1]["content"] = "word " * 41
    assert module.normalize_and_filter(turns, tok) is None
    turns[1]["content"] = "```code```"
    assert module.normalize_and_filter(turns, tok) is None
    assert module.normalize_and_filter([{"role": "system", "content": "Be brief"}] + turns, tok) is None


def test_nonrepetitive_output_at_token_limit_still_fails_stopping_check():
    report = assess_samples([{"answer": "This is a perfectly ordinary short response.", "stopped_on_eos": False}])
    assert report["failed_samples"] == 1
    assert not report["samples"][0]["obvious_repetition"]
    assert assess_samples([{"answer": "Hello!", "stopped_on_eos": True}])["failed_samples"] == 0


def test_stage_status_resumes_partial_and_preserves_completed_runs(tmp_path):
    module = script("train_clean_chat")
    assert module.stage_status(tmp_path, 8) == "new"
    state = {"training_config": {"epochs": 8}, "epoch": 3}
    torch.save(state, tmp_path / "last.pt")
    assert module.stage_status(tmp_path, 8) == "resume"
    state["epoch"] = 8
    torch.save(state, tmp_path / "last.pt")
    assert module.stage_status(tmp_path, 8) == "complete"
    with pytest.raises(ValueError, match="epochs differ"):
        module.stage_status(tmp_path, 4)


def test_stage_controller_does_not_start_replies_after_partial_language_run(tmp_path, monkeypatch):
    module = script("train_clean_chat")
    data_dir = tmp_path / "data"
    for directory, objective in [("dialogue", "dialogue_lm"), ("answers", "assistant_sft")]:
        path = data_dir / directory
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(json.dumps({"objective": objective, "dialogue_format": "assistant-eos-v1"}))
    checkpoint = tmp_path / "base.pt"
    checkpoint.touch()
    monkeypatch.setattr(module.sys, "argv", ["train_clean_chat.py", "--data-dir", str(data_dir),
                                              "--checkpoint", str(checkpoint), "--output-dir", str(tmp_path / "run")])
    statuses = iter(["new", "resume"])
    monkeypatch.setattr(module, "stage_status", lambda *args: next(statuses))
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    commands = []
    def launch(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(wait=lambda: 0)
    monkeypatch.setattr(module.subprocess, "Popen", launch)
    assert module.main() == 0
    assert len(commands) == 1
    assert str(data_dir / "dialogue") in commands[0]
