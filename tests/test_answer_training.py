"""Regression checks for assistant supervision and computer question routing."""

from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest
import torch

from robbie.core.data.answer_dataset import AnswerDataset, conversations_from_text
from robbie.core.tokenizer.char_tokenizer import CharTokenizer
from robbie.core.conversation.engine import ChatEngine
from robbie.core.training.trainer import Trainer, TrainingConfig
from robbie.core.models.transformer import RobbieTransformer, TransformerConfig


def tokenizer():
    tok = CharTokenizer()
    tok.train(["User: Robbie: System: What is RAM? RAM is working memory. \n"
               "Hello Hi Can you summarize? Short answer. abcdefghijklmnopqrstuvwxyz"])
    return tok


def test_supervise_only_answer_and_eos():
    tok = tokenizer()
    records = [{"conversation": [
        {"role": "user", "content": "What is RAM?"},
        {"role": "assistant", "content": "RAM is working memory."},
    ]}]
    dataset = AnswerDataset(records, tok, seq_len=100)
    x, y = dataset[0]
    answer_ids = tok.encode(" RAM is working memory.") + [tok.eos_token_id]
    assert y[y != -100].tolist() == answer_ids
    prompt_ids = tok.encode("User: What is RAM?\nRobbie:")
    assert x[:len(prompt_ids)].tolist() == prompt_ids
    assert (y[:len(prompt_ids) - 1] == -100).all()
    assert y[len(prompt_ids) - 1] == answer_ids[0]
    assert (y[len(prompt_ids) - 1 + len(answer_ids):] == -100).all()


def test_never_train_partial_answers():
    with pytest.raises(ValueError, match="No complete assistant answers"):
        AnswerDataset([{"conversation": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "a" * 200},
        ]}], tokenizer(), seq_len=50)


def test_multiline_and_followup_are_preserved():
    records = conversations_from_text(
        "User: Hello\nRobbie: Hi\nthere\nUser: Can you summarize?\nRobbie: Short answer."
        "\n<|endoftext|>\nUser: Hi\nRobbie: Hello"
    )
    assert records[0]["conversation"][1]["content"] == "Hi\nthere"
    ds = AnswerDataset(records, tokenizer(), 200)
    assert len(ds) == 3
    assert "User: Hello\nRobbie: Hi\nthere" in tokenizer().decode(ds[1][0].tolist(), True)


def test_concept_question_reaches_model_without_hardware_report():
    engine = ChatEngine(MagicMock(), tokenizer(), device="cpu")
    engine.generate_reply = MagicMock(return_value="RAM is working memory.")
    assert engine.respond("What is RAM?") == "RAM is working memory."
    engine.generate_reply.assert_called_once_with("What is RAM?", use_history=False)


def test_actual_model_forward_and_backward_with_masked_targets():
    tok = tokenizer()
    model = RobbieTransformer(TransformerConfig(
        vocab_size=tok.vocab_size, n_embd=16, n_layer=1, n_head=2, block_size=100
    ))
    dataset = AnswerDataset([{"conversation": [
        {"role": "user", "content": "What is RAM?"},
        {"role": "assistant", "content": "RAM is working memory."},
    ]}], tok, 100)
    x, y = dataset[0]
    _, loss = model(x[None], y[None])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.lm_head.weight.grad.abs().sum() > 0


def test_best_checkpoint_is_not_replaced_by_final_epoch(tmp_path):
    tok = tokenizer()
    model = RobbieTransformer(TransformerConfig(
        vocab_size=tok.vocab_size, n_embd=16, n_layer=1, n_head=2, block_size=32
    ))
    data = [(torch.ones(32, dtype=torch.long), torch.ones(32, dtype=torch.long))]
    trainer = Trainer(model, data, data, TrainingConfig(
        epochs=2, batch_size=1, use_amp=False, device="cpu",
        checkpoint_dir=str(tmp_path), best_model_name="best.pt",
    ))
    trainer.train_epoch = MagicMock(return_value=1.0)
    trainer.evaluate = MagicMock(side_effect=[1.0, 2.0])
    trainer.fit()
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    last = torch.load(tmp_path / "latest_model.pt", weights_only=False)
    assert best["epoch"] == 1
    assert last["epoch"] == 2


def test_windows_battery_sentinels_are_unsigned(monkeypatch):
    from robbie.system import environment

    def status(pointer):
        pointer._obj.ACLineStatus = 1
        pointer._obj.BatteryFlag = 128
        pointer._obj.BatteryLifePercent = 255
        return 1

    monkeypatch.setattr(environment.sys, "platform", "win32")
    monkeypatch.setattr(environment.ctypes, "windll", SimpleNamespace(
        kernel32=SimpleNamespace(GetSystemPowerStatus=status)
    ), raising=False)
    battery = environment.EnvironmentMonitor.get_battery_info()
    assert not battery.present
    assert battery.percent == 100
