"""Protect the data boundaries, accumulation math, and actual resume behavior."""

from copy import deepcopy
from dataclasses import replace
import math

import pytest
import torch

from robbie.core.data.chat_data import (
    collate_answers, encode_conversation, prompt_group, split_for_group,
)
from robbie.core.models.attention import CausalSelfAttention
from robbie.core.models.transformer import RobbieTransformer, TransformerConfig
from robbie.core.tokenizer.char_tokenizer import CharTokenizer
from robbie.core.training.chat_training import ChatTrainingConfig, ChatTrainingRun


def tokenizer():
    tok = CharTokenizer()
    tok.train(["User: Robbie: System: abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ?.!\n"])
    return tok


def conversation(question="Hi", answer="Hello"):
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def test_complete_answers_system_and_dynamic_padding():
    tok = tokenizer()
    messages = [{"role": "system", "content": "Be brief."}] + conversation() + conversation("Why?", "Because.")
    examples = encode_conversation(messages, tok, context=80)
    assert len(examples) == 2
    x, y = collate_answers(examples, tok.pad_token_id)
    for i, sample in enumerate(examples):
        answer = " Hello" if i == 0 else " Because."
        assert y[i][y[i] != -100].tolist() == tok.encode(answer) + [tok.eos_token_id]
        assert tok.decode(x[i].tolist(), skip_special_tokens=True).startswith("System: Be brief.")
    assert (y[0, len(examples[0]["tokens"]) - 1:] == -100).all()
    assert not encode_conversation(conversation(answer="a" * 300), tok, context=80)
    # A system instruction cannot silently disappear just to make an example fit.
    assert not encode_conversation([{"role": "system", "content": "a" * 100}] + conversation(), tok, 80)
    # A short follow-up must not be paired with an answer after throwing away
    # the exchange it refers to. The first answer fits; the follow-up does not.
    long_history = conversation("a" * 25, "b" * 25) + conversation("Why?", "Because.")
    assert len(encode_conversation(long_history, tok, context=85)) == 1


def test_opening_prompt_grouping_keeps_tree_branches_together():
    branch_a = conversation("What is RAM?", "Working memory.")
    branch_b = conversation("WHAT is ram!", "Temporary memory.") + conversation("Why?", "It helps.")
    assert prompt_group(branch_a) == prompt_group(branch_b)
    assert split_for_group(prompt_group(branch_a)) == split_for_group(prompt_group(branch_b))
    groups = [prompt_group(conversation(str(i))) for i in range(10000)]
    splits = {name: {group for group in groups if split_for_group(group) == name}
              for name in ("train", "validation", "test")}
    assert all(splits.values())
    assert not splits["train"] & splits["validation"]
    assert not splits["train"] & splits["test"]


def test_efficient_attention_matches_reference_and_masks_future():
    torch.manual_seed(42)
    attention = CausalSelfAttention(n_embd=16, n_head=2, block_size=8, dropout=0).eval()
    x = torch.randn(2, 7, 16)
    q, k, v = attention.c_attn(x).chunk(3, dim=-1)
    q, k, v = [tensor.reshape(2, 7, 2, 8).transpose(1, 2) for tensor in (q, k, v)]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(8)
    scores = scores.masked_fill(attention.bias[:, :, :7, :7] == 0, -float("inf"))
    reference = (scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 7, 16)
    assert torch.allclose(attention(x), attention.c_proj(reference), atol=1e-6)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 10
    assert torch.allclose(attention(x)[:, :4], attention(changed)[:, :4], atol=1e-6)


def setup_run(path, model=None, config=None):
    tok = tokenizer()
    data = []
    for i in range(7):
        data.extend(encode_conversation(conversation("a" * (i + 1), "b" * (i + 2)), tok, 48))
    model = model or RobbieTransformer(TransformerConfig(
        vocab_size=tok.vocab_size, n_layer=1, n_head=2, n_embd=16, block_size=48, dropout=0.1,
    ))
    config = config or ChatTrainingConfig(epochs=2, batch_size=1, accumulation_steps=2,
                                         device="cpu", amp=False, eval_every=2, save_every=1, log_every=100)
    return ChatTrainingRun(model, {"train": data, "validation": data[:2], "test": data[2:4]},
                           tok, config, path, "test-dataset")


def test_resume_matches_uninterrupted_training_with_dropout(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    prototype = setup_run(tmp_path / "prototype").model
    torch.manual_seed(123)
    full = setup_run(tmp_path / "full", deepcopy(prototype))
    full.fit()
    torch.manual_seed(123)
    partial = setup_run(tmp_path / "resumed", deepcopy(prototype))
    partial.fit(stop_after_steps=3)
    assert partial.cursor == 6
    resumed = setup_run(tmp_path / "resumed", deepcopy(prototype))
    resumed.resume(tmp_path / "resumed/last.pt")
    resumed.fit()
    assert resumed.step == full.step == 8
    assert resumed.seen_tokens == full.seen_tokens
    assert resumed.best_loss == pytest.approx(full.best_loss, abs=1e-8)
    for actual, expected in zip(resumed.model.parameters(), full.model.parameters()):
        assert torch.allclose(actual, expected, atol=1e-8, rtol=1e-7)


def test_accumulation_matches_single_batch_with_unequal_answer_lengths(tmp_path):
    torch.set_num_threads(1)
    prototype = setup_run(tmp_path / "prototype").model
    for module in prototype.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    config = ChatTrainingConfig(epochs=1, batch_size=1, accumulation_steps=2,
                                device="cpu", amp=False, eval_every=100, save_every=100)
    accumulated = setup_run(tmp_path / "accumulated", deepcopy(prototype), config)
    combined = setup_run(tmp_path / "combined", deepcopy(prototype), replace(config, batch_size=2, accumulation_steps=1))
    accumulated.fit(stop_after_steps=1)
    combined.fit(stop_after_steps=1)
    for actual, expected in zip(accumulated.model.parameters(), combined.model.parameters()):
        assert torch.allclose(actual, expected, atol=2e-6, rtol=1e-5)


def test_resume_rejects_different_data(tmp_path):
    run = setup_run(tmp_path)
    run.save()
    run.data_fingerprint = "different-dataset"
    with pytest.raises(ValueError, match="data differs"):
        run.resume(tmp_path / "last.pt")


def test_graceful_stop_saves_only_completed_update(tmp_path):
    run = setup_run(tmp_path)
    run.fit(stop_after_steps=1)
    run.stop_requested = True
    run.fit()
    payload = torch.load(tmp_path / "last.pt", weights_only=False)
    assert payload["step"] == 1
    assert payload["cursor"] == 2
