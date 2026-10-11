"""Tests for model inspector and parameter summary tools."""

from robbie.core.models.tiny_llm import TinyLLM, TinyLLMConfig
from robbie.core.models.inspector import get_model_parameter_summary


def test_model_parameter_summary():
    config = TinyLLMConfig(vocab_size=64, block_size=16, n_embd=32, n_hidden=64)
    model = TinyLLM(config)

    summary = get_model_parameter_summary(model)
    assert "layers" in summary
    assert summary["total_parameters"] > 0
    assert summary["trainable_parameters"] == summary["total_parameters"]
    assert any(layer["name"] == "fc1.weight" for layer in summary["layers"])
