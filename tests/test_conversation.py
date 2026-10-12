"""Comprehensive test suite for Phase 5 Conversational Model and Context Management."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import torch

from robbie.core.conversation import (
    Role,
    Message,
    ConversationFormatter,
    ConversationContext,
    ChatEngine,
    ConversationEvaluator,
)
from robbie.core.tokenizer import CharTokenizer


class TestMessages:
    """Test message data structure and role enum."""

    def test_role_normalization(self):
        assert Role.from_str("user") == Role.USER
        assert Role.from_str("HUMAN") == Role.USER
        assert Role.from_str("assistant") == Role.ASSISTANT
        assert Role.from_str("robbie") == Role.ASSISTANT
        assert Role.from_str("system") == Role.SYSTEM

    def test_message_creation_and_dict(self):
        msg = Message(role=Role.USER, content="Hello Robbie! ")
        assert msg.content == "Hello Robbie!"
        assert msg.role == Role.USER

        d = msg.to_dict()
        assert d["role"] == "User"
        assert d["content"] == "Hello Robbie!"

        reconstructed = Message.from_dict(d)
        assert reconstructed.role == Role.USER
        assert reconstructed.content == msg.content


class TestConversationFormatter:
    """Test formatting conversation turns into prompts."""

    def test_format_single_and_multi_turn(self):
        formatter = ConversationFormatter()
        messages = [
            Message(Role.USER, "Hello"),
            Message(Role.ASSISTANT, "Hi there!"),
            Message(Role.USER, "What can you do?"),
        ]

        prompt = formatter.format_dialogue(messages, add_generation_prompt=True)
        assert prompt == "User: Hello\nRobbie: Hi there!\nUser: What can you do?\nRobbie:"

    def test_format_with_system_prompt(self):
        formatter = ConversationFormatter(system_prompt="Be concise.")
        messages = [Message(Role.USER, "Ping")]
        prompt = formatter.format_dialogue(messages, add_generation_prompt=True, include_system_prompt=True)
        assert prompt.startswith("System: Be concise.")
        assert "User: Ping" in prompt
        assert prompt.endswith("Robbie:")

    def test_fit_to_context_budget_sliding_window(self):
        formatter = ConversationFormatter()
        tok = CharTokenizer()
        tok.train(["User: Robbie: Hello Hi What can you do? 1234567890"])

        messages = [
            Message(Role.USER, "Turn 1 question"),
            Message(Role.ASSISTANT, "Turn 1 answer"),
            Message(Role.USER, "Turn 2 question"),
            Message(Role.ASSISTANT, "Turn 2 answer"),
            Message(Role.USER, "Turn 3 latest"),
        ]

        # Restrict budget so only the last turns fit
        pruned, prompt, ids = formatter.fit_to_context_budget(messages, tok, max_prompt_tokens=40)
        assert len(pruned) < len(messages)
        assert pruned[-1].content == "Turn 3 latest"
        assert prompt.endswith("Robbie:")


class TestConversationContext:
    """Test conversation context, slot tracking, and session persistence."""

    def test_slot_extraction(self):
        tok = CharTokenizer()
        tok.train(["User: Robbie: My name is Dharsh"])
        ctx = ConversationContext(tokenizer=tok)

        ctx.add_user_message("Hello, my name is Dharsh!")
        assert ctx.slots.get("user_name") == "Dharsh"

        ctx.add_user_message("I am Alex")
        assert ctx.slots.get("user_name") == "Alex"

    def test_direct_slot_answer(self):
        tok = CharTokenizer()
        tok.train(["User: Robbie:"])
        ctx = ConversationContext(tokenizer=tok)

        # Before knowing name
        reply1 = ctx.check_direct_slot_answer("What is my name?")
        assert "don't believe" in reply1.lower()

        # After learning name
        ctx.add_user_message("My name is Dharsh")
        reply2 = ctx.check_direct_slot_answer("what is my name ?")
        assert reply2 == "Your name is Dharsh!"

    def test_session_save_and_load(self):
        tok = CharTokenizer()
        tok.train(["User: Robbie: testing persistence"])
        ctx = ConversationContext(tokenizer=tok)
        ctx.add_user_message("My name is Dharsh")
        ctx.add_assistant_message("Nice to meet you, Dharsh!")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            save_path = Path(tmp_dir) / "session.json"
            ctx.save_session(save_path)
            assert save_path.exists()

            loaded = ConversationContext.load_session(save_path, tokenizer=tok)
            assert loaded.turn_count == 2
            assert loaded.slots.get("user_name") == "Dharsh"
            assert loaded.messages[0].content == "My name is Dharsh"


class TestChatEngine:
    """Test ChatEngine commands and response generation."""

    def test_slash_commands(self):
        mock_model = MagicMock()
        tok = CharTokenizer()
        tok.train(["User: Robbie:"])
        engine = ChatEngine(model=mock_model, tokenizer=tok, device=torch.device("cpu"))

        # /help
        help_out = engine.respond("/help")
        assert "Available commands" in help_out

        # /memory
        engine.context.slots["user_name"] = "Dharsh"
        mem_out = engine.respond("/memory")
        assert "Dharsh" in mem_out

        # /history
        engine.context.add_user_message("Hello")
        engine.context.add_assistant_message("Hi")
        hist_out = engine.respond("/history")
        assert "[User] Hello" in hist_out
        assert "[Robbie] Hi" in hist_out

        # /clear
        clear_out = engine.respond("/clear")
        assert "cleared" in clear_out.lower()
        assert engine.context.turn_count == 0


class TestConversationEvaluator:
    """Test evaluator metrics."""

    def test_distinct_ngrams(self):
        repetitive = "the the the the the"
        diverse = "the quick brown fox jumps"

        dist1_rep = ConversationEvaluator.distinct_ngrams(repetitive, n=1)
        dist1_div = ConversationEvaluator.distinct_ngrams(diverse, n=1)

        assert dist1_rep < 0.3
        assert dist1_div == 1.0
