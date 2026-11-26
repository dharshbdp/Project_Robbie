"""Live facts must come from the right tool; concepts must reach the model."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import torch

from robbie.core.conversation.engine import ChatEngine
from robbie.core.tokenizer.char_tokenizer import CharTokenizer
from robbie.system.processes import ProcessInfo
from robbie.system.hardware import GPUInfo, MemoryInfo


def engine():
    tok = CharTokenizer()
    tok.train(["User: Robbie: \nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789?."])
    model = MagicMock()
    model.config.block_size = 256
    result = ChatEngine(model, tok, device="cpu")
    result.generate_reply = MagicMock(return_value="A model explanation.")
    return result


@pytest.mark.parametrize("question", [
    "what are the programs running currently?",
    "Which apps are open?",
    "what are the processes running currently?",
    "List running applications.",
])
def test_running_programs_are_not_ram_queries(question):
    chat = engine()
    with patch("robbie.system.ComputerAwareness") as awareness:
        monitor = awareness.return_value
        monitor.process_monitor.list_processes.return_value = [
            ProcessInfo("editor.exe", 10, 500), ProcessInfo("browser.exe", 11, 800)
        ]
        reply = chat.respond(question)
        assert "editor.exe" in reply and "browser.exe" in reply
        monitor.hardware.get_memory_info.assert_not_called()
        chat.generate_reply.assert_not_called()
        assert chat.context.messages[-1].content == reply


@pytest.mark.parametrize("question", [
    "which program consuming the most ram?",
    "Which process uses the most memory?",
    "What app uses the highest RAM?",
    "and what is taking the most ram?",
    "What's using the most memory?",
    "What is consuming the most RAM right now?",
    "Which is hogging the most memory?",
])
def test_largest_process_uses_live_process_measurements(question):
    chat = engine()
    with patch("robbie.system.ComputerAwareness") as awareness:
        monitor = awareness.return_value
        monitor.process_monitor.list_processes.return_value = [
            ProcessInfo("editor.exe", 10, 500), ProcessInfo("browser.exe", 11, 800)
        ]
        reply = chat.respond(question)
        assert "browser.exe" in reply and "800.0 MB" in reply
        assert "editor.exe" not in reply
        monitor.hardware.get_memory_info.assert_not_called()


def test_process_followup_refreshes_measurements():
    chat = engine()
    with patch("robbie.system.ComputerAwareness") as awareness:
        monitor = awareness.return_value
        monitor.process_monitor.list_processes.side_effect = [
            [ProcessInfo("editor.exe", 10, 500), ProcessInfo("browser.exe", 11, 800)],
            [ProcessInfo("editor.exe", 10, 1000), ProcessInfo("browser.exe", 11, 800)],
        ]
        chat.respond("what are the programs running on my pc currently")
        reply = chat.respond("and what is taking the most ram?")
        assert "editor.exe" in reply and "1000.0 MB" in reply
        assert "browser.exe" not in reply
        assert monitor.process_monitor.list_processes.call_count == 2
        monitor.hardware.get_memory_info.assert_not_called()
        chat.generate_reply.assert_not_called()


def test_time_comes_from_clock_even_after_ram_answer():
    chat = engine()
    chat.context.add_user_message("How much RAM is free?")
    chat.context.add_assistant_message("You have 19 GB RAM free.")
    local = datetime(2026, 10, 13, 9, 15, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    with patch("robbie.core.conversation.engine.datetime") as clock:
        clock.now.return_value.astimezone.return_value = local
        reply = chat.respond("whats the time now?")
    assert "09:15:30" in reply and "13 October 2026" in reply and "UTC+05:30" in reply
    assert "RAM" not in reply
    chat.generate_reply.assert_not_called()


@pytest.mark.parametrize("question", [
    "Can a computer still have a GPU if CUDA is not available?",
    "What is VRAM?",
    "Is RAM temporary memory?",
    "Explain the difference between my RAM and VRAM.",
    "What is time complexity?",
])
def test_general_questions_reach_model(question):
    chat = engine()
    with patch("robbie.system.ComputerAwareness") as awareness:
        assert chat.respond(question) == "A model explanation."
        if question == "What is time complexity?":
            chat.generate_reply.assert_called_once_with(question)
        else:
            chat.generate_reply.assert_called_once_with(question, use_history=False)
        awareness.assert_not_called()


def test_vram_inspects_gpu_not_system_ram():
    chat = engine()
    with patch("robbie.system.ComputerAwareness") as awareness:
        monitor = awareness.return_value
        monitor.hardware.get_gpu_info.return_value = GPUInfo(True, "NVIDIA GeForce RTX 4060", 1, 8, 6)
        reply = chat.respond("How much VRAM is free?")
        assert "6.0 GB" in reply and "VRAM" in reply
        assert "NVIDIA NVIDIA" not in reply
        monitor.hardware.get_memory_info.assert_not_called()


def test_ram_request_still_reads_system_memory():
    chat = engine()
    with patch("robbie.system.ComputerAwareness") as awareness:
        awareness.return_value.hardware.get_memory_info.return_value = MemoryInfo(32, 19, 13, 40.6)
        reply = chat.respond("How much RAM is free?")
        assert "19.00 GB" in reply and "32.00 GB" in reply
        chat.generate_reply.assert_not_called()


def test_explanation_followup_retains_history():
    chat = engine()
    chat.context.add_user_message("What is RAM?")
    chat.context.add_assistant_message("RAM is temporary working memory.")
    chat.respond("Explain that more simply.")
    chat.generate_reply.assert_called_once_with("Explain that more simply.", use_history=True)


def test_focused_generation_keeps_session_history():
    chat = engine()
    # Restore the real generator while intercepting only the model invocation.
    chat.generate_reply = ChatEngine.generate_reply.__get__(chat)
    chat.context.add_user_message("Which programs are running?")
    chat.context.add_assistant_message("browser.exe uses 800 MB RAM.")
    prompt_lengths = []

    def generate(input_ids, **kwargs):
        prompt = chat.tokenizer.decode(input_ids[0].tolist(), True)
        prompt_lengths.append(prompt)
        continuation = chat.tokenizer.encode("A GPU can exist without CUDA.")
        return torch.tensor([input_ids[0].tolist() + continuation])

    chat.model.generate.side_effect = generate
    chat.respond("Can a computer still have a GPU if CUDA is not available?")
    assert "browser" not in prompt_lengths[0]
    assert len(chat.context.messages) == 4
    assert chat.context.messages[1].content == "browser.exe uses 800 MB RAM."


def test_web_knowledge_query_classification():
    chat = engine()
    # Positive triggers
    assert chat.is_web_knowledge_query("what are the new games that relesed?") is True
    assert chat.is_web_knowledge_query("what are the new games that released?") is True
    assert chat.is_web_knowledge_query("what are the latest movies coming out?") is True
    assert chat.is_web_knowledge_query("what is the latest version of python?") is True
    assert chat.is_web_knowledge_query("what is the latest news about OpenAI?") is True
    assert chat.is_web_knowledge_query("weather in Tokyo") is True
    assert chat.is_web_knowledge_query("look up latest NVIDIA drivers") is True

    # Negative triggers (belong to model or local system)
    assert chat.is_web_knowledge_query("what is python") is False
    assert chat.is_web_knowledge_query("what is ram") is False
    assert chat.is_web_knowledge_query("whats the time now") is False
    assert chat.is_web_knowledge_query("how much ram is free") is False


def test_new_game_releases_routes_to_web_answer():
    chat = engine()
    with patch("robbie.research.ResearchAssistant") as mock_ra_cls:
        mock_ra = mock_ra_cls.return_value
        mock_ra.answer_query.return_value = (
            "Based on recent gaming release schedules, here are notable releases:\n"
            "• Resident Evil Requiem\n"
            "• Forza Horizon 6\n\n"
            "**Sources:**\n"
            "- [Dexerto 2026 Calendar](https://dexerto.com)"
        )
        reply = chat.respond("what are the new games that relesed?")
        assert "Resident Evil Requiem" in reply
        assert "Forza Horizon 6" in reply
        mock_ra.answer_query.assert_called_once_with("what are the new games that relesed?")
        chat.generate_reply.assert_not_called()
        assert chat.context.messages[-1].content == reply

