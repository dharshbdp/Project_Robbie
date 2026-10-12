"""ChatEngine orchestrating multi-turn conversation and model generation."""

from typing import Any, Dict, Optional, Tuple
import torch

from robbie.core.conversation.context import ConversationContext
from robbie.core.conversation.messages import Role
from robbie.core.conversation.personality import RobbiePersonality


class ChatEngine:
    """Orchestrates multi-turn conversation with sliding window context and Robbie transformer."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        context: Optional[ConversationContext] = None,
        max_tokens: int = 60,
        temperature: float = 0.5,
        top_k: int = 25,
        repetition_penalty: float = 1.25,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

        block_size = getattr(getattr(model, "config", None), "block_size", 128)
        self.context = context or ConversationContext(
            tokenizer=tokenizer,
            max_context_tokens=block_size,
            reserved_generation_tokens=max_tokens,
        )

    def handle_command(self, cmd: str) -> Optional[str]:
        """Process special in-chat slash commands."""
        parts = cmd.strip().split()
        if not parts:
            return None
        c = parts[0].lower()

        if c == "/clear":
            self.context.clear()
            return "Conversation history cleared! Let's start fresh."

        if c == "/memory":
            if len(parts) == 1 or parts[1].lower() == "list":
                if not self.context.slots:
                    return "Working memory is currently empty."
                items = "\n".join(f"  • {k} = '{v}'" for k, v in self.context.slots.items())
                return f"Remembered facts:\n{items}"
            if parts[1].lower() == "set" and len(parts) >= 4:
                k = parts[2]
                v = " ".join(parts[3:])
                self.context.slots[k] = v
                if self.context.persistent_memory is not None:
                    self.context.persistent_memory.set(k, v)
                return f"Saved to memory: {k} = '{v}'"
            if parts[1].lower() == "del" and len(parts) >= 3:
                k = parts[2]
                if k in self.context.slots:
                    del self.context.slots[k]
                    if self.context.persistent_memory is not None:
                        self.context.persistent_memory.delete(k)
                    return f"Deleted from memory: '{k}'"
                return f"Fact '{k}' was not found in memory."
            if parts[1].lower() == "clear":
                self.context.slots.clear()
                if self.context.persistent_memory is not None:
                    self.context.persistent_memory.clear()
                return "All memory attributes cleared!"
            return "Usage: /memory [list | set <key> <val> | del <key> | clear]"

        if c == "/history":
            if not self.context.messages:
                return "No messages in history yet."
            history_lines = [f"[{m.role.value}] {m.content}" for m in self.context.messages]
            return "\n".join(history_lines)

        if c == "/research":
            if len(parts) < 2:
                return "Usage: /research <topic or question>"
            query_str = " ".join(parts[1:])
            from robbie.research import ResearchAssistant
            ra = ResearchAssistant()
            report = ra.research(query_str, max_sources=4)
            return report.to_markdown()

        if c == "/search":
            if len(parts) < 2:
                return "Usage: /search <query>"
            query_str = " ".join(parts[1:])
            from robbie.research import ResearchAssistant
            ra = ResearchAssistant()
            results = ra.quick_search(query_str, max_results=5)
            if not results:
                return f"No search results found for '{query_str}'."
            res_lines = [f"Web search results for '{query_str}':"]
            for i, r in enumerate(results, 1):
                snippet_preview = (r.snippet[:120] + "...") if len(r.snippet) > 120 else r.snippet
                res_lines.append(f"  [{i}] {r.title} ({r.source})\n      URL: {r.url}\n      {snippet_preview}")
            return "\n".join(res_lines)

        if c == "/help":
            return (
                "Available commands:\n"
                "  /clear              — Clear current conversation history\n"
                "  /memory             — Inspect remembered facts (like your name)\n"
                "  /memory set <k> <v> — Store a custom fact in memory\n"
                "  /history            — View all conversation turns\n"
                "  /research <topic>   — Conduct multi-source web research with citations\n"
                "  /search <query>     — Quick web search across sources\n"
                "  exit/quit           — End chat session"
            )
        return None

    def respond(self, user_input: str) -> str:
        """Process a user message and generate Robbie's reply."""
        cleaned = user_input.strip().strip("\"' ")
        if not cleaned:
            return ""

        # Check for commands
        cmd_result = self.handle_command(cleaned)
        if cmd_result is not None:
            return cmd_result

        user_name = self.context.slots.get("user_name")

        # Check if user is introducing their name (e.g. "My name is Dharsh", "I am Dharsh")
        intro_slots = self.context.extract_slots(cleaned)
        intro_name = intro_slots.get("user_name")
        lower_cleaned = cleaned.lower()
        is_self_intro = intro_name and (
            lower_cleaned.startswith("my name is")
            or lower_cleaned.startswith("i am ")
            or lower_cleaned.startswith("call me ")
            or lower_cleaned.startswith("name's ")
            or "my name is " in lower_cleaned
        )
        if is_self_intro:
            self.context.add_user_message(cleaned)
            intro_reply = f"Nice to meet you, {intro_name}! I'll remember that. How can I help you today?"
            self.context.add_assistant_message(intro_reply)
            return intro_reply

        # Check for character affirmations & compliments (Phase 6)
        affirmation_reply = RobbiePersonality.get_affirmation_response(cleaned, user_name=user_name)
        if affirmation_reply is not None:
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(affirmation_reply)
            return affirmation_reply

        # Check for direct slot retrieval (e.g. 'what is my name?')
        slot_answer = self.context.check_direct_slot_answer(cleaned)
        if slot_answer is not None:
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(slot_answer)
            return slot_answer

        # If user gives a warm direct greeting and name is already known, personalize greeting
        if cleaned.lower() in ("hello", "hi", "hey", "hello robbie", "hey robbie") and user_name:
            greeting_reply = RobbiePersonality.get_personalized_greeting(user_name)
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(greeting_reply)
            return greeting_reply

        # Check for natural language research capability questions
        research_capability_triggers = (
            "can you research on the internet",
            "can you research the internet",
            "can you research online",
            "can you search the web",
            "can you search online",
            "can you search the internet",
            "can you browse the web",
            "do you have internet access",
            "can you look up things online",
            "can you search for things",
        )
        if any(trig in lower_cleaned for trig in research_capability_triggers):
            reply = (
                "Yes! I have a built-in research assistant. I can search the web and Wikipedia, "
                "read articles, and synthesize findings with citations.\n"
                "You can ask me to research any topic by typing `/research <topic>` or `/search <query>`, "
                "or simply say 'research <topic>'!"
            )
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(reply)
            return reply

        # Check for direct natural language research triggers
        if lower_cleaned.startswith("research ") or lower_cleaned.startswith("research on "):
            topic = cleaned[8:].strip()
            if topic.lower().startswith("on "):
                topic = topic[3:].strip()
            if topic:
                self.context.add_user_message(cleaned)
                from robbie.research import ResearchAssistant
                ra = ResearchAssistant()
                report = ra.research(topic, max_sources=4)
                report_md = report.to_markdown()
                self.context.add_assistant_message(report_md)
                return report_md

        if lower_cleaned.startswith("search the web for ") or lower_cleaned.startswith("search for "):
            query_text = cleaned.split("for ", 1)[-1].strip()
            if query_text:
                self.context.add_user_message(cleaned)
                from robbie.research import ResearchAssistant
                ra = ResearchAssistant()
                report = ra.research(query_text, max_sources=4)
                report_md = report.to_markdown()
                self.context.add_assistant_message(report_md)
                return report_md

        # Add user turn to context
        self.context.add_user_message(cleaned)

        # Build token-budgeted prompt
        prompt, prompt_ids = self.context.get_context_prompt()
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # Generate response from model
        generated_tensor = self.model.generate(
            input_tensor,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            stop_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=self.repetition_penalty,
        )

        gen_ids = generated_tensor[0].tolist()[len(prompt_ids):]
        response_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Stop token string post-processing
        if response_text.startswith("Robbie:"):
            response_text = response_text[len("Robbie:"):].strip()
        if response_text.startswith("Assistant:"):
            response_text = response_text[len("Assistant:"):].strip()

        stop_markers = [
            "\nUser:",
            "\nRobbie:",
            "\nAssistant:",
            "\nSystem:",
            "User:",
            "Robbie:",
            "Assistant:",
            "<|endoftext|>",
            "<isendoftextis>",
            "<|im_end|>",
        ]
        for marker in stop_markers:
            if marker in response_text:
                response_text = response_text.split(marker)[0].strip()

        # Clean trailing incomplete boundary tokens or tags
        response_text = response_text.rstrip("<\n ").strip()

        # Fallback if empty response generated
        if not response_text:
            response_text = "I am listening! How can I assist you with your project?"

        # Save assistant turn to context
        self.context.add_assistant_message(response_text)
        return response_text
