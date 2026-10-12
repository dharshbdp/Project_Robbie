"""ChatEngine orchestrating multi-turn conversation and model generation."""

from typing import Any, Dict, Optional, Tuple
import torch

from robbie.core.conversation.context import ConversationContext
from robbie.core.conversation.messages import Role


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
        c = cmd.strip().lower()
        if c == "/clear":
            self.context.clear()
            return "Conversation history cleared! Let's start fresh."
        if c == "/memory":
            if not self.context.slots:
                return "Working memory is currently empty."
            items = ", ".join(f"{k}='{v}'" for k, v in self.context.slots.items())
            return f"Remembered facts: {items}"
        if c == "/history":
            if not self.context.messages:
                return "No messages in history yet."
            history_lines = [f"[{m.role.value}] {m.content}" for m in self.context.messages]
            return "\n".join(history_lines)
        if c == "/help":
            return (
                "Available commands:\n"
                "  /clear    — Clear current conversation history\n"
                "  /memory   — Inspect remembered facts (like your name)\n"
                "  /history  — View all conversation turns\n"
                "  exit/quit — End chat session"
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

        # Check for direct slot retrieval (e.g. 'what is my name?')
        slot_answer = self.context.check_direct_slot_answer(cleaned)
        if slot_answer is not None:
            self.context.add_user_message(cleaned)
            self.context.add_assistant_message(slot_answer)
            return slot_answer

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
        if "\nUser:" in response_text:
            response_text = response_text.split("\nUser:")[0].strip()
        if "<|endoftext|>" in response_text:
            response_text = response_text.split("<|endoftext|>")[0].strip()
        if "<isendoftextis>" in response_text:
            response_text = response_text.split("<isendoftextis>")[0].strip()

        # Fallback if empty response generated
        if not response_text:
            response_text = "I am listening! How can I assist you with your project?"

        # Save assistant turn to context
        self.context.add_assistant_message(response_text)
        return response_text
