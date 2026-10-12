"""Dialogue context management, sliding window memory, and slot tracking."""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from robbie.core.conversation.messages import Message, Role
from robbie.core.conversation.formatter import ConversationFormatter


class ConversationContext:
    """Manages multi-turn conversation state, sliding context window, and working memory."""

    NAME_PATTERNS = [
        re.compile(r"(?:my name is|i am|call me|name's)\s+([A-Za-z]+)", re.IGNORECASE),
        re.compile(r"^([A-Za-z]+)\s+here\.?$", re.IGNORECASE),
    ]

    def __init__(
        self,
        tokenizer: Any,
        max_context_tokens: int = 128,
        reserved_generation_tokens: int = 40,
        formatter: Optional[ConversationFormatter] = None,
        persistent_memory: Optional[Any] = None,
    ):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.reserved_generation_tokens = reserved_generation_tokens
        self.formatter = formatter or ConversationFormatter()
        self.persistent_memory = persistent_memory
        self.messages: List[Message] = []
        self.slots: Dict[str, Any] = {}

        if self.persistent_memory is not None:
            self.slots.update(self.persistent_memory.list_profile())

    @property
    def max_prompt_budget(self) -> int:
        """Maximum tokens allowed for the input prompt."""
        return max(16, self.max_context_tokens - self.reserved_generation_tokens)

    def extract_slots(self, text: str) -> Dict[str, Any]:
        """Extract user attributes (such as user name) from input text."""
        extracted: Dict[str, Any] = {}
        for pat in self.NAME_PATTERNS:
            match = pat.search(text)
            if match:
                candidate = match.group(1).strip()
                # Exclude common false positives
                if candidate.lower() not in ("a", "an", "the", "ready", "good", "fine", "learning", "trying"):
                    extracted["user_name"] = candidate.capitalize()
                    break
        return extracted

    def add_user_message(self, content: str) -> Message:
        """Add a user message and update context slots."""
        cleaned = content.strip()
        msg = Message(role=Role.USER, content=cleaned)
        self.messages.append(msg)

        new_slots = self.extract_slots(cleaned)
        self.slots.update(new_slots)
        if self.persistent_memory is not None and new_slots:
            for k, v in new_slots.items():
                self.persistent_memory.set(k, v)
        return msg

    def add_assistant_message(self, content: str) -> Message:
        """Add an assistant (Robbie) response to the history."""
        cleaned = content.strip()
        msg = Message(role=Role.ASSISTANT, content=cleaned)
        self.messages.append(msg)
        return msg

    def get_context_prompt(self) -> Tuple[str, List[int]]:
        """Format the active conversation into a prompt within token limits.

        Returns:
            Tuple of (prompt_string, token_ids).
        """
        _, prompt, token_ids = self.formatter.fit_to_context_budget(
            messages=self.messages,
            tokenizer=self.tokenizer,
            max_prompt_tokens=self.max_prompt_budget,
            include_system_prompt=False,
        )
        return prompt, token_ids

    def check_direct_slot_answer(self, user_query: str) -> Optional[str]:
        """Check if query is asking for a remembered fact that can be answered accurately."""
        q = user_query.strip().lower()
        if "what is my name" in q or "what's my name" in q or "who am i" in q:
            name = self.slots.get("user_name")
            if name:
                return f"Your name is {name}!"
            return "I don't believe you've told me your name yet! What should I call you?"
        return None

    def clear(self, keep_slots: bool = True) -> None:
        """Clear conversation turn history."""
        self.messages.clear()
        if not keep_slots:
            self.slots.clear()
            if self.persistent_memory is not None:
                self.persistent_memory.clear()

    @property
    def turn_count(self) -> int:
        """Total turns in active session."""
        return len(self.messages)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation context."""
        return {
            "messages": [m.to_dict() for m in self.messages],
            "slots": self.slots,
            "turn_count": self.turn_count,
        }

    def save_session(self, filepath: Union[str, Path]) -> None:
        """Save chat session to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_session(
        cls,
        filepath: Union[str, Path],
        tokenizer: Any,
        max_context_tokens: int = 128,
    ) -> "ConversationContext":
        """Load chat session from JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        ctx = cls(tokenizer=tokenizer, max_context_tokens=max_context_tokens)
        ctx.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        ctx.slots = data.get("slots", {})
        return ctx
