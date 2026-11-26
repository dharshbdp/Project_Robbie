"""Dialogue context management, sliding window memory, session context, and persistent storage tracking."""

import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from robbie.core.conversation.messages import Message, Role
from robbie.core.conversation.formatter import ConversationFormatter


class SlotsView(dict):
    """Dictionary view that synchronizes mutations to session context or persistent storage."""

    def __init__(self, context: "ConversationContext"):
        super().__init__()
        self._context = context
        self.sync()

    def sync(self) -> None:
        super().clear()
        super().update(self._context.session_slots)
        super().update(self._context.persistent_slots)

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        if key in ("session_start_time", "current_topic"):
            self._context.session_slots[key] = value
        else:
            self._context.persistent_slots[key] = value
            if self._context.persistent_memory is not None:
                self._context.persistent_memory.set(key, value)

    def __delitem__(self, key: str) -> None:
        if key in self:
            super().__delitem__(key)
        self._context.session_slots.pop(key, None)
        if key in self._context.persistent_slots:
            del self._context.persistent_slots[key]
            if self._context.persistent_memory is not None:
                self._context.persistent_memory.delete(key)

    def clear(self) -> None:
        super().clear()
        self._context.session_slots.clear()
        self._context.persistent_slots.clear()
        if self._context.persistent_memory is not None:
            self._context.persistent_memory.clear()


class ConversationContext:
    """Manages multi-turn conversation state, sliding context window, and clearly separates

    transient session context (working memory) from persistent storage (disk-backed long-term memory).
    """

    NAME_PATTERNS = [
        re.compile(r"(?:my name is|i am|call me|name's)\s+([A-Za-z]+)", re.IGNORECASE),
        re.compile(r"^([A-Za-z]+)\s+here\.?$", re.IGNORECASE),
    ]

    ROLE_PATTERNS = [
        re.compile(r"(?:i am a|i work as a|my role is)\s+([A-Za-z\s]+?)(?:\.|!|,|$)", re.IGNORECASE),
    ]

    FAVORITE_PATTERNS = [
        re.compile(r"my favorite\s+([A-Za-z0-9_]+)\s+is\s+([A-Za-z0-9_\s]+?)(?:\.|!|,|$)", re.IGNORECASE),
    ]

    TOPIC_PATTERNS = [
        re.compile(r"(?:let's talk about|tell me about|what about|regarding)\s+([A-Za-z0-9_\s]+?)(?:\.|\?|!|$)", re.IGNORECASE),
    ]

    def __init__(
        self,
        tokenizer: Any,
        max_context_tokens: int = 128,
        reserved_generation_tokens: int = 40,
        formatter: Optional[ConversationFormatter] = None,
        persistent_memory: Optional[Any] = None,
        history_file: Optional[Union[str, Path]] = None,
        auto_save: bool = True,
        auto_load: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.reserved_generation_tokens = reserved_generation_tokens
        self.formatter = formatter or ConversationFormatter()
        self.persistent_memory = persistent_memory
        self.auto_save = auto_save
        self.messages: List[Message] = []

        import os
        env_hist = os.environ.get("ROBBIE_CHAT_HISTORY_PATH")
        chosen_hist = history_file or env_hist
        self.history_file: Optional[Path] = Path(chosen_hist) if chosen_hist else None

        # Explicit separation:
        # 1. Ephemeral session context (working memory, clears per session)
        self.session_slots: Dict[str, Any] = {
            "session_start_time": time.time(),
            "current_topic": None,
        }

        # 2. Durable persistent storage (profile, preferences, synced to disk)
        self.persistent_slots: Dict[str, Any] = {}
        if self.persistent_memory is not None:
            self.persistent_slots.update(self.persistent_memory.list_profile())

        # Live synchronized view
        self.slots = SlotsView(self)

        # Auto-load existing conversation history from root chat_history.json if present
        if auto_load and self.history_file is not None and self.history_file.is_file():
            self.load_history()

    @property
    def max_prompt_budget(self) -> int:
        """Maximum tokens allowed for the input prompt."""
        return max(16, self.max_context_tokens - self.reserved_generation_tokens)

    def extract_slots(self, text: str) -> Dict[str, Any]:
        """Extract user attributes (name, role, preferences) from input text."""
        extracted: Dict[str, Any] = {}

        # 1. Name extraction
        for pat in self.NAME_PATTERNS:
            match = pat.search(text)
            if match:
                candidate = match.group(1).strip()
                if candidate.lower() not in ("a", "an", "the", "ready", "good", "fine", "learning", "trying", "developer", "engineer"):
                    extracted["user_name"] = candidate.capitalize()
                    break

        # 2. Role extraction
        for pat in self.ROLE_PATTERNS:
            match = pat.search(text)
            if match:
                role_cand = match.group(1).strip()
                if len(role_cand) >= 3 and role_cand.lower() not in ("human", "person", "user"):
                    extracted["user_role"] = role_cand.capitalize()
                    break

        # 3. Favorite / Preference extraction
        for pat in self.FAVORITE_PATTERNS:
            match = pat.search(text)
            if match:
                cat = match.group(1).strip().lower()
                val = match.group(2).strip()
                extracted[f"favorite_{cat}"] = val
                break

        # 4. Session topic extraction
        for pat in self.TOPIC_PATTERNS:
            match = pat.search(text)
            if match:
                topic = match.group(1).strip()
                if len(topic) >= 2:
                    extracted["_session_topic"] = topic
                    break

        return extracted

    def add_user_message(self, content: str) -> Message:
        """Add a user message and update context slots with explicit session vs persistent separation."""
        cleaned = content.strip()
        msg = Message(role=Role.USER, content=cleaned)
        self.messages.append(msg)

        new_slots = self.extract_slots(cleaned)

        # Handle ephemeral session topic
        if "_session_topic" in new_slots:
            self.session_slots["current_topic"] = new_slots.pop("_session_topic")

        # Handle persistent attributes
        if new_slots:
            self.persistent_slots.update(new_slots)
            if self.persistent_memory is not None:
                for k, v in new_slots.items():
                    self.persistent_memory.set(k, v)

        self.slots.sync()
        self.save_history()
        return msg

    def add_assistant_message(self, content: str) -> Message:
        """Add an assistant (Robbie) response to the history."""
        cleaned = content.strip()
        msg = Message(role=Role.ASSISTANT, content=cleaned)
        self.messages.append(msg)
        self.save_history()
        return msg

    def get_context_prompt(self, assistant_eos: bool = True) -> Tuple[str, List[int]]:
        """Format the active conversation into a prompt within token limits.

        Strips raw source URL listings and footnotes from past assistant messages so
        multi-turn conversational context fits cleanly within the token budget.
        """
        condensed_messages = []
        for msg in self.messages:
            if msg.role == Role.ASSISTANT:
                content = msg.content
                # Strip URL sources section for prompt encoding
                content = re.sub(r"\n+\s*(?:\*\*)?Sources(?:\*\*)?:.*$", "", content, flags=re.DOTALL)
                content = re.sub(r"https?://\S+", "", content).strip()
                condensed_messages.append(Message(role=Role.ASSISTANT, content=content))
            else:
                condensed_messages.append(msg)

        _, prompt, token_ids = self.formatter.fit_to_context_budget(
            messages=condensed_messages,
            tokenizer=self.tokenizer,
            max_prompt_tokens=self.max_prompt_budget,
            include_system_prompt=False,
            assistant_eos=assistant_eos,
        )
        return prompt, token_ids

    def get_memory_breakdown(self) -> Dict[str, Any]:
        """Return a structured breakdown distinguishing session context from persistent storage."""
        return {
            "session_context": {
                "turn_count": len(self.messages),
                "current_topic": self.session_slots.get("current_topic"),
                "session_slots": dict(self.session_slots),
            },
            "persistent_storage": {
                "profile": dict(self.persistent_slots),
                "notes_count": len(getattr(self.persistent_memory, "notes", {})),
                "episodes_count": len(getattr(self.persistent_memory, "episodes", [])),
                "store_path": str(getattr(self.persistent_memory, "store_path", "none")),
            },
        }

    def check_direct_slot_answer(self, user_query: str) -> Optional[str]:
        """Check if query is asking for a remembered fact that can be answered accurately."""
        q = user_query.strip().lower().rstrip("?!., ")

        # 1. Name query
        name_inquiry_patterns = [
            "what is my name",
            "what's my name",
            "whats my name",
            "what my name is",
            "who am i",
            "do you remember me",
            "remember me",
            "remember my name",
            "know my name",
            "know who i am",
            "tell me my name",
            "say my name",
        ]
        is_name_query = any(pat in q for pat in name_inquiry_patterns) or (
            "my name" in q and any(w in q for w in ("what", "remember", "know", "recall", "tell", "forget"))
        )
        if is_name_query:
            name = self.persistent_slots.get("user_name")
            if name:
                return f"Your name is {name}!"
            return "I don't believe you've told me your name yet! What should I call you?"

        # 2. Role query
        if any(pat in q for pat in ("what is my role", "what do i do", "what's my job", "whats my job")):
            role = self.persistent_slots.get("user_role")
            if role:
                return f"You're a {role}!"
            return "You haven't told me your role or profession yet!"

        # 3. Current session topic query
        if any(pat in q for pat in ("what are we talking about", "what is our topic", "current topic")):
            topic = self.session_slots.get("current_topic")
            if topic:
                return f"In this session, we're currently discussing {topic}."
            return "We haven't focused on a specific conversation topic yet in this session."

        # 4. Overall memory breakdown query
        if any(pat in q for pat in ("what do you remember about me", "show memory", "what is in your memory")):
            lines = ["Here is what I remember:"]
            # Persistent memory
            if self.persistent_slots:
                lines.append("\n**Persistent Profile (Saved across sessions):**")
                for k, v in self.persistent_slots.items():
                    clean_k = k.replace("_", " ").title()
                    lines.append(f"• {clean_k}: {v}")
            else:
                lines.append("\n**Persistent Profile:** (Empty)")

            # Session context
            lines.append("\n**Active Session Context (Working Memory):**")
            lines.append(f"• Conversation turns: {len(self.messages)}")
            if self.session_slots.get("current_topic"):
                lines.append(f"• Current topic: {self.session_slots['current_topic']}")

            return "\n".join(lines)

        # 5. Previous question query
        if any(pat in q for pat in (
            "what was my previous question", "what was my last question",
            "what did i just ask", "what did i ask earlier", "what was my previous message",
            "what was the last thing i asked", "what did i ask before"
        )):
            user_msgs = [m for m in self.messages if m.role == Role.USER]
            if user_msgs:
                last_q = user_msgs[-1].content
                return f"Your previous question was: \"{last_q}\""
            return "You haven't asked any previous questions yet in this session."

        return None

    def save_history(self) -> None:
        """Save conversation turn history to the root chat_history.json file."""
        if not self.auto_save or self.history_file is None:
            return
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": "1.0",
                "last_updated": time.time(),
                "turn_count": len(self.messages),
                "messages": [m.to_dict() for m in self.messages],
                "session_slots": self.session_slots,
                "persistent_slots": self.persistent_slots,
            }
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            from robbie.utils.logger import logger
            logger.warning(f"Could not save chat history to {self.history_file}: {e}")

    def load_history(self) -> bool:
        """Load conversation turn history from the root chat_history.json file."""
        if self.history_file is None or not self.history_file.is_file():
            return False
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_msgs = data.get("messages", [])
            self.messages = [Message.from_dict(m) for m in raw_msgs]
            if "session_slots" in data and isinstance(data["session_slots"], dict):
                self.session_slots.update(data["session_slots"])
            if "persistent_slots" in data and isinstance(data["persistent_slots"], dict):
                self.persistent_slots.update(data["persistent_slots"])
            self.slots.sync()
            return True
        except Exception as e:
            from robbie.utils.logger import logger
            logger.warning(f"Could not load chat history from {self.history_file}: {e}")
            return False

    def delete_history(self) -> bool:
        """Delete chat history file from the root directory and clear in-memory turns."""
        self.messages.clear()
        self.session_slots = {
            "session_start_time": time.time(),
            "current_topic": None,
        }
        self.slots.sync()
        if self.history_file is not None and self.history_file.is_file():
            try:
                self.history_file.unlink()
                return True
            except Exception as e:
                from robbie.utils.logger import logger
                logger.warning(f"Could not delete chat history at {self.history_file}: {e}")
                return False
        return True

    def search_history(self, query: str, top_k: int = 3) -> List[Tuple[Message, float]]:
        """Search past messages in the conversation history by token overlap."""
        q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        if not q_tokens or not self.messages:
            return []
        results = []
        for msg in self.messages:
            m_tokens = set(re.findall(r"\b\w+\b", msg.content.lower()))
            overlap = len(q_tokens & m_tokens)
            if overlap > 0:
                score = overlap / len(q_tokens)
                results.append((msg, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_recent_turns(self, n: int = 5) -> List[Message]:
        """Return the n most recent conversation messages."""
        return self.messages[-n:] if self.messages else []

    def clear(self, keep_slots: bool = True, keep_persistent: bool = True, delete_file: bool = False) -> None:
        """Clear conversation turn history and session context.

        Args:
            keep_slots: Retain slots view.
            keep_persistent: If False, also clears persistent memory on disk.
            delete_file: If True, also deletes chat_history.json from disk.
        """
        self.messages.clear()
        self.session_slots = {
            "session_start_time": time.time(),
            "current_topic": None,
        }

        if not keep_slots or not keep_persistent:
            self.persistent_slots.clear()
            if self.persistent_memory is not None:
                self.persistent_memory.clear()

        self.slots.sync()
        if delete_file:
            self.delete_history()
        else:
            self.save_history()

    @property
    def turn_count(self) -> int:
        """Total turns in active session."""
        return len(self.messages)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation context."""
        return {
            "messages": [m.to_dict() for m in self.messages],
            "session_slots": self.session_slots,
            "persistent_slots": self.persistent_slots,
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
        persistent_memory: Optional[Any] = None,
    ) -> "ConversationContext":
        """Load chat session from JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        ctx = cls(
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            persistent_memory=persistent_memory,
        )
        ctx.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        ctx.session_slots = data.get("session_slots", {"session_start_time": time.time(), "current_topic": None})
        ctx.persistent_slots = data.get("persistent_slots", data.get("slots", {}))
        ctx.slots.sync()
        return ctx

