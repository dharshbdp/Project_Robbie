"""Structured dialogue message definitions for Project Robbie."""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, Optional, Union


class Role(str, Enum):
    """Dialogue speaker roles."""
    USER = "User"
    ASSISTANT = "Robbie"
    SYSTEM = "System"

    @classmethod
    def from_str(cls, val: str) -> "Role":
        """Normalize string to Role enum."""
        v = val.strip().capitalize()
        if v in ("User", "Human"):
            return cls.USER
        if v in ("Robbie", "Assistant", "Ai", "Bot"):
            return cls.ASSISTANT
        if v in ("System", "Instruction"):
            return cls.SYSTEM
        return cls.USER


@dataclass
class Message:
    """A single turn in a conversation."""
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.role, str) and not isinstance(self.role, Role):
            self.role = Role.from_str(self.role)
        self.content = self.content.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Instantiate Message from dictionary."""
        return cls(
            role=Role.from_str(data.get("role", "User")),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        return f"{self.role.value}: {self.content}"
