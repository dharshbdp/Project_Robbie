"""Persistent memory storage for Project Robbie (Phase 7)."""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from robbie.config import DATA_DIR
from robbie.utils.logger import logger


class PersistentMemory:
    """Long-term persistent memory for Robbie across conversation sessions.

    Maintains user profile attributes, preferences, project notes, and
    conversation episodes with disk persistence.
    """

    DEFAULT_STORE_PATH = DATA_DIR / "memory" / "robbie_memory.json"

    def __init__(self, store_path: Optional[Union[str, Path]] = None):
        self.store_path = Path(store_path or self.DEFAULT_STORE_PATH)
        self.profile: Dict[str, Any] = {}
        self.notes: Dict[str, Any] = {}
        self.episodes: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load memory state from disk if exists."""
        if not self.store_path.exists():
            return
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.profile = data.get("profile", {})
            self.notes = data.get("notes", {})
            self.episodes = data.get("episodes", [])
        except Exception as e:
            logger.warning(f"Could not load persistent memory from {self.store_path}: {e}")

    def save(self) -> None:
        """Save memory state to disk."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "last_updated": time.time(),
            "profile": self.profile,
            "notes": self.notes,
            "episodes": self.episodes,
        }
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # Profile & Preference Management
    # -------------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Get an attribute from the user profile."""
        return self.profile.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a user profile attribute and persist."""
        self.profile[key] = value
        self.save()

    def delete(self, key: str) -> bool:
        """Delete an attribute from profile."""
        if key in self.profile:
            del self.profile[key]
            self.save()
            return True
        return False

    def list_profile(self) -> Dict[str, Any]:
        """Return all saved profile attributes."""
        return dict(self.profile)

    # -------------------------------------------------------------------------
    # Notes & Project Knowledge
    # -------------------------------------------------------------------------

    def add_note(self, title: str, content: str) -> None:
        """Save a persistent note or project detail."""
        self.notes[title] = {
            "content": content,
            "created_at": time.time(),
        }
        self.save()

    def get_note(self, title: str) -> Optional[str]:
        """Retrieve content of a note by title."""
        note = self.notes.get(title)
        return note.get("content") if note else None

    def search_notes(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Search notes by token overlap relevance."""
        q_tokens = set(query.lower().split())
        if not q_tokens:
            return []

        results = []
        for title, item in self.notes.items():
            content = item.get("content", "")
            tokens = set((title + " " + content).lower().split())
            overlap = len(q_tokens & tokens)
            if overlap > 0:
                score = overlap / len(q_tokens)
                results.append((title, content, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    # -------------------------------------------------------------------------
    # Episodic Conversation Highlights
    # -------------------------------------------------------------------------

    def record_episode(self, summary: str, tags: Optional[List[str]] = None) -> None:
        """Record an episodic memory summary."""
        self.episodes.append({
            "summary": summary,
            "tags": tags or [],
            "timestamp": time.time(),
        })
        # Keep last 50 episodes to avoid unbounded file growth
        if len(self.episodes) > 50:
            self.episodes = self.episodes[-50:]
        self.save()

    def clear(self) -> None:
        """Clear all memory."""
        self.profile.clear()
        self.notes.clear()
        self.episodes.clear()
        self.save()
