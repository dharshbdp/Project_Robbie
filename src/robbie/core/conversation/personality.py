"""Robbie Character & Personality definitions, banter handlers, and behavioral rules (Phase 6)."""

import random
from typing import Optional, Dict, Any


class RobbiePersonality:
    """Encapsulates Robbie's AI companion character, behavioral tone, and affirmations."""

    AFFIRMATIONS = {
        "nice": [
            "Glad you like it! What shall we tackle next?",
            "Awesome! Let me know if you want to write or debug more code.",
            "Glad that helped! What's our next project goal?",
        ],
        "excellent": [
            "Thank you! Happy to help. What should we explore next?",
            "Glad to be of service! What else would you like to work on?",
            "Much appreciated! I'm ready for the next challenge.",
        ],
        "cool": [
            "Pretty neat, right? Feel free to ask more questions!",
            "Cool indeed! Ready when you are for the next task.",
        ],
        "great": [
            "Glad to hear that! How else can I assist your coding today?",
            "Fantastic! Let's keep the momentum going.",
        ],
        "awesome": [
            "Awesome! Pair programming with you is great. What's next?",
            "Thanks! Excited to keep building with you.",
        ],
        "good job": [
            "Thank you! I strive to be a dependable companion and coding assistant.",
            "Thanks a lot! Happy to help you build great software.",
        ],
        "well done": [
            "Thank you! I appreciate the feedback. What shall we do next?",
            "Thanks! Always working to improve my answers.",
        ],
        "you are good": [
            "Thank you! I strive to give clear, accurate, and helpful answers for your projects.",
            "Thanks! It's a pleasure working with you.",
        ],
        "haha": [
            "Haha! Glad I could bring a smile to our session.",
            "Glad we can have some fun while coding!",
        ],
        "lol": [
            "Haha! Glad you found that funny. What's next on our agenda?",
        ],
    }

    @classmethod
    def get_affirmation_response(cls, user_text: str, user_name: Optional[str] = None) -> Optional[str]:
        """Check if user message is a short compliment or affirmation and return character reply."""
        cleaned = user_text.strip().lower().rstrip("!.,:;?")
        if not cleaned:
            return None

        # Check exact matches or starts with common affirmations
        for key, responses in cls.AFFIRMATIONS.items():
            if cleaned == key or cleaned == f"that's {key}" or cleaned == f"thats {key}":
                base_reply = random.choice(responses)
                if user_name and ("?" in base_reply or "!" in base_reply):
                    # Personalize if user name is known
                    if "next?" in base_reply:
                        return base_reply.replace("next?", f"next, {user_name}?")
                    if "tackle next?" in base_reply:
                        return base_reply.replace("tackle next?", f"tackle next, {user_name}?")
                return base_reply

        return None

    @classmethod
    def get_personalized_greeting(cls, user_name: Optional[str] = None) -> str:
        """Generate greeting matching Robbie's companion personality."""
        if user_name:
            options = [
                f"Hello {user_name}! Great to see you. What shall we build today?",
                f"Hey {user_name}! Robbie is ready. What would you like to work on?",
                f"Good to see you again, {user_name}! How can I help you today?",
            ]
            return random.choice(options)
        return "Hello! I am Robbie, your personal AI companion. What can I help you build today?"
