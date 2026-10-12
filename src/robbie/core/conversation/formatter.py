"""Conversation formatting and prompt building for Robbie language models."""

from typing import List, Optional, Tuple, Any
from robbie.core.conversation.messages import Message, Role


class ConversationFormatter:
    """Format structured conversation turns into text prompts for Robbie."""

    DEFAULT_SYSTEM_PROMPT = (
        "Robbie is a personal artificial intelligence assistant built from scratch with Python and PyTorch. "
        "Robbie is helpful, direct, polite, and skilled in programming, math, and science."
    )

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        delimiter: str = "\n",
        assistant_tag: str = "Robbie:",
        user_tag: str = "User:",
        system_tag: str = "System:",
    ):
        self.system_prompt = system_prompt
        self.delimiter = delimiter
        self.assistant_tag = assistant_tag
        self.user_tag = user_tag
        self.system_tag = system_tag

    def format_turn(self, message: Message) -> str:
        """Format a single message turn."""
        if message.role == Role.USER:
            tag = self.user_tag
        elif message.role == Role.ASSISTANT:
            tag = self.assistant_tag
        elif message.role == Role.SYSTEM:
            tag = self.system_tag
        else:
            tag = f"{message.role.value}:"
        return f"{tag} {message.content}"

    def format_dialogue(
        self,
        messages: List[Message],
        add_generation_prompt: bool = True,
        include_system_prompt: bool = False,
    ) -> str:
        """Format a sequence of messages into a complete text prompt.

        Args:
            messages: List of Message instances in chronological order.
            add_generation_prompt: If True, append 'Robbie:' ready for next token generation.
            include_system_prompt: If True, include self.system_prompt at the start.

        Returns:
            Formatted string ready for tokenization and model generation.
        """
        lines: List[str] = []

        if include_system_prompt and self.system_prompt:
            lines.append(f"{self.system_tag} {self.system_prompt}")

        for msg in messages:
            lines.append(self.format_turn(msg))

        prompt = self.delimiter.join(lines)

        if add_generation_prompt:
            if prompt:
                prompt = f"{prompt}{self.delimiter}{self.assistant_tag}"
            else:
                prompt = self.assistant_tag

        return prompt

    def fit_to_context_budget(
        self,
        messages: List[Message],
        tokenizer: Any,
        max_prompt_tokens: int = 100,
        include_system_prompt: bool = False,
    ) -> Tuple[List[Message], str, List[int]]:
        """Prune older turns from messages until formatted prompt fits max_prompt_tokens.

        Preserves the most recent messages (sliding window) while guaranteeing
        the prompt ends with 'Robbie:' ready for model generation.

        Returns:
            Tuple of (pruned_messages, formatted_prompt_text, token_ids).
        """
        if not messages:
            empty_prompt = self.format_dialogue([], add_generation_prompt=True)
            encoded = tokenizer.encode(empty_prompt, add_special_tokens=False)
            return [], empty_prompt, encoded

        # Try keeping from all messages down to just the last message
        for start_idx in range(len(messages)):
            subset = messages[start_idx:]
            prompt = self.format_dialogue(
                subset,
                add_generation_prompt=True,
                include_system_prompt=include_system_prompt,
            )
            encoded = tokenizer.encode(prompt, add_special_tokens=False)
            if len(encoded) <= max_prompt_tokens:
                return subset, prompt, encoded

        # If even the last message exceeds budget, return the last message alone
        last_subset = [messages[-1]]
        last_prompt = self.format_dialogue(
            last_subset,
            add_generation_prompt=True,
            include_system_prompt=include_system_prompt,
        )
        encoded = tokenizer.encode(last_prompt, add_special_tokens=False)
        return last_subset, last_prompt, encoded
