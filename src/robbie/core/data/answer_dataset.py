"""Conversation-level supervised fine-tuning, with loss only on the latest answer."""

import re

import torch
from torch.utils.data import Dataset

from robbie.core.conversation.formatter import ConversationFormatter
from robbie.core.conversation.messages import Message, Role


class AnswerDataset(Dataset):
    """Keep questions and answers together; never supervise padding or user text.

    Each assistant turn is one example, with preceding turns as context. Overlong
    answers are skipped rather than teaching the model that an incomplete answer
    is complete. Older turns may be dropped, but the current question is preserved.
    """

    def __init__(self, conversations, tokenizer, seq_len=256):
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        self.samples = []
        self.skipped = 0
        self.supervised_tokens = 0
        formatter = ConversationFormatter()
        for conversation in conversations:
            turns = conversation["conversation"] if isinstance(conversation, dict) else conversation
            history = []
            for turn in turns:
                message = Message(Role.from_str(turn["role"]), turn["content"])
                if message.role == Role.ASSISTANT and history:
                    answer = tokenizer.encode(" " + message.content, add_special_tokens=False)
                    answer.append(tokenizer.eos_token_id)
                    prefix = list(history)
                    while len(prefix) > 1:
                        prompt = formatter.format_dialogue(prefix)
                        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                        if len(prompt_ids) + len(answer) <= seq_len + 1:
                            break
                        # Remove a complete earlier exchange, not half of a turn.
                        prefix = prefix[2:] if len(prefix) > 2 else prefix[-1:]
                    prompt_ids = tokenizer.encode(
                        formatter.format_dialogue(prefix), add_special_tokens=False
                    )
                    tokens = prompt_ids + answer
                    if len(tokens) > seq_len + 1:
                        self.skipped += 1
                    else:
                        x = torch.full((seq_len,), tokenizer.pad_token_id, dtype=torch.long)
                        y = torch.full((seq_len,), -100, dtype=torch.long)
                        x[:len(tokens) - 1] = torch.tensor(tokens[:-1])
                        start = len(prompt_ids) - 1
                        y[start:len(tokens) - 1] = torch.tensor(answer)
                        self.samples.append((x, y))
                        self.supervised_tokens += len(answer)
                history.append(message)
        if not self.samples:
            raise ValueError("No complete assistant answers fit the context window")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def conversations_from_text(text):
    """Read the existing stage-2 format, retaining multiline answer content."""
    conversations = []
    for block in text.split("<|endoftext|>"):
        matches = list(re.finditer(r"^(User|Robbie|Assistant|System):\s*", block, re.M))
        turns = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(block)
            content = block[match.end():end].strip()
            if content:
                turns.append({"role": match.group(1), "content": content})
        if any(t["role"] in ("Robbie", "Assistant") for t in turns):
            conversations.append({"conversation": turns})
    return conversations
