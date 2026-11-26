"""Compact, complete-answer SFT examples and conversation-grouped splits."""

import hashlib
import json
import re

import torch

from robbie.core.conversation.formatter import ConversationFormatter
from robbie.core.conversation.messages import Message, Role


FORMAT_VERSION = 1


def digest_file(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def prompt_group(messages):
    """All branches/parapunctuation of the same opening prompt stay together."""
    first = next(m["content"] for m in messages if m["role"] == "user")
    normalized = " ".join(re.findall(r"\w+", first.casefold()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_for_group(group):
    bucket = int(group[:8], 16) % 1000
    return "test" if bucket < 10 else "validation" if bucket < 30 else "train"


def encode_conversation(messages, tokenizer, context, assistant_eos=False, with_turn_index=False):
    """Supervise each complete answer and EOS; retain system instructions.

    Drop complete old exchanges to fit, retaining the immediately preceding
    exchange for follow-ups. Never truncate the current user or answer, and
    never silently drop a system instruction to make it fit.
    """
    formatter = ConversationFormatter()
    history, examples = [], []
    for turn_index, turn in enumerate(messages):
        message = Message(Role.from_str(turn["role"]), turn["content"])
        if message.role == Role.ASSISTANT and history and history[-1].role == Role.USER:
            answer = tokenizer.encode(" " + message.content, add_special_tokens=False)
            answer.append(tokenizer.eos_token_id)
            prefix = list(history)
            while True:
                ids = formatter.encode_dialogue(prefix, tokenizer, assistant_eos=assistant_eos)
                if len(ids) + len(answer) <= context + 1:
                    example = {"tokens": torch.tensor(ids + answer, dtype=torch.int32),
                               "answer_start": len(ids) - 1}
                    if with_turn_index:
                        example["assistant_turn"] = turn_index
                    examples.append(example)
                    break
                # Preserve initial system turn(s), then drop one old exchange.
                first_user = next((i for i, msg in enumerate(prefix)
                                   if msg.role == Role.USER), len(prefix))
                if first_user + 4 >= len(prefix):
                    break
                prefix = prefix[:first_user] + prefix[first_user + 2:]
        history.append(message)
    return examples


def collate_answers(samples, pad_id):
    length = max(len(sample["tokens"]) - 1 for sample in samples)
    x = torch.full((len(samples), length), pad_id, dtype=torch.long)
    y = torch.full_like(x, -100)
    for row, sample in enumerate(samples):
        ids = sample["tokens"].long()
        size, start = len(ids) - 1, sample["answer_start"]
        x[row, :size] = ids[:-1]
        y[row, start:size] = ids[start + 1:]
    return x, y


def load_prepared(directory, tokenizer_path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("Re-prepare data: unsupported data format")
    if digest_file(tokenizer_path) != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from the tokenizer used to prepare this data")
    splits = {}
    for name in ("train", "validation", "test"):
        path = directory / (name + ".pt")
        if digest_file(path) != manifest["splits"][name]["sha256"]:
            raise ValueError(f"Prepared {name} data changed; re-prepare or restore it")
        splits[name] = torch.load(path, map_location="cpu", weights_only=True)
        if not splits[name]:
            raise ValueError(f"Prepared {name} split is empty")
    return splits, manifest
