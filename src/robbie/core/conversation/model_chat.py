"""Model-only conversation, shared by interactive chat and generation reviews."""

import torch

from robbie.core.conversation.formatter import ConversationFormatter
from robbie.core.conversation.messages import Message, Role


class ModelChat:
    def __init__(self, model, tokenizer, max_tokens=160, temperature=0.5,
                 assistant_eos=True, repetition_penalty=1.2):
        self.model, self.tokenizer = model, tokenizer
        self.device = next(model.parameters()).device
        self.max_tokens, self.temperature = max_tokens, temperature
        self.assistant_eos, self.repetition_penalty = assistant_eos, repetition_penalty
        self.messages, self.last_generation = [], {}

    def reply(self, question):
        import re
        q_clean = question.strip().lower().rstrip("?!. ")
        is_greeting = q_clean in ("hi", "hello", "hey", "good morning", "good evening", "greetings")
        refers_to_history = bool(re.search(
            r"\b(that|it|this|those|them|above|earlier|previous|why|more|again|too|also|continue)\b",
            q_clean,
        ))
        is_fresh_concept = bool(re.search(r"^(?:what (?:is|are)|whats|what's|explain|define)\b", q_clean)) or q_clean in (
            "web", "python", "code", "internet", "git", "ram", "cpu", "gpu", "vram", "cuda", "storage", "disk"
        )
        if is_greeting or (is_fresh_concept and not refers_to_history):
            prompt_messages = [Message(Role.USER, question)]
        else:
            prompt_messages = self.messages + [Message(Role.USER, question)]

        budget = self.model.config.block_size - self.max_tokens
        formatter = ConversationFormatter()
        while True:
            ids = formatter.encode_dialogue(prompt_messages, self.tokenizer, self.assistant_eos)
            if len(ids) <= budget:
                break
            if len(prompt_messages) <= 1:
                raise ValueError(f"Question exceeds the {budget}-token prompt budget. Shorten it or reduce --max-tokens.")
            prompt_messages = prompt_messages[2:]

        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        output = self.model.generate(x, max_new_tokens=self.max_tokens, temperature=self.temperature,
                                     top_k=40, repetition_penalty=self.repetition_penalty,
                                     stop_token_id=self.tokenizer.eos_token_id)
        generated = output[0, len(ids):].tolist()
        self.last_generation = {"stopped_on_eos": bool(generated and generated[-1] == self.tokenizer.eos_token_id),
                                "generated_tokens": len(generated)}
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        for marker in ("\nUser:", "User:", "\nRobbie:", "Robbie:", "<|endoftext|>", "<|im_end|>"):
            if marker in answer:
                answer = answer.split(marker)[0].strip()
        self.messages.append(Message(Role.USER, question))
        self.messages.append(Message(Role.ASSISTANT, answer))
        return answer
