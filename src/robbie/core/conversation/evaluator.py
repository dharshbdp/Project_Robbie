"""Conversation evaluation metrics and response quality assessment."""

from typing import Dict, List, Any
import math
import time
import torch


class ConversationEvaluator:
    """Evaluate response quality, n-gram diversity, latency, and consistency."""

    @staticmethod
    def distinct_ngrams(text: str, n: int = 1) -> float:
        """Compute ratio of unique n-grams to total n-grams (distinct-n metric)."""
        tokens = text.split()
        if len(tokens) < n:
            return 1.0
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 1.0
        return len(set(ngrams)) / len(ngrams)

    @staticmethod
    def measure_generation_speed(
        engine: Any,
        prompt: str = "Explain what an attention mechanism is in machine learning.",
    ) -> Dict[str, Any]:
        """Benchmark token throughput and latency for the chat engine."""
        t0 = time.time()
        reply = engine.respond(prompt)
        elapsed = time.time() - t0

        token_count = len(engine.tokenizer.encode(reply, add_special_tokens=False))
        tokens_per_sec = token_count / elapsed if elapsed > 0 else 0.0

        return {
            "elapsed_seconds": round(elapsed, 4),
            "generated_tokens": token_count,
            "tokens_per_second": round(tokens_per_sec, 2),
            "reply_length_chars": len(reply),
            "distinct_1": round(ConversationEvaluator.distinct_ngrams(reply, 1), 3),
            "distinct_2": round(ConversationEvaluator.distinct_ngrams(reply, 2), 3),
        }

    @staticmethod
    def run_multi_turn_test(engine: Any) -> Dict[str, Any]:
        """Run standard multi-turn conversation scenario and verify context flow."""
        engine.context.clear(keep_slots=False)

        turns = [
            ("Hello Robbie!", "greeting"),
            ("My name is Dharsh", "introduction"),
            ("What is my name?", "memory_retrieval"),
            ("How do I reverse a list in Python?", "code_question"),
        ]

        results: List[Dict[str, Any]] = []
        for user_msg, intent in turns:
            t0 = time.time()
            reply = engine.respond(user_msg)
            elapsed = time.time() - t0
            results.append({
                "intent": intent,
                "user": user_msg,
                "reply": reply,
                "latency_sec": round(elapsed, 3),
            })

        # Name retrieval test check
        name_turn = results[2]["reply"]
        name_recalled = "Dharsh" in name_turn

        return {
            "total_turns": len(results),
            "turns": results,
            "name_recalled": name_recalled,
            "slots": engine.context.slots,
        }
