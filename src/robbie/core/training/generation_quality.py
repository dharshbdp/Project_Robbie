"""Detect obvious generation failures without claiming to score correctness."""

from collections import Counter
import re


def repetition_metrics(text, n=4):
    words = re.findall(r"\w+", text.casefold())
    grams = [tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]
    counts = Counter(grams)
    repeated_fraction = (sum(count - 1 for count in counts.values()) / len(grams)
                         if grams else 0.0)
    # Short natural responses, such as "Very, very good", are not loops.
    looping = len(words) >= 16 and repeated_fraction >= 0.4
    return {"words": len(words), "repeated_4gram_fraction": round(repeated_fraction, 4),
            "obvious_repetition": looping, "empty": not text.strip()}


def assess_samples(rows):
    assessed = [{**row, **repetition_metrics(row["answer"])} for row in rows]
    failures = sum(row["obvious_repetition"] or row["empty"] or row.get("stopped_on_eos") is False for row in assessed)
    return {"status": "failed_generation_checks" if failures else "needs_manual_review",
            "failed_samples": failures, "sample_count": len(assessed), "samples": assessed,
            "note": "Checks loops, empty output, and EOS stopping when recorded. Passing does not establish relevance, factual correctness, or chatbot readiness."}
