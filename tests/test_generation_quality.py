"""Do not confuse a lower teacher-forced loss with a working chatbot."""

from robbie.core.training.generation_quality import assess_samples, repetition_metrics


def test_detects_the_observed_loops_without_matching_ram_or_other_topics():
    for output in ["RAM in " * 25, "the bit of " * 20, "a purple bird sings " * 15]:
        assert repetition_metrics(output)["obvious_repetition"]


def test_short_repetition_and_ordinary_prose_are_not_loop_failures():
    for output in ["Very, very good.", "Hello! How can I help you today?",
                   "RAM stores the data that programs are using. Storage keeps files when the computer is off."]:
        assert not repetition_metrics(output)["obvious_repetition"]


def test_failed_sample_flags_report_but_clean_samples_do_not_claim_readiness():
    bad = assess_samples([{"question": "Hi", "answer": "the bit of " * 25}])
    assert bad["status"] == "failed_generation_checks"
    assert bad["failed_samples"] == 1
    assert assess_samples([{"answer": ""}])["failed_samples"] == 1
    clean = assess_samples([{"answer": "RAM is a city in the sky."}])
    assert clean["status"] == "needs_manual_review"  # Fluent falsehoods need human review.
