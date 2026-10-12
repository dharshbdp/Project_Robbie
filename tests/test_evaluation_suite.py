"""Repeatable unit tests for the Model, Research Pipeline, and Computer-Awareness Evaluation Harness (Priority 4)."""

from unittest.mock import MagicMock
import pytest
import torch

from robbie.evaluation import (
    ModelEvaluator,
    ModelEvaluationResult,
    ResearchEvaluator,
    ResearchEvaluationResult,
    ComputerAwarenessEvaluator,
    ComputerEvaluationResult,
    RobbieEvaluationSuite,
    SuiteReport,
)
from robbie.research import (
    ResearchReport,
    SearchResult,
)
from robbie.system import ComputerAwareness
from robbie.computer.tools import ComputerToolManager


class TestModelEvaluation:
    """Test repeatable model evaluation routines."""

    def test_distinct_ngrams(self):
        text_diverse = "attention mechanism allows parallel representation learning"
        text_repetitive = "the model the model the model the model"

        d1_div = ModelEvaluator.distinct_ngrams(text_diverse, n=1)
        d1_rep = ModelEvaluator.distinct_ngrams(text_repetitive, n=1)

        assert d1_div == 1.0
        assert d1_rep < 0.40

    def test_evaluate_diversity(self):
        samples = [
            "transformers utilize multi head self attention",
            "cross entropy loss penalizes incorrect token probabilities",
        ]
        d1, d2 = ModelEvaluator.evaluate_diversity(samples)
        assert d1 > 0.8
        assert d2 > 0.8

    def test_multi_turn_retention_mock(self):
        mock_engine = MagicMock()
        mock_context = MagicMock()
        mock_engine.context = mock_context

        # Mock responses simulating correct memory recall
        def mock_respond(msg):
            if "What is my name?" in msg:
                return "Your name is Alex!"
            if "What is my role?" in msg:
                return "You are a roboticist."
            return "Got it!"

        mock_engine.respond = mock_respond

        res = ModelEvaluator.evaluate_multi_turn_retention(mock_engine)
        assert res["name_recalled"] is True
        assert res["role_recalled"] is True
        assert res["recall_rate"] == 1.0


class TestResearchEvaluation:
    """Test repeatable research pipeline evaluation routines."""

    def test_html_validation_valid_and_invalid(self):
        valid_html = """
        <article class="robbie-research-report">
            <header><h2>Report</h2></header>
            <section><p>Finding <sup><a href="#source-1">[1]</a></sup></p></section>
            <section><ol><li id="source-1"><a href="https://example.com">Source</a></li></ol></section>
        </article>
        """
        assert ResearchEvaluator.validate_html(valid_html) is True

        # Invalid due to broken tag / missing target
        invalid_html = """
        <article class="robbie-research-report">
            <p>Finding <sup><a href="#source-99">[99]</a></sup></p>
        </article>
        """
        assert ResearchEvaluator.validate_html(invalid_html) is False

    def test_evaluate_report_structure(self):
        report = ResearchReport(
            query="Deep Learning",
            summary="Overview of deep learning.",
            key_findings=["Neural networks learn hierarchical representations [1]."],
            citations=[{"index": 1, "title": "DL Book", "url": "https://deeplearning.org", "source": "web"}],
            sources_count=1,
            citation_validation_rate=1.0,
        )

        res = ResearchEvaluator.evaluate_report(report)
        assert res["sources_count"] == 1
        assert res["findings_count"] == 1
        assert res["has_no_orphans"] is True
        assert res["html_valid"] is True


class TestComputerAwarenessEvaluation:
    """Test repeatable computer awareness and safety evaluation routines."""

    def test_awareness_invariants(self):
        ca = ComputerAwareness()
        res = ComputerAwarenessEvaluator.evaluate_awareness_invariants(ca)

        assert res["cpu_valid"] is True
        assert res["ram_valid"] is True
        assert res["storage_valid"] is True
        assert res["all_invariants_pass"] is True
        assert res["git_detected"] is True

    def test_process_monitor_invariants(self):
        ca = ComputerAwareness()
        res = ComputerAwarenessEvaluator.evaluate_process_monitor(ca)

        assert res["passed"] is True
        assert res["is_populated"] is True
        assert res["has_valid_pids"] is True
        assert res["is_sorted_by_memory"] is True

    def test_tool_safety_guards(self):
        tm = ComputerToolManager(read_only=True)
        res = ComputerAwarenessEvaluator.evaluate_tool_safety(tm)

        assert res["passed"] is True
        assert res["read_only_default_enforced"] is True
        assert res["sensitive_path_blocked"] is True
        assert res["dangerous_command_blocked"] is True


class TestEvaluationSuiteRunner:
    """Test unified evaluation suite runner and scorecard output."""

    def test_suite_runner_and_markdown_scorecard(self):
        report = RobbieEvaluationSuite.run_all()

        assert isinstance(report, SuiteReport)
        assert report.all_passed is True
        assert report.model.passed is True
        assert report.research.passed is True
        assert report.computer.passed is True

        md = report.to_markdown()
        assert "# Project Robbie — System Evaluation Scorecard" in md
        assert "1. Conversational Model Benchmarks" in md
        assert "2. Research Pipeline Benchmarks" in md
        assert "3. Computer Awareness & Tool Safety Benchmarks" in md
        assert "PASS ✅" in md
