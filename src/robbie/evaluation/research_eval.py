"""Repeatable evaluation harness for research accuracy, source ranking, citations, and HTML formatting (Priority 4)."""

from dataclasses import dataclass, field
from html.parser import HTMLParser
import re
import time
from typing import List, Dict, Any, Optional

from robbie.research.assistant import ResearchAssistant, ResearchReport
from robbie.research.search import SearchResult
from robbie.research.synthesizer import CitationValidator


@dataclass
class ResearchEvaluationResult:
    """Summary of research pipeline evaluation metrics and factuality benchmarks."""
    queries_evaluated: int = 0
    citation_validity_rate: float = 1.0
    html_validity_rate: float = 1.0
    avg_latency_sec: float = 0.0
    findings_count: int = 0
    passed: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queries_evaluated": self.queries_evaluated,
            "citation_validity_rate": round(self.citation_validity_rate, 3),
            "html_validity_rate": round(self.html_validity_rate, 3),
            "avg_latency_sec": round(self.avg_latency_sec, 3),
            "findings_count": self.findings_count,
            "passed": self.passed,
            "details": self.details,
        }


class SimpleHTMLValidator(HTMLParser):
    """Validates HTML syntax balance, anchor targets, and tag closures."""

    SELF_CLOSING = {"img", "br", "hr", "input", "meta", "link"}

    def __init__(self):
        super().__init__()
        self.tag_stack: List[str] = []
        self.ids: set = set()
        self.href_targets: List[str] = []
        self.has_script_or_error = False

    def handle_starttag(self, tag: str, attrs: List[tuple]):
        tag_lower = tag.lower()
        if tag_lower == "script":
            self.has_script_or_error = True
        attr_dict = dict(attrs)
        if "id" in attr_dict and attr_dict["id"]:
            self.ids.add(attr_dict["id"])
        if "href" in attr_dict and attr_dict["href"] and attr_dict["href"].startswith("#"):
            self.href_targets.append(attr_dict["href"][1:])
        if tag_lower not in self.SELF_CLOSING:
            self.tag_stack.append(tag_lower)

    def handle_endtag(self, tag: str):
        tag_lower = tag.lower()
        if tag_lower in self.SELF_CLOSING:
            return
        if self.tag_stack and self.tag_stack[-1] == tag_lower:
            self.tag_stack.pop()
        elif tag_lower in self.tag_stack:
            while self.tag_stack and self.tag_stack[-1] != tag_lower:
                self.tag_stack.pop()
            if self.tag_stack:
                self.tag_stack.pop()


class ResearchEvaluator:
    """Evaluates factual grounding, citation validity, and HTML formatting integrity."""

    @staticmethod
    def validate_html(html_str: str) -> bool:
        """Verify that report HTML has balanced tags, no scripts, and valid citation anchors."""
        if not html_str or "<article" not in html_str:
            return False
        validator = SimpleHTMLValidator()
        try:
            validator.feed(html_str)
            if validator.has_script_or_error:
                return False
            # Tag stack should be cleanly closed or minimal
            if len(validator.tag_stack) > 2:
                return False
            # Check citation internal anchors (e.g. #source-1 must map to id="source-1")
            for target in validator.href_targets:
                if target not in validator.ids:
                    return False
            return True
        except Exception:
            return False

    @classmethod
    def evaluate_report(cls, report: ResearchReport) -> Dict[str, Any]:
        """Assess a single research report's citation validity, grounding, and HTML structure."""
        html_text = report.to_html()
        html_valid = cls.validate_html(html_text)

        # Check citations coverage
        cited_indices = set()
        for f in report.key_findings:
            matches = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", f)
            for m in matches:
                for idx_str in m.split(","):
                    cited_indices.add(int(idx_str.strip()))

        source_indices = {c["index"] for c in report.citations}
        orphan_citations = cited_indices - source_indices
        has_no_orphans = len(orphan_citations) == 0

        return {
            "query": report.query,
            "sources_count": report.sources_count,
            "findings_count": len(report.key_findings),
            "citation_validation_rate": report.citation_validation_rate,
            "html_valid": html_valid,
            "has_no_orphans": has_no_orphans,
            "orphan_citations": list(orphan_citations),
        }

    @classmethod
    def run_benchmark(
        cls,
        assistant: ResearchAssistant,
        test_queries: Optional[List[str]] = None,
    ) -> ResearchEvaluationResult:
        """Run repeatable evaluation across multiple research queries."""
        queries = test_queries or [
            "Transformer neural network architecture",
            "Byte Pair Encoding subword tokenization",
            "Cross entropy loss and validation perplexity",
        ]

        total_validity = 0.0
        html_valid_count = 0
        total_findings = 0
        total_latency = 0.0
        query_evals = []

        for q in queries:
            t0 = time.perf_counter()
            report = assistant.research(q, max_sources=3, scrape_full_pages=False)
            dt = time.perf_counter() - t0
            total_latency += dt

            ev = cls.evaluate_report(report)
            query_evals.append(ev)

            total_validity += report.citation_validation_rate
            if ev["html_valid"]:
                html_valid_count += 1
            total_findings += len(report.key_findings)

        num_q = len(queries)
        avg_validity = total_validity / num_q if num_q > 0 else 1.0
        html_rate = html_valid_count / num_q if num_q > 0 else 1.0
        avg_latency = total_latency / num_q if num_q > 0 else 0.0

        passed = (avg_validity >= 0.80) and (html_rate >= 0.66)

        return ResearchEvaluationResult(
            queries_evaluated=num_q,
            citation_validity_rate=avg_validity,
            html_validity_rate=html_rate,
            avg_latency_sec=avg_latency,
            findings_count=total_findings,
            passed=passed,
            details={"queries": query_evals},
        )
