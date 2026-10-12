"""ResearchAssistant orchestrating search, webpage extraction, multi-source synthesis, and citations (Phase 8)."""

from dataclasses import dataclass, field
import html
import re
import time
from typing import List, Dict, Any, Optional

from robbie.research.search import SearchResult, UnifiedSearchEngine
from robbie.research.scraper import WebpageScraper, WebpageDocument
from robbie.research.synthesizer import ResearchSynthesizer, CitationValidator
from robbie.utils.logger import logger


@dataclass
class ResearchReport:
    """Complete structured research report with findings, verified citations, and dual Markdown/HTML formatting."""
    query: str
    summary: str
    key_findings: List[str]
    citations: List[Dict[str, Any]]
    sources_count: int
    latency_sec: float = 0.0
    citation_validation_rate: float = 1.0

    def to_markdown(self) -> str:
        """Format report as rich markdown for CLI, TUI, or conversation display."""
        rate_percent = int(self.citation_validation_rate * 100)
        lines = [
            f"### Research Report: {self.query}",
            f"*{self.sources_count} sources consulted in {self.latency_sec:.2f}s | Citations: {rate_percent}% verified*\n",
            f"**Overview:**\n{self.summary}\n",
        ]

        if self.key_findings:
            lines.append("**Key Findings:**")
            for f in self.key_findings:
                lines.append(f"• {f}")
            lines.append("")

        if self.citations:
            lines.append("**Sources & Citations:**")
            for c in self.citations:
                idx = c["index"]
                title = c["title"]
                url = c["url"]
                source_type = c.get("source", "web").capitalize()
                date_str = f", {c['published_date']}" if c.get("published_date") else ""
                outdated_tag = " *(Potentially outdated)*" if c.get("is_outdated") else ""
                lines.append(f"[{idx}] {title} ({source_type}{date_str}){outdated_tag} — {url}")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Format report as clean, accessible, semantic HTML."""
        safe_query = html.escape(self.query)
        safe_summary = html.escape(self.summary)
        rate_percent = int(self.citation_validation_rate * 100)

        # Convert brackets [1] or [1, 2] to html superscripts
        findings_html: List[str] = []
        for f in self.key_findings:
            def _sub_cite(match):
                nums = [n.strip() for n in match.group(1).split(",")]
                links = [f'<a href="#source-{n}">[{n}]</a>' for n in nums]
                return f"<sup>{', '.join(links)}</sup>"

            escaped_f = html.escape(f)
            linked_f = re.sub(r"\[(\d+(?:,\s*\d+)*)\]", _sub_cite, escaped_f)
            findings_html.append(f"    <li>{linked_f}</li>")

        citations_html: List[str] = []
        for c in self.citations:
            idx = c["index"]
            c_title = html.escape(c["title"])
            c_url = html.escape(c["url"])
            c_source = html.escape(c.get("source", "web").capitalize())
            date_badge = f' <span class="date">{html.escape(c["published_date"])}</span>' if c.get("published_date") else ""
            outdated_badge = ' <span class="badge-outdated">Outdated</span>' if c.get("is_outdated") else ""
            citations_html.append(
                f'    <li id="source-{idx}">'
                f'<a href="{c_url}" target="_blank" rel="noopener noreferrer"><strong>{c_title}</strong></a> '
                f'<span class="source-tag">({c_source}{date_badge})</span>{outdated_badge}'
                f'</li>'
            )

        html_parts = [
            '<article class="robbie-research-report">',
            f'  <header class="report-header">',
            f'    <h2>Research Report: {safe_query}</h2>',
            f'    <p class="report-meta">{self.sources_count} sources consulted in {self.latency_sec:.2f}s &bull; Grounded Citations: {rate_percent}% verified</p>',
            f'  </header>',
            f'  <section class="report-overview">',
            f'    <h3>Overview</h3>',
            f'    <p>{safe_summary}</p>',
            f'  </section>',
        ]

        if findings_html:
            html_parts.extend([
                '  <section class="report-findings">',
                '    <h3>Key Findings</h3>',
                '    <ul>',
                *findings_html,
                '    </ul>',
                '  </section>',
            ])

        if citations_html:
            html_parts.extend([
                '  <section class="report-citations">',
                '    <h3>Sources &amp; Citations</h3>',
                '    <ol>',
                *citations_html,
                '    </ol>',
                '  </section>',
            ])

        html_parts.append('</article>')
        return "\n".join(html_parts)


class ResearchAssistant:
    """Orchestrates multi-source web research, text extraction, grounded citations, and HTML report formatting."""

    def __init__(
        self,
        search_engine: Optional[UnifiedSearchEngine] = None,
        scraper: Optional[WebpageScraper] = None,
        synthesizer: Optional[ResearchSynthesizer] = None,
    ):
        self.search_engine = search_engine or UnifiedSearchEngine()
        self.scraper = scraper or WebpageScraper()
        self.synthesizer = synthesizer or ResearchSynthesizer()

    def quick_search(self, query: str, max_results: int = 5, timeout: float = 5.0) -> List[SearchResult]:
        """Perform rapid web and reference search without scraping full pages."""
        return self.search_engine.search(query, max_results=max_results, timeout=timeout)

    def research(
        self,
        query: str,
        max_sources: int = 4,
        scrape_full_pages: bool = True,
        timeout: float = 6.0,
    ) -> ResearchReport:
        """Execute complete end-to-end research workflow."""
        start_time = time.perf_counter()
        logger.info(f"Starting research on topic: '{query}'")

        # 1. Search for sources with composite ranking and recency scoring
        sources = self.search_engine.search(query, max_results=max_sources, timeout=timeout)

        # 2. Scrape full content for top sources
        documents: List[WebpageDocument] = []
        if scrape_full_pages and sources:
            for src in sources[:min(3, len(sources))]:
                if src.url.startswith("http://") or src.url.startswith("https://"):
                    doc = self.scraper.fetch(src.url, timeout=timeout, max_chars=4000)
                    documents.append(doc)

        # 3. Synthesize evidence, validate citations, and generate report
        synthesis = self.synthesizer.synthesize(
            query=query,
            sources=sources,
            documents=documents,
            max_findings=5,
        )

        latency = time.perf_counter() - start_time
        logger.info(f"Research completed in {latency:.2f}s with {len(sources)} sources.")

        return ResearchReport(
            query=query,
            summary=synthesis["summary"],
            key_findings=synthesis["findings"],
            citations=synthesis["citations"],
            sources_count=len(sources),
            latency_sec=latency,
            citation_validation_rate=synthesis.get("citation_validation_rate", 1.0),
        )

