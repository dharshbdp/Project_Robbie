"""ResearchAssistant orchestrating search, webpage extraction, multi-source synthesis, and citations (Phase 8)."""

from dataclasses import dataclass, field
import time
from typing import List, Dict, Any, Optional

from robbie.research.search import SearchResult, UnifiedSearchEngine
from robbie.research.scraper import WebpageScraper, WebpageDocument
from robbie.research.synthesizer import ResearchSynthesizer
from robbie.utils.logger import logger


@dataclass
class ResearchReport:
    """Complete structured research report with findings and verified citations."""
    query: str
    summary: str
    key_findings: List[str]
    citations: List[Dict[str, Any]]
    sources_count: int
    latency_sec: float = 0.0

    def to_markdown(self) -> str:
        """Format report as rich markdown for CLI, TUI, or conversation display."""
        lines = [
            f"### Research Report: {self.query}",
            f"*{self.sources_count} sources consulted in {self.latency_sec:.2f}s*\n",
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
                lines.append(f"[{idx}] {title} ({source_type}) — {url}")

        return "\n".join(lines)


class ResearchAssistant:
    """Orchestrates multi-source web research, text extraction, and grounded citations."""

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

        # 1. Search for sources
        sources = self.search_engine.search(query, max_results=max_sources, timeout=timeout)

        # 2. Scrape full content for top sources
        documents: List[WebpageDocument] = []
        if scrape_full_pages and sources:
            # Scrape top 2-3 sources within timeout
            for src in sources[:min(3, len(sources))]:
                if src.url.startswith("http://") or src.url.startswith("https://"):
                    doc = self.scraper.fetch(src.url, timeout=timeout, max_chars=4000)
                    documents.append(doc)

        # 3. Synthesize evidence and citations
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
        )
