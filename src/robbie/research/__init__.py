"""Web search, content retrieval, multi-source synthesis, and citations (Phase 8)."""

from robbie.research.search import (
    SearchResult,
    DuckDuckGoSearchEngine,
    WikipediaSearchEngine,
    UnifiedSearchEngine,
)
from robbie.research.scraper import (
    WebpageDocument,
    HTMLContentExtractor,
    WebpageScraper,
)
from robbie.research.synthesizer import (
    ScoredEvidence,
    ResearchSynthesizer,
)
from robbie.research.assistant import (
    ResearchReport,
    ResearchAssistant,
)

__all__ = [
    "SearchResult",
    "DuckDuckGoSearchEngine",
    "WikipediaSearchEngine",
    "UnifiedSearchEngine",
    "WebpageDocument",
    "HTMLContentExtractor",
    "WebpageScraper",
    "ScoredEvidence",
    "ResearchSynthesizer",
    "ResearchReport",
    "ResearchAssistant",
]
