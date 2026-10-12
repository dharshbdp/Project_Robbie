"""Comprehensive unit tests for Project Robbie Phase 8 Web Research Subsystem."""

from unittest.mock import MagicMock, patch
import pytest

from robbie.research.search import (
    SearchResult,
    DuckDuckGoHTMLParser,
    WikipediaSearchEngine,
    DuckDuckGoSearchEngine,
    UnifiedSearchEngine,
)
from robbie.research.scraper import (
    HTMLContentExtractor,
    WebpageScraper,
    WebpageDocument,
)
from robbie.research.synthesizer import (
    ScoredEvidence,
    ResearchSynthesizer,
    CitationValidator,
    CitationValidationResult,
)
from robbie.research.assistant import (
    ResearchReport,
    ResearchAssistant,
)
from robbie.core.conversation.engine import ChatEngine
from robbie.core.tokenizer import CharTokenizer
import torch


class TestHTMLContentExtractor:
    """Test HTML parsing, script/style stripping, and paragraph extraction."""

    def test_basic_html_extraction(self):
        sample_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Transformer Neural Networks - Guide</title>
            <script>console.log("ignore me");</script>
            <style>body { color: red; }</style>
        </head>
        <body>
            <nav><a href="/">Home</a><a href="/about">About</a></nav>
            <article>
                <h1>Attention Is All You Need</h1>
                <p>The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.</p>
                <p>We propose the Transformer, a model architecture eschewing recurrence and relying entirely on an attention mechanism.</p>
            </article>
            <footer>Copyright 2026 Robbie AI</footer>
        </body>
        </html>
        """
        extractor = HTMLContentExtractor()
        extractor.feed(sample_html)
        title, text = extractor.get_extracted_content()

        assert "Transformer Neural Networks - Guide" in title
        assert "console.log" not in text
        assert "color: red" not in text
        assert "dominant sequence transduction models" in text
        assert "eschewing recurrence" in text

    def test_entity_unescaping(self):
        sample_html = "<p>Robbie &amp; Python &quot;deep learning&quot; &#39;transformers&#39;</p>"
        extractor = HTMLContentExtractor()
        extractor.feed(sample_html)
        _, text = extractor.get_extracted_content()

        assert "Robbie & Python \"deep learning\" 'transformers'" in text


class TestSearchEngines:
    """Test search result models, parsers, and engines."""

    def test_search_result_dataclass(self):
        sr = SearchResult(
            title="PyTorch Documentation",
            url="https://pytorch.org",
            snippet="An open source machine learning framework.",
            source="test",
        )
        data = sr.to_dict()
        assert data["title"] == "PyTorch Documentation"
        assert data["url"] == "https://pytorch.org"
        assert data["source"] == "test"

    def test_duckduckgo_parser(self):
        sample_html = """
        <div class="result">
            <a class="result__url" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Ftransformers"></a>
            <a class="result__a" href="https://example.com/transformers">Transformers in Deep Learning</a>
            <a class="result__snippet">A comprehensive guide on self-attention and multi-head mechanisms.</a>
        </div>
        """
        parser = DuckDuckGoHTMLParser()
        parser.feed(sample_html)
        assert len(parser.results) >= 1
        assert parser.results[0].title == "Transformers in Deep Learning"
        assert "https://example.com/transformers" in parser.results[0].url

    @patch("urllib.request.urlopen")
    def test_wikipedia_search(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b"""{
            "query": {
                "search": [
                    {
                        "title": "Transformer (deep learning architecture)",
                        "pageid": 63973950,
                        "snippet": "A <span>transformer</span> is a deep learning architecture relying on self-attention."
                    }
                ]
            }
        }"""
        mock_urlopen.return_value.__enter__.return_value = mock_response

        engine = WikipediaSearchEngine()
        results = engine.search("transformer deep learning", max_results=3)

        assert len(results) == 1
        assert "Transformer" in results[0].title
        assert "relying on self-attention" in results[0].snippet
        assert "wikipedia.org" in results[0].url

    def test_unified_search_offline_fallback(self):
        engine = UnifiedSearchEngine()
        engine.wiki_engine.search = MagicMock(return_value=[])
        engine.web_engine.search = MagicMock(return_value=[])

        results = engine.search("neural network training", max_results=3)
        assert len(results) >= 1
        assert results[0].source == "local"
        assert "Offline Reference" in results[0].title


class TestWebpageScraper:
    """Test safe page fetching and character budgeting."""

    def test_invalid_url_protocol(self):
        scraper = WebpageScraper()
        doc = scraper.fetch("ftp://example.com/file.txt")
        assert doc.status_code == 400
        assert "Invalid URL protocol" in doc.error

    @patch("urllib.request.urlopen")
    def test_fetch_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.headers.get.return_value = "text/html; charset=utf-8"
        mock_resp.headers.get_content_charset.return_value = "utf-8"
        mock_resp.read.return_value = b"<html><head><title>Test Page</title></head><body><p>Deep learning models process tokens using causal attention.</p></body></html>"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        scraper = WebpageScraper()
        doc = scraper.fetch("https://example.com/article", max_chars=1000)

        assert doc.status_code == 200
        assert doc.title == "Test Page"
        assert "process tokens using causal attention" in doc.text


class TestResearchSynthesizer:
    """Test query tokenization, sentence scoring, and citation generation."""

    def test_tokenize_query(self):
        tokens = ResearchSynthesizer.tokenize_query("What is validation loss in machine learning?")
        assert "validation" in tokens
        assert "loss" in tokens
        assert "machine" in tokens
        assert "learning" in tokens
        assert "what" not in tokens
        assert "is" not in tokens
        assert "in" not in tokens

    def test_score_sentence(self):
        keywords = {"validation", "loss", "overfitting"}
        good_sent = "Validation loss evaluates generalization and reveals when overfitting occurs."
        poor_sent = "The weather in Seattle is quite rainy during the autumn months."

        score_good = ResearchSynthesizer.score_sentence(good_sent, keywords)
        score_poor = ResearchSynthesizer.score_sentence(poor_sent, keywords)

        assert score_good > score_poor
        assert score_poor == 0.0

    def test_synthesis_and_citations(self):
        synthesizer = ResearchSynthesizer()
        sources = [
            SearchResult(
                title="Deep Learning Optimization",
                url="https://example.com/opt",
                snippet="Validation loss evaluates generalization on held-out test data.",
                source="web",
            ),
            SearchResult(
                title="Cross-Entropy in PyTorch",
                url="https://example.com/ce",
                snippet="Cross-entropy loss measures the distance between predictions and targets.",
                source="web",
            ),
        ]
        docs = [
            WebpageDocument(
                url="https://example.com/opt",
                title="Deep Learning Optimization",
                text="Validation loss detects when training diverges or when overfitting begins. Early stopping monitors validation loss.",
                char_count=120,
            )
        ]

        result = synthesizer.synthesize("validation loss", sources=sources, documents=docs)

        assert "findings" in result
        assert "citations" in result
        assert len(result["citations"]) == 2
        assert any("[1]" in f for f in result["findings"])


class TestResearchAssistant:
    """Test end-to-end ResearchAssistant workflow and report generation."""

    def test_research_report_markdown(self):
        report = ResearchReport(
            query="Transformer Attention",
            summary="Transformers use multi-head attention to model relationships between tokens in parallel.",
            key_findings=[
                "Self-attention computes query, key, and value matrices [1].",
                "Causal masking prevents tokens from attending to future tokens [2].",
            ],
            citations=[
                {"index": 1, "title": "Attention Paper", "url": "https://arxiv.org/abs/1706.03762", "source": "arxiv"},
                {"index": 2, "title": "PyTorch Tutorial", "url": "https://pytorch.org/tutorials", "source": "web"},
            ],
            sources_count=2,
            latency_sec=0.45,
        )
        md = report.to_markdown()

        assert "### Research Report: Transformer Attention" in md
        assert "2 sources consulted in 0.45s" in md
        assert "Self-attention computes query, key, and value matrices [1]" in md
        assert "[1] Attention Paper (Arxiv) — https://arxiv.org/abs/1706.03762" in md

    def test_research_assistant_pipeline(self):
        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(
                title="Validation Loss Guide",
                url="https://example.com/val",
                snippet="Validation loss indicates generalization accuracy on unseen datasets.",
                source="wikipedia",
            )
        ]
        mock_scraper = MagicMock()
        mock_scraper.fetch.return_value = WebpageDocument(
            url="https://example.com/val",
            title="Validation Loss Guide",
            text="Validation loss measures error on test sets to detect overfitting.",
            char_count=80,
        )

        ra = ResearchAssistant(search_engine=mock_search, scraper=mock_scraper)
        report = ra.research("validation loss", max_sources=2)

        assert report.query == "validation loss"
        assert report.sources_count == 1
        assert len(report.citations) == 1
        assert len(report.key_findings) >= 1


class TestChatEngineResearchCommands:
    """Test /research and /search in ChatEngine."""

    def test_search_command(self):
        mock_model = MagicMock()
        tok = CharTokenizer()
        tok.train(["User: Robbie:"])
        engine = ChatEngine(model=mock_model, tokenizer=tok, device=torch.device("cpu"))

        # Empty query
        res_empty = engine.respond("/search")
        assert "Usage: /search" in res_empty

        with patch("robbie.research.assistant.ResearchAssistant.quick_search") as mock_qs:
            mock_qs.return_value = [
                SearchResult(title="PyTorch Docs", url="https://pytorch.org", snippet="Deep learning library", source="web")
            ]
            res = engine.respond("/search pytorch")
            assert "Web search results for 'pytorch'" in res
            assert "PyTorch Docs" in res
            assert "https://pytorch.org" in res

    def test_research_command(self):
        mock_model = MagicMock()
        tok = CharTokenizer()
        tok.train(["User: Robbie:"])
        engine = ChatEngine(model=mock_model, tokenizer=tok, device=torch.device("cpu"))

        res_empty = engine.respond("/research")
        assert "Usage: /research" in res_empty

        with patch("robbie.research.assistant.ResearchAssistant.research") as mock_r:
            mock_r.return_value = ResearchReport(
                query="BPE Tokenization",
                summary="Byte-pair encoding is a subword tokenization method.",
                key_findings=["BPE iteratively merges frequent character pairs [1]."],
                citations=[{"index": 1, "title": "BPE Guide", "url": "https://example.com/bpe", "source": "web"}],
                sources_count=1,
                latency_sec=0.25,
            )
            res = engine.respond("/research BPE Tokenization")
            assert "Research Report: BPE Tokenization" in res
            assert "BPE iteratively merges frequent character pairs [1]" in res
            assert "[1] BPE Guide" in res


class TestResearchAccuracyAndValidation:
    """Rigorous tests for research accuracy, recency, ranking, citation validation, and HTML formatting."""

    def test_citation_validator_grounded_vs_ungrounded(self):
        sources = [
            SearchResult(
                title="Stage 2 Training Logs",
                url="https://robbie.local/training",
                snippet="Robbie achieved optimal validation loss 1.2942 with perplexity 3.65 at epoch 1.",
                source="web",
            ),
            SearchResult(
                title="Generic Machine Learning",
                url="https://example.com/ml",
                snippet="Neural networks are optimized using stochastic gradient descent and Adam.",
                source="web",
            )
        ]

        # 1. Grounded assertion with matching number
        grounded_claim = "The optimal validation loss reached 1.2942 at epoch 1."
        res1 = CitationValidator.validate_claim(grounded_claim, sources)
        assert res1.is_grounded is True
        assert 1 in res1.matched_sources

        # 2. Number mismatch / ungrounded claim
        mismatched_claim = "The optimal validation loss reached 0.0543 at epoch 99."
        res2 = CitationValidator.validate_claim(mismatched_claim, sources)
        assert 1 not in res2.matched_sources

        # 3. Completely hallucinated claim
        hallucinated = "Quantum superconducting qubits performed telepathic teleportation in 1890."
        res3 = CitationValidator.validate_claim(hallucinated, sources)
        assert res3.is_grounded is False
        assert len(res3.matched_sources) == 0

    def test_source_ranking_authority_and_recency(self):
        engine = UnifiedSearchEngine()
        candidates = [
            SearchResult(
                title="Deep Learning Breakthroughs 2025",
                url="https://arxiv.org/abs/2501.99999",
                snippet="A 2025 survey of state of the art transformer architectures and self-attention.",
                source="web",
                published_date="2025-01-15",
                authority_score=1.6,
            ),
            SearchResult(
                title="Old Notes on AI",
                url="https://free-ad-click.com/ai",
                snippet="Transformer architectures from 2017.",
                source="web",
                published_date="2017-06-01",
                authority_score=0.7,
            ),
        ]

        ranked = engine.rank_sources(candidates, "latest 2025 transformer architectures")
        assert len(ranked) == 2
        # arXiv 2025 should rank far above outdated ad click site
        assert ranked[0].url == "https://arxiv.org/abs/2501.99999"
        assert ranked[0].score > ranked[1].score
        assert ranked[1].is_outdated is True

    def test_url_sanitization(self):
        from robbie.research.search import normalize_and_clean_url
        raw_ddg = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fpytorch.org%2Fdocs%3Futm_source%3Dtwitter%26utm_medium%3Dsocial&rut=1"
        clean = normalize_and_clean_url(raw_ddg)
        assert "duckduckgo.com" not in clean
        assert "utm_source" not in clean
        assert "utm_medium" not in clean
        assert clean == "https://pytorch.org/docs"

    def test_html_report_formatting(self):
        report = ResearchReport(
            query="Transformer Attention & Heads",
            summary="Multi-head attention allows joint modeling of diverse representations.",
            key_findings=[
                "Self-attention computes Q, K, and V matrices [1].",
                "FlashAttention speeds up training by reducing HBM memory reads [1, 2].",
            ],
            citations=[
                {
                    "index": 1,
                    "title": "Attention Paper",
                    "url": "https://arxiv.org/abs/1706.03762",
                    "source": "arxiv",
                    "published_date": "2017-06-12",
                    "is_outdated": False,
                },
                {
                    "index": 2,
                    "title": "FlashAttention Paper",
                    "url": "https://arxiv.org/abs/2205.14135",
                    "source": "arxiv",
                    "published_date": "2022-05-27",
                    "is_outdated": False,
                },
            ],
            sources_count=2,
            latency_sec=0.35,
            citation_validation_rate=1.0,
        )

        html_out = report.to_html()
        assert "<article class=\"robbie-research-report\">" in html_out
        assert "<h2>Research Report: Transformer Attention &amp; Heads</h2>" in html_out
        assert "Grounded Citations: 100% verified" in html_out
        assert "<sup><a href=\"#source-1\">[1]</a></sup>" in html_out
        assert "<sup><a href=\"#source-1\">[1]</a>, <a href=\"#source-2\">[2]</a></sup>" in html_out
        assert "<li id=\"source-1\">" in html_out
        assert "<li id=\"source-2\">" in html_out

    def test_html_boilerplate_stripping(self):
        html_with_boilerplate = """
        <html>
        <head><title>Clean AI Research</title></head>
        <body>
            <div class="banner">Accept all cookies to continue to our website.</div>
            <p>Subscribe to our newsletter for daily spam updates.</p>
            <article>
                <p>Causal transformers use lower-triangular attention masks to prevent information leakage.</p>
            </article>
            <div class="footer">All rights reserved 2026.</div>
        </body>
        </html>
        """
        extractor = HTMLContentExtractor()
        extractor.feed(html_with_boilerplate)
        title, text = extractor.get_extracted_content()

        assert title == "Clean AI Research"
        assert "Accept all cookies" not in text
        assert "Subscribe to our newsletter" not in text
        assert "All rights reserved" not in text
        assert "Causal transformers use lower-triangular attention masks" in text

