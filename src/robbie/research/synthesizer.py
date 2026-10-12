"""Multi-source research synthesis, fact extraction, and citation building (Phase 8)."""

from dataclasses import dataclass
import re
from typing import List, Dict, Set, Tuple, Any

from robbie.research.search import SearchResult
from robbie.research.scraper import WebpageDocument


@dataclass
class ScoredEvidence:
    """A scored snippet of evidence linked to its parent source."""
    sentence: str
    source_index: int
    source_title: str
    source_url: str
    relevance_score: float


class ResearchSynthesizer:
    """Extracts, ranks, and synthesizes key findings with academic-style citations."""

    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in", "on",
        "for", "with", "at", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
        "will", "would", "shall", "should", "what", "which", "who", "whom", "this",
        "that", "these", "those", "am", "it", "its", "how", "tell", "me", "about"
    }

    @classmethod
    def tokenize_query(cls, query: str) -> Set[str]:
        """Extract significant lowercase keywords from user query."""
        tokens = re.findall(r"\b[A-Za-z0-9_]+\b", query.lower())
        return {t for t in tokens if len(t) > 2 and t not in cls.STOPWORDS}

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """Split text into clean, bounded sentences."""
        raw_splits = re.split(r"(?<=[.!?])\s+", text)
        sentences = []
        for s in raw_splits:
            cleaned = s.strip()
            # Retain informative sentences of reasonable length
            if 30 <= len(cleaned) <= 300 and not cleaned.startswith("http"):
                sentences.append(cleaned)
        return sentences

    @classmethod
    def score_sentence(cls, sentence: str, query_keywords: Set[str]) -> float:
        """Score sentence relevance based on query term overlap and information density."""
        if not sentence or not query_keywords:
            return 0.0

        sent_words = set(re.findall(r"\b[A-Za-z0-9_]+\b", sentence.lower()))
        matched = sent_words.intersection(query_keywords)
        if not matched:
            return 0.0

        # Term coverage ratio
        coverage = len(matched) / len(query_keywords)
        # Slight bonus for sentences that aren't overly repetitive
        density = len(matched) / max(1, len(sent_words))

        return (coverage * 2.0) + (density * 1.0)

    def synthesize(
        self,
        query: str,
        sources: List[SearchResult],
        documents: List[WebpageDocument],
        max_findings: int = 5,
    ) -> Dict[str, Any]:
        """Synthesize multiple sources and documents into coherent evidence and citations."""
        query_keywords = self.tokenize_query(query)
        evidence_pool: List[ScoredEvidence] = []

        # 1. Collect evidence from SearchResult snippets
        for idx, src in enumerate(sources, 1):
            if src.snippet:
                sentences = self.split_sentences(src.snippet) or [src.snippet]
                for sent in sentences:
                    score = self.score_sentence(sent, query_keywords) + (0.5 if idx <= 2 else 0.2)
                    evidence_pool.append(
                        ScoredEvidence(
                            sentence=sent,
                            source_index=idx,
                            source_title=src.title,
                            source_url=src.url,
                            relevance_score=score,
                        )
                    )

        # 2. Collect evidence from scraped full text
        doc_map = {doc.url: doc for doc in documents if doc.text and not doc.error}
        for idx, src in enumerate(sources, 1):
            doc = doc_map.get(src.url)
            if doc and doc.text:
                sentences = self.split_sentences(doc.text)
                for sent in sentences:
                    score = self.score_sentence(sent, query_keywords)
                    if score > 0.0:
                        evidence_pool.append(
                            ScoredEvidence(
                                sentence=sent,
                                source_index=idx,
                                source_title=src.title,
                                source_url=src.url,
                                relevance_score=score,
                            )
                        )

        # 3. Sort evidence by relevance score
        evidence_pool.sort(key=lambda e: e.relevance_score, reverse=True)

        # 4. Deduplicate findings
        selected_findings: List[ScoredEvidence] = []
        seen_stems: List[Set[str]] = []

        for ev in evidence_pool:
            words = set(re.findall(r"\b[A-Za-z0-9]+\b", ev.sentence.lower()))
            # Check overlap with already selected findings
            too_similar = False
            for prev_words in seen_stems:
                jaccard = len(words.intersection(prev_words)) / max(1, len(words.union(prev_words)))
                if jaccard > 0.60:
                    too_similar = True
                    break

            if not too_similar:
                selected_findings.append(ev)
                seen_stems.append(words)
                if len(selected_findings) >= max_findings:
                    break

        # Fallback if no specific keyword match was found
        if not selected_findings and sources:
            for idx, src in enumerate(sources[:max_findings], 1):
                if src.snippet:
                    selected_findings.append(
                        ScoredEvidence(
                            sentence=src.snippet,
                            source_index=idx,
                            source_title=src.title,
                            source_url=src.url,
                            relevance_score=0.5,
                        )
                    )

        # 5. Format citations
        formatted_findings: List[str] = [
            f"{f.sentence} [{f.source_index}]" for f in selected_findings
        ]

        citations: List[Dict[str, Any]] = [
            {
                "index": idx,
                "title": src.title,
                "url": src.url,
                "source": src.source,
            }
            for idx, src in enumerate(sources, 1)
        ]

        # 6. Build high-level summary overview
        if selected_findings:
            top_texts = [f.sentence for f in selected_findings[:2]]
            summary_paragraph = " ".join(top_texts)
        else:
            summary_paragraph = f"No detailed sources found matching '{query}'."

        return {
            "query": query,
            "summary": summary_paragraph,
            "findings": formatted_findings,
            "citations": citations,
        }
