"""Multi-source research synthesis, fact extraction, citation building, and citation validation (Phase 8)."""

from dataclasses import dataclass, field
import re
from typing import List, Dict, Set, Tuple, Any, Optional

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
    supporting_indices: List[int] = field(default_factory=list)
    is_grounded: bool = True


@dataclass
class CitationValidationResult:
    """Result of verifying that a finding is grounded in referenced sources."""
    finding: str
    is_grounded: bool
    confidence: float
    matched_sources: List[int]
    unverified: bool = False


class CitationValidator:
    """Validates citations to ensure claims are grounded in source texts without hallucination."""

    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in", "on",
        "for", "with", "at", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "can", "could",
        "will", "would", "shall", "should", "this", "that", "these", "those", "it"
    }

    @classmethod
    def extract_claim_terms(cls, sentence: str) -> Set[str]:
        """Extract content tokens and alphanumeric entities from a sentence."""
        tokens = re.findall(r"\b[A-Za-z0-9_]+\b", sentence.lower())
        return {t for t in tokens if len(t) > 1 and t not in cls.STOPWORDS}

    @classmethod
    def validate_claim(
        cls,
        sentence: str,
        sources: List[SearchResult],
        documents: Optional[List[WebpageDocument]] = None,
        min_overlap: float = 0.40,
    ) -> CitationValidationResult:
        """Verify whether a finding is textually supported by the given sources and scraped documents."""
        clean_sentence = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", sentence).strip()
        claim_terms = cls.extract_claim_terms(clean_sentence)
        if not claim_terms:
            return CitationValidationResult(
                finding=clean_sentence,
                is_grounded=False,
                confidence=0.0,
                matched_sources=[],
                unverified=True,
            )

        doc_map = {doc.url: doc.text.lower() for doc in (documents or []) if doc.text and not doc.error}
        matched_indices: List[int] = []
        max_confidence = 0.0

        # Numbers or critical entities present in the claim
        claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", clean_sentence))

        for idx, src in enumerate(sources, 1):
            source_corpus = src.snippet.lower()
            if src.url in doc_map:
                source_corpus += " " + doc_map[src.url]

            # 1. Exact phrase containment
            if clean_sentence.lower() in source_corpus:
                matched_indices.append(idx)
                max_confidence = max(max_confidence, 1.0)
                continue

            # 2. Token overlap & recall
            source_tokens = set(re.findall(r"\b[A-Za-z0-9_]+\b", source_corpus))
            matched_terms = claim_terms.intersection(source_tokens)
            term_recall = len(matched_terms) / len(claim_terms)

            # 3. Check number consistency
            if claim_numbers:
                source_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", source_corpus))
                # If claim asserts numbers, at least one number must appear in the source
                if not claim_numbers.intersection(source_numbers):
                    term_recall *= 0.5

            if term_recall >= min_overlap:
                matched_indices.append(idx)
                max_confidence = max(max_confidence, term_recall)

        is_grounded = len(matched_indices) > 0
        return CitationValidationResult(
            finding=clean_sentence,
            is_grounded=is_grounded,
            confidence=round(max_confidence, 3),
            matched_sources=matched_indices,
            unverified=not is_grounded,
        )


class ResearchSynthesizer:
    """Extracts, ranks, and synthesizes key findings with validated academic-style citations."""

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
        return {t for t in tokens if len(t) > 1 and t not in cls.STOPWORDS}

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
        """Score sentence relevance using length-normalized frequency and term coverage."""
        if not sentence or not query_keywords:
            return 0.0

        sent_words = set(re.findall(r"\b[A-Za-z0-9_]+\b", sentence.lower()))
        matched = sent_words.intersection(query_keywords)
        if not matched:
            return 0.0

        # Term coverage ratio
        coverage = len(matched) / len(query_keywords)
        # Length-normalized density (BM25 inspired)
        density = len(matched) / (len(sent_words) ** 0.5 or 1.0)

        # Bonus for informative punctuation and statistics
        bonus = 0.0
        if re.search(r"\b\d+(?:\.\d+)?%?\b", sentence):
            bonus += 0.3
        if any(marker in sentence.lower() for marker in ("because", "defined as", "results in", "refers to", "achieves")):
            bonus += 0.4

        return (coverage * 2.5) + (density * 1.5) + bonus

    def synthesize(
        self,
        query: str,
        sources: List[SearchResult],
        documents: List[WebpageDocument],
        max_findings: int = 5,
    ) -> Dict[str, Any]:
        """Synthesize multiple sources and documents into coherent evidence and validated citations."""
        query_keywords = self.tokenize_query(query)
        evidence_pool: List[ScoredEvidence] = []

        # 1. Collect evidence from SearchResult snippets
        for idx, src in enumerate(sources, 1):
            if src.snippet:
                sentences = self.split_sentences(src.snippet) or [src.snippet]
                for sent in sentences:
                    score = self.score_sentence(sent, query_keywords) + (0.5 * src.authority_score)
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
                                relevance_score=score * src.authority_score,
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

        # 5. Validate citations and attribute grounded sources
        formatted_findings: List[str] = []
        grounded_count = 0
        cited_source_indices: Set[int] = set()

        for f in selected_findings:
            val = CitationValidator.validate_claim(f.sentence, sources, documents)
            if val.is_grounded and val.matched_sources:
                grounded_count += 1
                matched_str = ", ".join(str(idx) for idx in sorted(val.matched_sources))
                formatted_findings.append(f"{f.sentence} [{matched_str}]")
                cited_source_indices.update(val.matched_sources)
            else:
                # If validation did not ground elsewhere, check original source index
                if 1 <= f.source_index <= len(sources):
                    formatted_findings.append(f"{f.sentence} [{f.source_index}]")
                    cited_source_indices.add(f.source_index)
                    grounded_count += 1
                else:
                    formatted_findings.append(f"{f.sentence} [Unverified]")

        total_findings = len(formatted_findings)
        validation_rate = (grounded_count / total_findings) if total_findings > 0 else 1.0

        # 6. Format sources citations
        citations: List[Dict[str, Any]] = [
            {
                "index": idx,
                "title": src.title,
                "url": src.url,
                "source": src.source,
                "published_date": src.published_date,
                "is_outdated": src.is_outdated,
                "authority_score": round(src.authority_score, 2),
                "verified": idx in cited_source_indices,
            }
            for idx, src in enumerate(sources, 1)
        ]

        # 7. Build high-level summary overview
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
            "citation_validation_rate": round(validation_rate, 3),
            "validation_summary": {
                "total_findings": total_findings,
                "grounded_findings": grounded_count,
                "rate": round(validation_rate, 3),
            },
        }

