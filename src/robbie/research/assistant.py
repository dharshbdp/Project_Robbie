"""ResearchAssistant orchestrating search, webpage extraction, multi-source synthesis, and citations (Phase 8)."""

from dataclasses import dataclass, field
from datetime import datetime
import html
import re
import time
from typing import List, Dict, Any, Optional, Tuple

from robbie.research.search import SearchResult, UnifiedSearchEngine
from robbie.research.scraper import WebpageScraper, WebpageDocument
from robbie.research.synthesizer import ResearchSynthesizer, CitationValidator
from robbie.utils.logger import logger


def sanitize_display_text(text: str) -> str:
    """Sanitize text to safe readable ASCII/standard characters, preventing terminal charmap errors."""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2022": "-",
        "\u2026": "...",
        "\u200b": "",
        "\ufeff": "",
        "\u2191": "^",
        "\u2193": "v",
        "\u2192": "->",
        "\u2190": "<-",
        "↑": "^",
        "↓": "v",
        "→": "->",
        "←": "<-",
        "†": "",
        "‡": "",
        "•": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("ascii", errors="ignore").decode("ascii")


@dataclass
class SearchIntent:
    """Structured research and retrieval plan parsed from a user query."""
    search_query: str
    category: str = "general"
    target_year: Optional[str] = None
    target_count: Optional[int] = None
    is_list_request: bool = False
    entity: Optional[str] = None
    media_type: str = "items"
    original_topic: str = ""


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
            f"### Research Report: {sanitize_display_text(self.query)}",
            f"*{self.sources_count} sources consulted in {self.latency_sec:.2f}s | Citations: {rate_percent}% verified*\n",
            f"**Overview:**\n{sanitize_display_text(self.summary)}\n",
        ]

        if self.key_findings:
            lines.append("**Key Findings:**")
            for f in self.key_findings:
                lines.append(f"- {sanitize_display_text(f)}")
            lines.append("")

        if self.citations:
            lines.append("**Sources & Citations:**")
            for c in self.citations:
                idx = c["index"]
                title = sanitize_display_text(c["title"])
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

    @classmethod
    def parse_query_intent(cls, query: str) -> SearchIntent:
        """Parse user query into structured search intent, recognizing lists, counts, and entities."""
        cleaned = query.strip().strip("\"' ")
        if not cleaned:
            return SearchIntent(search_query="", category="general")

        # 1. Normalize contractions and common typos
        contractions = {
            r"\bcant\b": "cannot",
            r"\bcan't\b": "cannot",
            r"\bdont\b": "do not",
            r"\bdon't\b": "do not",
            r"\bwont\b": "will not",
            r"\bwon't\b": "will not",
            r"\bisnt\b": "is not",
            r"\bisn't\b": "is not",
            r"\barent\b": "are not",
            r"\baren't\b": "are not",
            r"\bdidnt\b": "did not",
            r"\bdidn't\b": "did not",
            r"\bdoesnt\b": "does not",
            r"\bdoesn't\b": "does not",
            r"\bwhats\b": "what is",
            r"\bwhat's\b": "what is",
            r"\bwheres\b": "where is",
            r"\bwhere's\b": "where is",
            r"\bhows\b": "how is",
            r"\bhow's\b": "how is",
            r"\brelesed\b": "released",
        }
        lower_q = cleaned.lower()
        for pat, repl in contractions.items():
            lower_q = re.sub(pat, repl, lower_q)

        # 2. Strip conversational command and request prefixes
        stripped_q = re.sub(
            r"^(?:(?:can\s+you\s+|please\s+)?(?:do\s+(?:a\s+)?)?"
            r"(?:research\s+(?:on|about|and\s+give|and\s+list)?|"
            r"search\s+(?:the\s+web\s+)?(?:for)?|"
            r"look\s+up|find\s+(?:out\s+about)?|"
            r"name|list|give(?:\s+me)?|tell\s+me(?:\s+about)?|"
            r"what\s+are|show\s+me|provide)\s+)+",
            "",
            lower_q,
            flags=re.IGNORECASE,
        ).strip()
        if not stripped_q:
            stripped_q = lower_q

        curr_year = str(datetime.now().year)
        target_year = None
        m_year = re.search(r"\b(19\d\d|20\d\d)\b", stripped_q)
        if m_year:
            target_year = m_year.group(1)

        # Check for requested count (e.g. "10 nolan movies", "top 5 games")
        target_count = None
        m_cnt = re.search(r"\b(\d{1,2})\b", stripped_q)
        if m_cnt:
            target_count = int(m_cnt.group(1))

        is_list = (
            target_count is not None
            or any(w in lower_q for w in ("name", "list", "give", "top", "best", "some", "several"))
            or stripped_q.endswith(("movies", "films", "games", "albums", "songs", "books"))
        )

        DIRECTORS = {
            "nolan": "Christopher Nolan",
            "christopher nolan": "Christopher Nolan",
            "tarantino": "Quentin Tarantino",
            "quentin tarantino": "Quentin Tarantino",
            "scorsese": "Martin Scorsese",
            "martin scorsese": "Martin Scorsese",
            "spielberg": "Steven Spielberg",
            "steven spielberg": "Steven Spielberg",
            "cameron": "James Cameron",
            "james cameron": "James Cameron",
            "fincher": "David Fincher",
            "david fincher": "David Fincher",
            "fincher": "David Fincher",
            "david fincher": "David Fincher",
            "kubrick": "Stanley Kubrick",
            "stanley kubrick": "Stanley Kubrick",
            "villeneuve": "Denis Villeneuve",
            "denis villeneuve": "Denis Villeneuve",
            "hitchcock": "Alfred Hitchcock",
            "alfred hitchcock": "Alfred Hitchcock",
            "miyazaki": "Hayao Miyazaki",
            "hayao miyazaki": "Hayao Miyazaki",
            "ridley scott": "Ridley Scott",
            "wes anderson": "Wes Anderson",
            "coen": "Coen brothers",
            "edgar wright": "Edgar Wright",
            "guillermo del toro": "Guillermo del Toro",
        }

        STUDIOS = {
            "fromsoftware": "FromSoftware",
            "from software": "FromSoftware",
            "rockstar": "Rockstar Games",
            "rockstar games": "Rockstar Games",
            "bethesda": "Bethesda Game Studios",
            "kojima": "Hideo Kojima",
            "hideo kojima": "Hideo Kojima",
            "kojima productions": "Kojima Productions",
            "miyamoto": "Shigeru Miyamoto",
            "shigeru miyamoto": "Shigeru Miyamoto",
            "naughty dog": "Naughty Dog",
            "valve": "Valve",
            "capcom": "Capcom",
            "square enix": "Square Enix",
            "nintendo": "Nintendo",
            "bioware": "BioWare",
            "blizzard": "Blizzard Entertainment",
            "cd projekt": "CD Projekt Red",
            "cd projekt red": "CD Projekt Red",
        }

        # Check for movies by creator / director
        if any(w in stripped_q for w in ("movie", "movies", "film", "films")):
            for alias, full_name in DIRECTORS.items():
                if re.search(r"\b" + re.escape(alias) + r"\b", stripped_q):
                    return SearchIntent(
                        search_query=f"{full_name} filmography",
                        category="movies_list",
                        target_year=target_year,
                        target_count=target_count or 10,
                        is_list_request=True,
                        entity=full_name,
                        media_type="movies",
                        original_topic=cleaned,
                    )
            # Generic person director movies
            m_pers = re.search(r"([A-Za-z\s]+?)\s+(?:movies|films)", stripped_q)
            if m_pers and len(m_pers.group(1).strip().split()) <= 3:
                name_cand = m_pers.group(1).strip().title()
                if not any(k in name_cand.lower() for k in ("new", "latest", "recent", "horror", "action", "sci-fi", "comedy", "animated")):
                    return SearchIntent(
                        search_query=f"{name_cand} filmography",
                        category="movies_list",
                        target_year=target_year,
                        target_count=target_count or 10,
                        is_list_request=True,
                        entity=name_cand,
                        media_type="movies",
                        original_topic=cleaned,
                    )
            if is_list:
                clean_term = re.sub(r"\b\d+\b", "", stripped_q).strip()
                return SearchIntent(
                    search_query=f"{clean_term} list",
                    category="movies_list",
                    target_year=target_year,
                    target_count=target_count or 10,
                    is_list_request=True,
                    media_type="movies",
                    original_topic=cleaned,
                )

        # Check for video games by studio or creator
        if any(w in stripped_q for w in ("game", "games", "gaming")):
            for alias, full_name in STUDIOS.items():
                if re.search(r"\b" + re.escape(alias) + r"\b", stripped_q):
                    return SearchIntent(
                        search_query=f"{full_name} video games",
                        category="games_list",
                        target_year=target_year,
                        target_count=target_count or 10,
                        is_list_request=True,
                        entity=full_name,
                        media_type="video games",
                        original_topic=cleaned,
                    )
            if is_list:
                clean_term = re.sub(r"\b\d+\b", "", stripped_q).strip()
                return SearchIntent(
                    search_query=f"{clean_term} video games",
                    category="games_list",
                    target_year=target_year,
                    target_count=target_count or 10,
                    is_list_request=True,
                    media_type="video games",
                    original_topic=cleaned,
                )

        # Awards
        if "goty" in lower_q or "game of the year" in lower_q:
            year_str = target_year if target_year else ""
            return SearchIntent(
                search_query=f"The Game Awards {year_str} Game of the Year winner".strip(),
                category="award",
                target_year=target_year,
                original_topic=cleaned,
            )
        if "oscar" in lower_q or "academy award" in lower_q or "best picture" in lower_q:
            year_str = target_year if target_year else ""
            return SearchIntent(
                search_query=f"Academy Awards {year_str} Best Picture winner".strip(),
                category="award",
                target_year=target_year,
                original_topic=cleaned,
            )

        # Rankings
        if "imdb" in lower_q and any(w in lower_q for w in ("top", "best", "highest", "list")):
            return SearchIntent(
                search_query="IMDb Top 250",
                category="ranking",
                target_count=target_count or 10,
                is_list_request=True,
                media_type="movies",
                original_topic=cleaned,
            )

        # Games news
        if any(w in lower_q for w in ("game", "games", "gaming")):
            if any(w in lower_q for w in ("update", "updates", "patch", "patches", "news", "today")):
                return SearchIntent(
                    search_query="video game updates",
                    category="games",
                    original_topic=cleaned,
                )
            elif target_year:
                return SearchIntent(
                    search_query=f"{target_year} in video games",
                    category="games",
                    target_year=target_year,
                    original_topic=cleaned,
                )
            elif any(w in lower_q for w in ("new", "latest", "recent", "released", "upcoming")):
                return SearchIntent(
                    search_query="new video games releases",
                    category="games",
                    original_topic=cleaned,
                )
            else:
                return SearchIntent(
                    search_query=f"{stripped_q} video games",
                    category="games",
                    original_topic=cleaned,
                )

        # Movie releases
        if any(w in lower_q for w in ("movie", "movies", "film", "films")):
            if any(w in lower_q for w in ("new", "latest", "recent", "released", "upcoming")):
                return SearchIntent(
                    search_query=f"{target_year or curr_year} in film",
                    category="movies",
                    target_year=target_year or curr_year,
                    original_topic=cleaned,
                )
            else:
                return SearchIntent(
                    search_query=stripped_q,
                    category="movies",
                    original_topic=cleaned,
                )

        # General open-domain queries
        stopwords = {
            "why", "how", "what", "which", "where", "when", "who", "whom", "whose",
            "can", "cannot", "could", "do", "does", "did", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "the", "a", "an", "and", "or",
            "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
            "about", "into", "over", "after", "it", "its", "they", "them", "their",
            "this", "that", "these", "those", "tell", "me", "please", "explain",
        }
        tokens = re.findall(r"\b[A-Za-z0-9_]+\b", stripped_q)
        content_tokens = [t for t in tokens if t not in stopwords]
        has_temporal = any(
            w in lower_q
            for w in ("new", "latest", "recent", "recently", "upcoming", "released", "release")
        )
        if any(stripped_q.startswith(w) for w in ("why", "how", "where", "when", "what", "is", "can")):
            sq = " ".join(content_tokens) if content_tokens else stripped_q
        elif has_temporal and curr_year not in stripped_q:
            sq = f"{stripped_q} {curr_year}"
        else:
            sq = stripped_q

        return SearchIntent(
            search_query=sq,
            category="general",
            target_year=target_year,
            target_count=target_count,
            is_list_request=is_list,
            original_topic=cleaned,
        )

    @classmethod
    def formulate_search_query(cls, query: str) -> Tuple[str, str, Optional[str]]:
        """Normalize typos, expand chat contractions, and formulate clean search query."""
        intent = cls.parse_query_intent(query)
        return intent.search_query, intent.category, intent.target_year

    def extract_structured_list_items(
        self,
        documents: List[WebpageDocument],
        category: str,
        target_count: int = 10,
        entity: Optional[str] = None,
    ) -> List[Tuple[str, str, str]]:
        """Extract orderly item records (title, release year, description) from scraped pages."""
        items: List[Tuple[str, str, str]] = []
        seen_titles: Set[str] = set()

        bad_keywords = {
            "award", "awards", "box office", "gross", "total", "imdb", "rotten tomatoes",
            "metacritic", "wikipedia", "united states", "warner bros", "paramount",
            "universal pictures", "sony pictures", "columbia pictures", "marvel",
            "director", "producers", "screenplay", "release date", "filmography",
            "discography", "bibliography", "note", "notes", "references", "external links"
        }
        if entity:
            for part in entity.lower().split():
                if len(part) > 2:
                    bad_keywords.add(part)

        non_directorial_terms = {
            "assistant cameraman", "matte photography", "photographic effects",
            "special photographic effects", "visual effects", "cameo", "extra",
            "himself", "uncredited", "soundtrack", "music video", "music videos",
            "executive producer only", "acting role", "acting roles", "actor", "idea spy"
        }

        # 1. First pass: extract from HTML tables in raw_html
        for doc in documents:
            if not doc.raw_html:
                continue
            tables = re.findall(r'<table[^>]*>([\s\S]*?)</table>', doc.raw_html)
            for tbl in tables:
                # If table header indicates acting or crew credits, skip when searching for directed movies or games
                header_text = " ".join(re.findall(r'<th[^>]*>(.*?)</th>', tbl, re.IGNORECASE)).lower()
                clean_header = re.sub(r'<[^>]+>', ' ', header_text)
                if category in ("movies_list", "games_list"):
                    if any(bad_hdr in clean_header for bad_hdr in ("role", "actor", "cast", "music video", "other credit", "acting")):
                        continue

                rows = re.findall(r'<tr[^>]*>[\s\S]*?</tr>', tbl)
                for r in rows:
                    r_low = re.sub(r'<[^>]+>', ' ', r).lower()
                    if any(term in r_low for term in non_directorial_terms):
                        continue

                    m_yr = re.search(r'<(?:th|td)[^>]*>\s*(19\d\d|20\d\d)\s*</(?:th|td)>', r)
                    m_title = re.search(r'<i[^>]*>(?:<span[^>]*>)?(?:<a[^>]*>)?([^<]+)', r)
                    if m_yr and m_title:
                        yr = m_yr.group(1)
                        t = html.unescape(m_title.group(1).strip())
                        t_clean = re.sub(r"\s+", " ", t).strip()
                        t_low = t_clean.lower()
                        cols = re.findall(r'<td[^>]*>(?:<[^>]+>)?(Yes|No)', r, re.IGNORECASE)
                        if cols and cols[0].lower() == 'no':
                            continue
                        if len(t_clean) >= 2 and t_low not in seen_titles:
                            if not any(bad == t_low for bad in bad_keywords):
                                seen_titles.add(t_low)
                                items.append((t_clean, yr, ""))

            # 2. Extract from italic links with year: <i><a ...>Title</a></i> (YYYY)
            for m in re.finditer(r'<i[^>]*>(?:<a[^>]*>)?([^<]+)(?:</a>)?</i>\s*\((19\d\d|20\d\d)\)', doc.raw_html):
                surrounding = doc.raw_html[max(0, m.start() - 60):min(len(doc.raw_html), m.end() + 60)].lower()
                if any(term in surrounding for term in ("cameo", "music video", "assistant cameraman")):
                    continue
                t = html.unescape(m.group(1).strip())
                yr = m.group(2)
                t_clean = re.sub(r"\s+", " ", t).strip()
                t_low = t_clean.lower()
                if len(t_clean) >= 2 and t_low not in seen_titles:
                    if not any(bad == t_low for bad in bad_keywords):
                        seen_titles.add(t_low)
                        items.append((t_clean, yr, ""))

        # 3. Third pass: extract from text blocks and narrative sentences if table items < 3
        if len(items) < 3:
            for doc in documents:
                if not doc.text:
                    continue
                for line in doc.text.split("\n"):
                    matches = re.finditer(r'\b([A-Z][A-Za-z0-9\s:\'-]+?)\s*\((19\d\d|20\d\d)\)', line)
                    for m in matches:
                        cand_title = m.group(1).strip()
                        yr = m.group(2)
                        cand_title = re.sub(
                            r'^(?:his\s+|the\s+|he\s+|directed\s+(?:the\s+)?|feature\s+|debut\s+|psychological\s+|mystery\s+|thriller\s+|superhero\s+|film\s+|movie\s+|remake\s+|action\s+|science\s+fiction\s+|sci-fi\s+|sequel\s+(?:to\s+)?)+',
                            '', cand_title, flags=re.IGNORECASE
                        ).strip()
                        cand_low = cand_title.lower()
                        if len(cand_title) >= 2 and cand_low not in seen_titles:
                            if not any(bad in cand_low for bad in bad_keywords):
                                seen_titles.add(cand_low)
                                items.append((cand_title, yr, ""))

        # 4. Enrich items with 1-line descriptions from doc text
        combined_text = "\n".join(doc.text for doc in documents if doc.text)
        enriched: List[Tuple[str, str, str]] = []
        for title, yr, _ in items:
            desc = ""
            for line in combined_text.split("\n"):
                if title.lower() in line.lower():
                    sentences = re.split(r"(?<=[.!?])\s+", line)
                    for s in sentences:
                        s_clean = s.strip()
                        if title.lower() in s_clean.lower() and 25 <= len(s_clean) <= 220:
                            clean_desc = re.sub(r"\[\s*(?:note\s*\d+|[a-z]|\d+|update|edit)\s*\]", "", s_clean).strip()
                            clean_desc = re.sub(r"\s+", " ", clean_desc)
                            if any(frag in clean_desc.lower() for frag in ("yes yes", "yes no", "no no", "retrieved", "archived", "ref.", "http")):
                                continue
                            if clean_desc and not clean_desc.lower().startswith("retrieved"):
                                desc = clean_desc
                                break
                    if desc:
                        break
            enriched.append((title, yr, desc))

        return enriched[:target_count]

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

        # 0. Formulate clean search query
        search_query, _, _ = self.formulate_search_query(query)

        # 1. Search for sources with composite ranking and recency scoring
        sources = self.search_engine.search(search_query, max_results=max_sources, timeout=timeout)

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

    def answer_query(
        self,
        query: str,
        max_sources: int = 3,
        timeout: float = 4.0,
        include_sources: bool = True,
    ) -> str:
        """Search the web, extract salient facts, summarize them directly, and return clean conversational markdown.

        Tailored for natural conversational questions (e.g. 'what are the new games that released?', 'which game won goty 2019?').
        """
        cleaned = query.strip().strip("\"' ")
        if not cleaned:
            return ""

        intent = self.parse_query_intent(cleaned)
        search_query = intent.search_query
        category = intent.category
        target_year = intent.target_year
        curr_year = str(datetime.now().year)
        lower_q = cleaned.lower()

        # 2. Search across open web and reference engines
        raw_sources = self.search_engine.search(search_query, max_results=max_sources + 3, timeout=timeout)
        if not raw_sources:
            return (
                "I attempted to look that up on the web, but couldn't retrieve live results right now. "
                "Please verify your internet connection."
            )

        # Filter sources to ensure domain alignment
        sources: List[SearchResult] = []
        for s in raw_sources:
            t_low = s.title.lower()
            if category == "games" and any(m in t_low for m in ("(film)", "(movie)", "(2026 film)", "american films")):
                continue
            if category == "movies" and any(g in t_low for g in ("(video game)", "(game)")):
                continue
            if not any(existing.url == s.url for existing in sources):
                sources.append(s)

        if not sources:
            sources = raw_sources

        # Prioritize year-matching sources
        if target_year:
            sources.sort(key=lambda s: (target_year in s.title, target_year in s.snippet), reverse=True)

        # If this is an orderly list request (e.g. movies, games, top rankings)
        if intent.is_list_request:
            list_docs: List[WebpageDocument] = []
            for src in sources[:3]:
                if src.url.startswith("http://") or src.url.startswith("https://"):
                    doc = self.scraper.fetch(src.url, timeout=timeout, max_chars=8000)
                    if doc.text and not doc.error:
                        list_docs.append(doc)

            items = self.extract_structured_list_items(
                documents=list_docs,
                category=category,
                target_count=intent.target_count or 10,
                entity=intent.entity,
            )
            if len(items) >= 2:
                entity_desc = f" directed by **{intent.entity}**" if intent.entity else (f" for **{intent.original_topic}**" if intent.original_topic else "")
                lines = [f"Here are {len(items)} notable {intent.media_type}{entity_desc} in chronological order:\n"]
                for i, (title, yr, desc) in enumerate(items, 1):
                    yr_part = f" ({yr})" if yr else ""
                    desc_part = f" — {desc}" if desc else ""
                    lines.append(f"{i}. **{title}**{yr_part}{desc_part}\n")

                if include_sources and sources:
                    lines.append("\n**Sources:**")
                    added_urls = set()
                    added_titles = set()
                    for s in sources:
                        clean_t = sanitize_display_text(s.title).strip()
                        clean_t_low = clean_t.lower()
                        if s.url not in added_urls and clean_t_low not in added_titles and not s.url.startswith("local://"):
                            added_urls.add(s.url)
                            added_titles.add(clean_t_low)
                            lines.append(f"- [{clean_t}]({s.url})")
                        if len(added_titles) >= max_sources:
                            break
                return sanitize_display_text("\n".join(lines))

        # 3. Fetch top 2 scrapeable pages
        documents: List[WebpageDocument] = []
        for src in sources:
            if len(documents) >= 2:
                break
            if "news.google.com" in src.url:
                continue
            if src.url.startswith("http://") or src.url.startswith("https://"):
                doc = self.scraper.fetch(src.url, timeout=timeout, max_chars=3500)
                if doc.text and not doc.error:
                    lower_doc = doc.text.lower()
                    if len(doc.text) >= 200 and "sign in to your" not in lower_doc:
                        documents.append(doc)

        # 4. Extract high-signal bullet points
        q_tokens = set(re.findall(r"\b[A-Za-z0-9_]+\b", lower_q)) - {
            "which", "what", "are", "the", "that", "is", "for", "in", "to", "of", "and", "a", "an", "on", "with",
            "why", "how", "can", "cant", "cannot"
        }
        # Morphological expansions across related lexical variants
        expanded_tokens = set(q_tokens)
        for t in q_tokens:
            if t.endswith("s") and len(t) > 3:
                expanded_tokens.add(t[:-1])
            elif not t.endswith("s"):
                expanded_tokens.add(t + "s")
            if t in ("fly", "flying", "flies"):
                expanded_tokens.update(["flight", "flightless"])
            elif t in ("swim", "swimming", "swims"):
                expanded_tokens.update(["swimmer", "aquatic"])
        q_tokens = expanded_tokens

        candidates = []

        # From scraped documents
        for doc in documents:
            for p in doc.text.split("\n"):
                sentences = re.split(r"(?<=[.!?])\s+", p) if ("." in p or "!" in p or "?" in p) else [p]
                for raw_line in sentences:
                    line = re.sub(r"\s+", " ", raw_line).strip()
                    if len(line) < 20 or len(line) > 280:
                        continue
                    # Skip Wikipedia footnotes and citations
                    if line.startswith(("^", "↑")) or re.match(r"^\d+\.\s*\^", line):
                        continue
                    if any(junk in line.lower() for junk in (
                        "archived from the original", "retrieved from", "external links",
                        "redirects to", "redirect to", "this page redirects", "may refer to",
                        "disambiguation", "other uses"
                    )):
                        continue
                    # Skip pure theater location lines
                    if any(th in line.lower() for th in ("odeon luxe", "amc theatres venue")):
                        continue
                    lower_line = line.lower()
                    if lower_line.startswith(("list of ", "the following is", "information regarding", "see also", "main article", "background color indicates", "the top highest", "the following table", "the original versions", "table lists")):
                        continue
                    # Skip platform acronym tables
                    if any(abbr in lower_line for abbr in ("ps5 playstation", "lin linux", "ns2 nintendo", "win windows", "osx macos", "playstation 5 psvr")):
                        continue

                    # Strip bracket citations [ 1 ], [ a ], [ update ]
                    cleaned_line = re.sub(r"\[\s*([a-z]|\d+|update|edit)\s*\]", "", line)
                    cleaned_line = sanitize_display_text(re.sub(r"\s+", " ", cleaned_line).strip())
                    if len(cleaned_line) < 20:
                        continue

                    words = set(re.findall(r"\b[A-Za-z0-9_]+\b", lower_line))
                    overlap = len(words.intersection(q_tokens))

                    bonus = 0.0
                    if target_year:
                        if target_year in lower_line or target_year in doc.title.lower():
                            bonus += 3.0
                        other_years = [y for y in re.findall(r"\b(19\d\d|20\d\d)\b", lower_line) if y != target_year]
                        if other_years:
                            bonus -= 2.0

                    if category == "award":
                        if any(w in lower_line for w in ("game of the year", "goty", "best picture", "winner", "awarded", "wins", "won", "accepted")):
                            bonus += 4.0
                    elif category in ("games", "movies"):
                        if any(k in lower_line for k in ("scoring 90", "metacritic", "top-rated", "releases", "launches", "scheduled", "premiered", "highest-grossing")):
                            bonus += 2.0
                    elif category == "ranking":
                        if any(k in lower_line for k in ("top", "rating", "highest", "ranked", "voted", "list", "poll")):
                            bonus += 2.0
                    elif any(w in lower_q for w in ("why", "how", "cause", "causes", "reason")):
                        if any(k in lower_line for k in ("because", "due to", "adapted to", "evolved", "cannot", "unable to", "lost the ability", "instead of", "flightless")):
                            bonus += 3.0

                    score = overlap + bonus
                    if score > 1.2:
                        candidates.append((score, cleaned_line))

        # From search snippets
        for r in sources:
            for s in re.split(r"(?<=[.!?])\s+", r.snippet):
                s = re.sub(r"\s+", " ", s).strip()
                if 25 <= len(s) <= 250:
                    lower_s = s.lower()
                    if any(junk in lower_s for junk in (
                        "list of ", "the following is", "information regarding", "see also",
                        "redirects to", "redirect to", "this page redirects", "may refer to",
                        "disambiguation", "other uses", "refer to:"
                    )):
                        continue
                    if s.startswith(("^", "↑")):
                        continue
                    clean_s = re.sub(r"\[\s*([a-z]|\d+|update|edit)\s*\]", "", s)
                    clean_s = sanitize_display_text(re.sub(r"\s+", " ", clean_s).strip())
                    words = set(re.findall(r"\b[A-Za-z0-9_]+\b", lower_s))
                    overlap = len(words.intersection(q_tokens))
                    bonus = 0.0
                    if target_year and target_year in lower_s:
                        bonus += 2.0
                    if category == "award" and any(w in lower_s for w in ("winner", "awarded", "wins", "game of the year", "goty")):
                        bonus += 3.0
                    elif category in ("games", "movies") and any(w in lower_s for w in ("releases", "launches", "scheduled", "premiered")):
                        bonus += 2.0
                    elif category == "ranking" and any(w in lower_s for w in ("top", "rating", "highest", "ranked", "list")):
                        bonus += 2.0
                    elif any(w in lower_q for w in ("why", "how", "cause", "causes", "reason")):
                        if any(k in lower_s for k in ("because", "due to", "adapted to", "evolved", "cannot", "unable to", "lost the ability", "instead of", "flightless")):
                            bonus += 3.0
                    score = overlap + bonus
                    if score > 1.2:
                        candidates.append((score, clean_s))

        # Sort and deduplicate
        candidates.sort(key=lambda x: x[0], reverse=True)
        unique_points: List[str] = []
        seen_words: List[set] = []
        for score, item in candidates:
            w_set = set(re.findall(r"\b[A-Za-z0-9_]+\b", item.lower()))
            is_dup = False
            for prev in seen_words:
                jaccard = len(w_set.intersection(prev)) / max(1, len(w_set.union(prev)))
                if jaccard > 0.40:
                    is_dup = True
                    break
            if not is_dup:
                seen_words.append(w_set)
                unique_points.append(item)
                if len(unique_points) >= 4:
                    break

        # Fallback if no points extracted
        if not unique_points and sources:
            for r in sources[:3]:
                if r.snippet:
                    unique_points.append(sanitize_display_text(r.snippet))

        # 5. Format conversational output
        active_year = target_year or curr_year
        if category == "award":
            lead = "Based on official award records and ceremony results:"
        elif category == "games":
            lead = f"Based on {active_year} video game release calendars and announcements, here are notable releases:"
        elif category == "movies":
            lead = f"Based on {active_year} film release schedules and announcements, here are notable releases:"
        elif category == "ranking":
            lead = f"Based on official rating records and rankings for **{cleaned.strip('?.!')}**:"
        elif any(w in lower_q for w in ("version", "update", "patch")):
            lead = f"Based on recent release data, here is the latest information regarding **{cleaned.strip('?.!')}**:"
        elif any(w in lower_q for w in ("news", "event", "happened", "headline")):
            lead = f"Based on recent reports, here is the latest update on **{cleaned.strip('?.!')}**:"
        else:
            lead = f"Here is what I found regarding **{cleaned.strip('?.!')}**:"

        lines: List[str] = [lead, ""]
        for p in unique_points:
            lines.append(f"- {p}")

        should_include_sources = include_sources or any(w in lower_q for w in ("source", "sources", "citation", "citations", "link", "links", "url"))
        if should_include_sources:
            lines.append("\n**Sources:**")
            added_links = set()
            for r in sources:
                if r.url not in added_links and not r.url.startswith("local://"):
                    added_links.add(r.url)
                    lines.append(f"- [{sanitize_display_text(r.title)}]({r.url})")
                if len(added_links) >= max_sources:
                    break

        return sanitize_display_text("\n".join(lines))

