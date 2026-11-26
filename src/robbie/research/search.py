"""Web search adapters, recency extraction, source ranking, and query execution for Project Robbie (Phase 8)."""

from dataclasses import dataclass, field
import datetime
import json
from html import unescape
from html.parser import HTMLParser
import re
import urllib.parse
import urllib.request
from typing import List, Optional, Dict, Any, Tuple, Set

from robbie.utils.logger import logger


@dataclass
class SearchResult:
    """Represents a single web or reference search result with authority and recency metadata."""
    title: str
    url: str
    snippet: str
    source: str = "web"
    score: float = 1.0
    published_date: Optional[str] = None
    is_outdated: bool = False
    authority_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "score": round(self.score, 3),
            "published_date": self.published_date,
            "is_outdated": self.is_outdated,
            "authority_score": round(self.authority_score, 3),
        }


def normalize_and_clean_url(url: str) -> str:
    """Sanitize URL by resolving redirects, unquoting, and stripping tracking parameters."""
    if not url:
        return ""

    # 1. DuckDuckGo redirect unwrapping
    if "uddg=" in url:
        try:
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            if "uddg" in parsed:
                url = urllib.parse.unquote(parsed["uddg"][0])
        except Exception:
            pass

    # 2. Protocol prefix fix
    if url.startswith("//"):
        url = "https:" + url

    # 3. Strip tracking parameters
    try:
        parsed_url = urllib.parse.urlparse(url)
        tracking_keys = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term",
            "utm_content", "fbclid", "gclid", "ref", "rut", "source"
        }
        qs = urllib.parse.parse_qs(parsed_url.query, keep_blank_values=False)
        clean_qs = {k: v for k, v in qs.items() if k.lower() not in tracking_keys}
        clean_query = urllib.parse.urlencode(clean_qs, doseq=True)

        url = urllib.parse.urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path.rstrip("/") if parsed_url.path != "/" else "/",
            parsed_url.params,
            clean_query,
            ""  # Strip fragment
        ))
    except Exception:
        pass

    return url


def extract_date_from_text(text: str) -> Optional[str]:
    """Extract publication or update date from snippet text."""
    if not text:
        return None

    # ISO date: YYYY-MM-DD
    m_iso = re.search(r"\b(20\d\d)[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b", text)
    if m_iso:
        return f"{m_iso.group(1)}-{m_iso.group(2)}-{m_iso.group(3)}"

    # Month DD, YYYY
    months = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    m_named = re.search(rf"\b({months})\s+(\d{{1,2}}),?\s+(20\d\d)\b", text, re.IGNORECASE)
    if m_named:
        return f"{m_named.group(1).capitalize()} {m_named.group(2)}, {m_named.group(3)}"

    # Relative time indicator
    m_rel = re.search(r"\b(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago\b", text, re.IGNORECASE)
    if m_rel:
        return m_rel.group(0).lower()

    # Plain year: 2015-2026
    m_yr = re.search(r"\b(20[12]\d)\b", text)
    if m_yr:
        return m_yr.group(1)

    return None


def calculate_domain_authority(url: str, source: str) -> float:
    """Compute authority weight based on domain reputation and source type."""
    if source == "wikipedia":
        return 1.4
    if source == "local":
        return 1.0

    domain = ""
    try:
        domain = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        pass

    # High-authority academic, government, documentation domains
    if any(domain.endswith(tld) for tld in (".edu", ".gov", ".org", "arxiv.org", "github.com", "python.org", "pytorch.org")):
        return 1.6
    # Reputable news / encyclopedia / tech / gaming platforms
    if any(kw in domain for kw in (
        "wikipedia", "reuters", "nature", "sciencedirect", "ieee", "acm", "huggingface", "developer.",
        "gamespot", "ign", "polygon", "pcgamer", "dexerto", "theverge", "arstechnica", "cnet", "wired", "bloomberg"
    )):
        return 1.4
    # Content farms or low-signal URL fragments
    if any(spam in domain for spam in ("click", "buzz", "free", "ad")):
        return 0.7

    return 1.0


class DuckDuckGoHTMLParser(HTMLParser):
    """Parses search results from DuckDuckGo HTML Lite."""

    def __init__(self):
        super().__init__()
        self.results: List[SearchResult] = []
        self._current_tag: Optional[str] = None
        self._current_url: Optional[str] = None
        self._current_title: List[str] = []
        self._current_snippet: List[str] = []
        self._in_title_link = False
        self._in_snippet = False
        self._snippet_tag: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        attr_dict = dict(attrs)
        self._current_tag = tag

        if tag == "a":
            classes = attr_dict.get("class", "")
            href = attr_dict.get("href", "")
            if "result-link" in classes or "result__url" in classes or "result__a" in classes:
                self._in_title_link = True
                self._current_url = href
                self._current_title = []
            elif "result-snippet" in classes or "result__snippet" in classes:
                self._in_snippet = True
                self._snippet_tag = "a"
                self._current_snippet = []

        elif tag == "td":
            classes = attr_dict.get("class", "")
            if "result-snippet" in classes:
                self._in_snippet = True
                self._snippet_tag = "td"
                self._current_snippet = []

    def handle_endtag(self, tag: str):
        if tag == "a" and self._in_title_link:
            self._in_title_link = False
        elif self._in_snippet and (self._snippet_tag is None or tag == self._snippet_tag):
            self._in_snippet = False
            self._snippet_tag = None
            title = unescape(" ".join(self._current_title).strip())
            snippet = unescape(" ".join(self._current_snippet).strip())
            url = normalize_and_clean_url(self._current_url or "")

            if title and url:
                pub_date = extract_date_from_text(snippet) or extract_date_from_text(title)
                auth_score = calculate_domain_authority(url, "duckduckgo")
                self.results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="duckduckgo",
                        published_date=pub_date,
                        authority_score=auth_score,
                    )
                )
            self._current_url = None
            self._current_title = []
            self._current_snippet = []

    def handle_data(self, data: str):
        if self._in_title_link:
            self._current_title.append(data)
        elif self._in_snippet:
            self._current_snippet.append(data)


class GoogleNewsSearchEngine:
    """Fetches real-time, live news via Google News RSS search (zero captcha, high freshness)."""

    RSS_URL = "https://news.google.com/rss/search"

    def search(self, query: str, max_results: int = 5, timeout: float = 6.0) -> List[SearchResult]:
        params = {
            "q": query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        }
        url = f"{self.RSS_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        results: List[SearchResult] = []
        try:
            import xml.etree.ElementTree as ET
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                xml_data = resp.read()
                root = ET.fromstring(xml_data)
                items = root.findall(".//item")
                for item in items[: max_results * 2]:
                    title_elem = item.find("title")
                    link_elem = item.find("link")
                    pub_elem = item.find("pubDate")
                    desc_elem = item.find("description")

                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                    link = link_elem.text.strip() if link_elem is not None and link_elem.text else ""
                    pub_date_str = pub_elem.text.strip() if pub_elem is not None and pub_elem.text else None
                    desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""

                    if not title or not link:
                        continue

                    # Clean snippet from HTML description
                    clean_desc = unescape(re.sub(r"<[^>]+>", " ", desc).strip())
                    clean_desc = re.sub(r"\s+", " ", clean_desc)
                    snippet = clean_desc if len(clean_desc) > 20 else title

                    # Extract publication date
                    pub_date = None
                    if pub_date_str:
                        m_date = re.search(r"\d{1,2}\s+[A-Za-z]{3}\s+20\d\d", pub_date_str)
                        if m_date:
                            pub_date = m_date.group(0)
                        else:
                            pub_date = pub_date_str

                    auth_score = calculate_domain_authority(link, "googlenews")
                    results.append(
                        SearchResult(
                            title=title,
                            url=link,
                            snippet=snippet,
                            source="googlenews",
                            published_date=pub_date,
                            authority_score=auth_score * 1.3,
                        )
                    )
                    if len(results) >= max_results:
                        break
        except Exception as e:
            logger.warning(f"Google News search failed for '{query}': {e}")
        return results


class DuckDuckGoInstantEngine:
    """Queries DuckDuckGo Instant Answer API for direct definitions and facts without captcha."""

    API_URL = "https://api.duckduckgo.com/"

    def search(self, query: str, max_results: int = 5, timeout: float = 2.5) -> List[SearchResult]:
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        results: List[SearchResult] = []
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
                abstract = data.get("AbstractText") or data.get("Abstract")
                abstract_url = data.get("AbstractURL")
                heading = data.get("Heading") or query

                if abstract and abstract_url:
                    results.append(
                        SearchResult(
                            title=heading,
                            url=abstract_url,
                            snippet=abstract,
                            source="duckduckgo",
                            authority_score=1.4,
                        )
                    )

                for topic in data.get("RelatedTopics", [])[:max_results]:
                    if isinstance(topic, dict) and "Text" in topic and "FirstURL" in topic:
                        results.append(
                            SearchResult(
                                title=topic.get("Text", "")[:60],
                                url=topic.get("FirstURL", ""),
                                snippet=topic.get("Text", ""),
                                source="duckduckgo",
                                authority_score=1.2,
                            )
                        )
        except Exception as e:
            logger.warning(f"DuckDuckGo Instant search failed for '{query}': {e}")
        return results[:max_results]


class WikipediaSearchEngine:
    """Queries Wikipedia API for factual, encyclopedic information with redirect & disambiguation filtering."""

    API_URL = "https://en.wikipedia.org/w/api.php"

    def search(self, query: str, max_results: int = 5, timeout: float = 6.0) -> List[SearchResult]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srprop": "snippet|timestamp",
            "format": "json",
            "utf8": "1",
            "srlimit": str(max_results * 2),
        }
        url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ProjectRobbie/1.0 (https://github.com/dharshbdp/Project_Robbie; research-bot)"
            },
        )
        results: List[SearchResult] = []
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
                search_items = data.get("query", {}).get("search", [])
                for item in search_items:
                    title = item.get("title", "")
                    page_id = item.get("pageid", "")
                    raw_snippet = item.get("snippet", "")
                    raw_timestamp = item.get("timestamp", "")
                    clean_snippet = unescape(re.sub(r"<[^>]+>", "", raw_snippet).strip())
                    clean_snippet = re.sub(r"\s+", " ", clean_snippet)
                    lower_snip = clean_snippet.lower()

                    # Filter out useless redirect notices, disambiguation, and 'may refer to'
                    if any(junk in lower_snip for junk in (
                        "redirects to", "redirect to", "this page redirects",
                        "may refer to", "disambiguation", "other uses", "refer to:"
                    )):
                        continue

                    # Parse timestamp (e.g. 2024-03-15T12:00:00Z)
                    pub_date = None
                    if raw_timestamp and "T" in raw_timestamp:
                        pub_date = raw_timestamp.split("T")[0]

                    article_url = f"https://en.wikipedia.org/?curid={page_id}" if page_id else f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"
                    clean_url = normalize_and_clean_url(article_url)
                    auth_score = calculate_domain_authority(clean_url, "wikipedia")

                    results.append(
                        SearchResult(
                            title=title,
                            url=clean_url,
                            snippet=clean_snippet,
                            source="wikipedia",
                            published_date=pub_date,
                            authority_score=auth_score,
                        )
                    )
                    if len(results) >= max_results:
                        break
        except Exception as e:
            logger.warning(f"Wikipedia search failed for '{query}': {e}")
        return results


class DuckDuckGoSearchEngine:
    """Searches the open web via DuckDuckGo Lite / HTML."""

    SEARCH_URL = "https://lite.duckduckgo.com/lite/"
    FALLBACK_URL = "https://html.duckduckgo.com/html/"

    def search(self, query: str, max_results: int = 5, timeout: float = 6.0) -> List[SearchResult]:
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        for endpoint in [self.SEARCH_URL, self.FALLBACK_URL]:
            try:
                req = urllib.request.Request(endpoint, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    html_text = resp.read().decode("utf-8", errors="replace")
                    parser = DuckDuckGoHTMLParser()
                    parser.feed(html_text)
                    if parser.results:
                        return parser.results[:max_results]
            except Exception as e:
                logger.warning(f"DuckDuckGo search on {endpoint} failed for '{query}': {e}")
        return []


class UnifiedSearchEngine:
    """Combines real-time news, instant answers, open web, and Wikipedia with smart recency routing."""

    TEMPORAL_KEYWORDS = {
        "latest", "recent", "current", "new", "today", "todays", "now", "update", "updates",
        "release", "releases", "patch", "patches", "2025", "2026"
    }

    def __init__(self):
        self.news_engine = GoogleNewsSearchEngine()
        self.instant_engine = DuckDuckGoInstantEngine()
        self.wiki_engine = WikipediaSearchEngine()
        self.web_engine = DuckDuckGoSearchEngine()

    def rank_sources(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rank candidates using keyword relevance, domain authority, and recency."""
        q_tokens = [t.lower() for t in re.findall(r"\b[A-Za-z0-9_]+\b", query) if len(t) > 2]
        is_temporal_query = any(tk in self.TEMPORAL_KEYWORDS for tk in q_tokens)
        current_year = datetime.datetime.now().year

        for r in results:
            title_lower = r.title.lower()
            snippet_lower = r.snippet.lower()

            # 1. Relevance: title match has highest weight
            title_matches = sum(1 for tk in q_tokens if tk in title_lower)
            snippet_matches = sum(1 for tk in q_tokens if tk in snippet_lower)
            relevance = (title_matches * 2.5) + (snippet_matches * 1.2) + 0.5

            # Exact query match bonus
            if query.lower() in title_lower:
                relevance += 3.0
            elif query.lower() in snippet_lower:
                relevance += 1.5

            # 2. Recency Scoring & Outdated Detection
            recency_multiplier = 1.0
            if r.source == "googlenews":
                recency_multiplier = 2.0 if is_temporal_query else 1.1
            elif r.published_date:
                # Extract year
                yr_match = re.search(r"\b(20\d\d)\b", r.published_date)
                if yr_match:
                    pub_year = int(yr_match.group(1))
                    age_years = max(0, current_year - pub_year)
                    if age_years <= 1:
                        recency_multiplier = 1.6 if is_temporal_query else 1.2
                    elif age_years <= 3:
                        recency_multiplier = 1.1 if is_temporal_query else 1.05
                    elif age_years >= 5:
                        r.is_outdated = True
                        recency_multiplier = 0.3 if is_temporal_query else 0.8
                elif any(w in r.published_date.lower() for w in ("ago", "gmt", "utc")):
                    recency_multiplier = 1.8 if is_temporal_query else 1.2

            # 3. Composite score
            r.score = relevance * r.authority_score * recency_multiplier

        # Sort highest score first
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def search(self, query: str, max_results: int = 5, timeout: float = 6.0) -> List[SearchResult]:
        """Execute multi-backend search with deduplication and composite ranking."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        q_tokens = [t.lower() for t in re.findall(r"\b[A-Za-z0-9_]+\b", cleaned_query) if len(t) > 2]
        is_temporal = any(tk in self.TEMPORAL_KEYWORDS for tk in q_tokens)
        current_year = datetime.datetime.now().year

        raw_candidates: List[SearchResult] = []
        seen_urls = set()

        def add_candidates(candidates: List[SearchResult]):
            for r in candidates:
                norm_url = normalize_and_clean_url(r.url)
                if norm_url and norm_url not in seen_urls:
                    seen_urls.add(norm_url)
                    r.url = norm_url
                    raw_candidates.append(r)

        # 1. For temporal queries (e.g. today's gaming updates, latest releases), prioritize Google News RSS
        if is_temporal:
            news_res = self.news_engine.search(cleaned_query, max_results=max_results + 2, timeout=timeout)
            add_candidates(news_res)

        # 2. Query Instant Answer API
        instant_res = self.instant_engine.search(cleaned_query, max_results=max(2, max_results), timeout=timeout)
        add_candidates(instant_res)

        # 3. Query Wikipedia (for temporal queries, discard articles older than 3 years unless list/catalog)
        wiki_res = self.wiki_engine.search(cleaned_query, max_results=max(3, max_results), timeout=timeout)
        is_catalog_query = any(w in cleaned_query.lower() for w in ("list", "filmography", "discography", "games", "movies", "works"))
        for r in wiki_res:
            if is_temporal and not is_catalog_query and r.published_date:
                m_yr = re.search(r"\b(20\d\d)\b", r.published_date)
                if m_yr and int(m_yr.group(1)) < current_year - 2:
                    continue
            add_candidates([r])

        # 4. Fallback to web engine if candidates are low
        if len(raw_candidates) < max_results:
            web_res = self.web_engine.search(cleaned_query, max_results=max_results + 2, timeout=timeout)
            add_candidates(web_res)

        # 5. Rank and filter
        if raw_candidates:
            ranked = self.rank_sources(raw_candidates, cleaned_query)
            return ranked[:max_results]

        # 6. Offline fallback
        return [
            SearchResult(
                title=f"Offline Reference: {cleaned_query}",
                url="local://knowledge-base",
                snippet=f"Local search for '{cleaned_query}'. Robbie's local models and knowledge cover Python, transformers, neural networks, and algorithms.",
                source="local",
                score=1.0,
            )
        ]

