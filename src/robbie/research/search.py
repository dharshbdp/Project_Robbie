"""Web search adapters and query execution for Project Robbie (Phase 8)."""

from dataclasses import dataclass
import json
from html.parser import HTMLParser
import re
import urllib.parse
import urllib.request
from typing import List, Optional, Dict, Any, Tuple

from robbie.utils.logger import logger


@dataclass
class SearchResult:
    """Represents a single web or reference search result."""
    title: str
    url: str
    snippet: str
    source: str = "web"
    score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "score": self.score,
        }


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
                self._current_snippet = []

        elif tag == "td":
            classes = attr_dict.get("class", "")
            if "result-snippet" in classes:
                self._in_snippet = True
                self._current_snippet = []

    def handle_endtag(self, tag: str):
        if tag == "a" and self._in_title_link:
            self._in_title_link = False
        elif (tag in ("a", "td")) and self._in_snippet:
            self._in_snippet = False
            title = "".join(self._current_title).strip()
            snippet = "".join(self._current_snippet).strip()
            url = self._current_url or ""
            # Clean DuckDuckGo redirect url if applicable
            if "uddg=" in url:
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                if "uddg" in parsed:
                    url = parsed["uddg"][0]
            if title and url:
                self.results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="duckduckgo",
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


class WikipediaSearchEngine:
    """Queries Wikipedia API for factual, encyclopedic information."""

    API_URL = "https://en.wikipedia.org/w/api.php"

    def search(self, query: str, max_results: int = 5, timeout: float = 6.0) -> List[SearchResult]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": "1",
            "srlimit": str(max_results),
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
                    clean_snippet = re.sub(r"<[^>]+>", "", raw_snippet).strip()
                    article_url = f"https://en.wikipedia.org/?curid={page_id}" if page_id else f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"
                    results.append(
                        SearchResult(
                            title=f"{title} (Wikipedia)",
                            url=article_url,
                            snippet=clean_snippet,
                            source="wikipedia",
                        )
                    )
        except Exception as e:
            logger.warning(f"Wikipedia search failed for '{query}': {e}")
        return results


class DuckDuckGoSearchEngine:
    """Searches the open web via DuckDuckGo HTML."""

    SEARCH_URL = "https://html.duckduckgo.com/html/"

    def search(self, query: str, max_results: int = 5, timeout: float = 6.0) -> List[SearchResult]:
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        req = urllib.request.Request(
            self.SEARCH_URL,
            data=data,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                html_text = resp.read().decode("utf-8", errors="replace")
                parser = DuckDuckGoHTMLParser()
                parser.feed(html_text)
                return parser.results[:max_results]
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed for '{query}': {e}")
            return []


class UnifiedSearchEngine:
    """Combines open web search and Wikipedia reference sources with offline fallback."""

    def __init__(self):
        self.wiki_engine = WikipediaSearchEngine()
        self.web_engine = DuckDuckGoSearchEngine()

    def search(self, query: str, max_results: int = 5, timeout: float = 6.0) -> List[SearchResult]:
        """Execute multi-backend search with deduplication."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        results: List[SearchResult] = []
        seen_urls = set()

        # 1. Fetch encyclopedia results (reliable, clean, structured)
        wiki_res = self.wiki_engine.search(cleaned_query, max_results=max(2, max_results // 2), timeout=timeout)
        for r in wiki_res:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                results.append(r)

        # 2. Fetch general web search results
        web_res = self.web_engine.search(cleaned_query, max_results=max_results, timeout=timeout)
        for r in web_res:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                results.append(r)

        # 3. Offline fallback if no internet or both services fail
        if not results:
            results.append(
                SearchResult(
                    title=f"Offline Reference: {cleaned_query}",
                    url="local://knowledge-base",
                    snippet=f"Local search for '{cleaned_query}'. Robbie's local models and knowledge cover Python, transformers, neural networks, and algorithms.",
                    source="local",
                )
            )

        return results[:max_results]
