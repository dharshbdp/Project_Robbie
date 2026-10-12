"""Webpage content retrieval, HTML cleaning, and text extraction (Phase 8)."""

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
import re
import urllib.request
from typing import Optional, List, Tuple

from robbie.utils.logger import logger


@dataclass
class WebpageDocument:
    """Represents the extracted text and metadata of a fetched webpage."""
    url: str
    title: str
    text: str
    char_count: int
    status_code: int = 200
    error: Optional[str] = None


class HTMLContentExtractor(HTMLParser):
    """Robust HTML text extractor that strips scripts, styles, and chrome elements."""

    IGNORED_TAGS = {
        "script", "style", "nav", "footer", "header", "noscript",
        "svg", "aside", "form", "button", "iframe"
    }

    BLOCK_TAGS = {
        "p", "h1", "h2", "h3", "h4", "h5", "h6", "li",
        "div", "article", "section", "blockquote", "tr"
    }

    def __init__(self):
        super().__init__()
        self.title_parts: List[str] = []
        self.text_blocks: List[str] = []
        self._current_block: List[str] = []
        self._tag_stack: List[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        tag_lower = tag.lower()
        self._tag_stack.append(tag_lower)
        if tag_lower == "title":
            self._in_title = True
        elif tag_lower in self.BLOCK_TAGS:
            self._flush_block()

    def handle_endtag(self, tag: str):
        tag_lower = tag.lower()
        if self._tag_stack and self._tag_stack[-1] == tag_lower:
            self._tag_stack.pop()
        elif tag_lower in self._tag_stack:
            # Pop up to matching tag
            while self._tag_stack and self._tag_stack[-1] != tag_lower:
                self._tag_stack.pop()
            if self._tag_stack:
                self._tag_stack.pop()

        if tag_lower == "title":
            self._in_title = False
        elif tag_lower in self.BLOCK_TAGS:
            self._flush_block()

    def handle_data(self, data: str):
        if self._in_title:
            self.title_parts.append(data.strip())
            return

        if any(ignored in self._tag_stack for ignored in self.IGNORED_TAGS):
            return

        cleaned = data.strip()
        if cleaned:
            self._current_block.append(cleaned)

    def _flush_block(self):
        if self._current_block:
            block_str = " ".join(self._current_block)
            # Remove excessive inner spaces
            block_str = re.sub(r"\s+", " ", block_str).strip()
            if len(block_str) >= 20:  # Ignore trivial tiny snippets like 'Home', 'Menu'
                self.text_blocks.append(block_str)
            self._current_block = []

    def get_extracted_content(self) -> Tuple[str, str]:
        """Return (title, clean_text_paragraphs)."""
        self._flush_block()
        title = unescape(" ".join(self.title_parts).strip())
        text = unescape("\n\n".join(self.text_blocks).strip())
        return title, text


class WebpageScraper:
    """Safe HTTP requester and text extractor for research pages."""

    def __init__(self, user_agent: Optional[str] = None):
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )

    def fetch(self, url: str, timeout: float = 6.0, max_chars: int = 5000) -> WebpageDocument:
        """Fetch webpage HTML, extract text, and enforce character budget."""
        if not url.startswith("http://") and not url.startswith("https://"):
            return WebpageDocument(
                url=url,
                title="",
                text="",
                char_count=0,
                status_code=400,
                error=f"Invalid URL protocol: '{url}'",
            )

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw_status = getattr(response, "status", None) or getattr(response, "code", None)
                status = int(raw_status) if isinstance(raw_status, int) else 200
                content_type = response.headers.get("Content-Type", "").lower()
                if "text" not in content_type and "html" not in content_type:
                    return WebpageDocument(
                        url=url,
                        title="",
                        text="",
                        char_count=0,
                        status_code=status,
                        error=f"Unsupported non-text Content-Type: '{content_type}'",
                    )

                raw_bytes = response.read(150_000)  # Read up to 150KB
                encoding = response.headers.get_content_charset() or "utf-8"
                html_text = raw_bytes.decode(encoding, errors="replace")

                extractor = HTMLContentExtractor()
                extractor.feed(html_text)
                title, text = extractor.get_extracted_content()

                if not title:
                    # Fallback title from domain / URL
                    title = url.split("//")[-1].split("/")[0]

                if len(text) > max_chars:
                    text = text[:max_chars].rsplit(".", 1)[0] + "."

                return WebpageDocument(
                    url=url,
                    title=title,
                    text=text,
                    char_count=len(text),
                    status_code=status,
                )
        except Exception as e:
            logger.debug(f"Failed to fetch '{url}': {e}")
            return WebpageDocument(
                url=url,
                title="",
                text="",
                char_count=0,
                status_code=500,
                error=str(e),
            )
