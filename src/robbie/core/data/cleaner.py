"""Text normalization and cleaning utilities for Project Robbie datasets."""

import re
import unicodedata
from typing import List, Optional


DOCUMENT_DELIMITER = "<|endoftext|>"


class TextCleaner:
    """Cleans and standardizes raw text for language model training."""

    def __init__(self, document_delimiter: str = DOCUMENT_DELIMITER):
        self.document_delimiter = document_delimiter

    def normalize_unicode(self, text: str) -> str:
        """Apply NFKC normalization to ensure canonical character representations."""
        return unicodedata.normalize("NFKC", text)

    def clean_quotes_and_dashes(self, text: str) -> str:
        """Normalize smart quotes, curly apostrophes, and unusual dashes."""
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("‘", "'").replace("’", "'").replace("`", "'")
        text = text.replace("—", " - ").replace("–", " - ")
        return text

    def strip_control_characters(self, text: str) -> str:
        """Remove unprintable control characters while preserving newlines and tabs."""
        return "".join(ch for ch in text if ch in "\n\t\r" or not unicodedata.category(ch).startswith("C"))

    def collapse_whitespace(self, text: str) -> str:
        """Remove trailing spaces from lines and limit consecutive blank lines to at most two."""
        lines = [line.rstrip() for line in text.splitlines()]
        cleaned_text = "\n".join(lines)
        # Collapse 3 or more newlines into 2
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        return cleaned_text.strip()

    def clean(self, text: str) -> str:
        """Run full cleaning pipeline on a raw text string."""
        text = self.normalize_unicode(text)
        text = self.clean_quotes_and_dashes(text)
        text = self.strip_control_characters(text)
        text = self.collapse_whitespace(text)
        return text

    def join_documents(self, documents: List[str]) -> str:
        """Clean and concatenate multiple documents with standard boundary delimiters."""
        cleaned_docs = [self.clean(doc) for doc in documents if doc.strip()]
        delimiter = f"\n\n{self.document_delimiter}\n\n"
        return delimiter.join(cleaned_docs)
