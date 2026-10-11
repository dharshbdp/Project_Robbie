"""Data acquisition, cleaning, and preparation modules for Project Robbie."""

from robbie.core.data.cleaner import TextCleaner, DOCUMENT_DELIMITER
from robbie.core.data.sources import fetch_stage1_pretrain_data, fetch_stage2_dialogue_data
from robbie.core.data.preprocessor import DatasetPipeline, calculate_text_stats

__all__ = [
    "TextCleaner",
    "DOCUMENT_DELIMITER",
    "fetch_stage1_pretrain_data",
    "fetch_stage2_dialogue_data",
    "DatasetPipeline",
    "calculate_text_stats",
]
