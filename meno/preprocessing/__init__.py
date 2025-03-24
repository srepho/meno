"""Text preprocessing module for cleaning and normalizing text data."""

from typing import List, Dict, Optional, Union

# Re-export key functions
from .spelling import correct_spelling, SpellingCorrector
from .acronyms import expand_acronyms, AcronymExpander
from .normalization import normalize_text, TextNormalizer
from .deduplication import deduplicate_text, TextDeduplicator

__all__ = [
    "correct_spelling",
    "expand_acronyms",
    "normalize_text",
    "deduplicate_text",
    "TextNormalizer",
    "SpellingCorrector",
    "AcronymExpander",
    "TextDeduplicator",
]