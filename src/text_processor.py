"""
Advanced text processing for speech synthesis.
Handles phonemization, normalization, and multi-language support.
"""
import re
import unicodedata
from typing import List, Dict, Optional, Tuple
import string
import inflect
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from unidecode import unidecode


class TextProcessor:
    """Advanced text processor for TTS systems."""
    
    def __init__(self, 
                 language: str = "en",
                 phoneme_backend: str = "espeak",
                 normalize_numbers: bool = True,
                 normalize_abbreviations: bool = True,
                 add_punctuation_pause: bool = True):
        """
        Initialize text processor.
        
        Args:
            language: Language code (en, es, fr, de, etc.)
            phoneme_backend: Phonemizer backend (espeak, festival)
            normalize_numbers: Whether to normalize numbers to words
            normalize_abbreviations: Whether to expand abbreviations
            add_punctuation_pause: Whether to add pauses for punctuation
        """
        self.language = language
        self.phoneme_backend = phoneme_backend
        self.normalize_numbers = normalize_numbers
        self.normalize_abbreviations = normalize_abbreviations
        self.add_punctuation_pause = add_punctuation_pause
        
        # Initialize components
        self._init_phonemizer()
        self._init_normalizers()
        self._init_language_specific()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def _init_phonemizer(self):
        """Initialize phonemizer backend."""
        if self.phoneme_backend == "espeak":
            self.phonemizer = EspeakBackend(
                language=self.language,
                preserve_punctuation=True,
                with_stress=True
            )
    
    def _init_normalizers(self):
        """Initialize text normalizers."""
        if self.normalize_numbers:
            self.inflect_engine = inflect.engine()
        
        # Common abbreviations
        self.abbreviations = {
            "dr.": "doctor",
            "mr.": "mister", 
            "mrs.": "misses",
            "ms.": "miss",
            "prof.": "professor",
            "st.": "saint",
            "vs.": "versus",
            "etc.": "etcetera",
            "inc.": "incorporated",
            "ltd.": "limited",
            "co.": "company",
            "corp.": "corporation",
            "dept.": "department",
            "gov.": "government",
            "min.": "minutes",
            "max.": "maximum",
            "avg.": "average",
            "approx.": "approximately",
            "temp.": "temperature",
        }
        
        # Time patterns
        self.time_patterns = [
            (re.compile(r'\b(\d{1,2}):(\d{2})\s*(am|pm)\b', re.IGNORECASE), self._normalize_time),
            (re.compile(r'\b(\d{1,2}):(\d{2})\b'), self._normalize_24h_time),
        ]
        
        # Date patterns
        self.date_patterns = [
            (re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b'), self._normalize_date_slash),
            (re.compile(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b'), self._normalize_date_dash),
        ]
        
        # Currency patterns
        self.currency_patterns = [
            (re.compile(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)\b'), self._normalize_dollars),
            (re.compile(r'€(\d+(?:,\d{3})*(?:\.\d{2})?)\b'), self._normalize_euros),
            (re.compile(r'£(\d+(?:,\d{3})*(?:\.\d{2})?)\b'), self._normalize_pounds),
        ]
        
        # Number patterns
        self.number_patterns = [
            (re.compile(r'\b(\d+)\.(\d+)\b'), self._normalize_decimal),
            (re.compile(r'\b(\d+),(\d{3})\b'), self._normalize_thousands),
            (re.compile(r'\b(\d+)%\b'), self._normalize_percentage),
            (re.compile(r'\b(\d+)\b'), self._normalize_integer),
        ]
    
    def _init_language_specific(self):
        """Initialize language-specific components."""
        try:
            if self.language == "en":
                self.nlp = spacy.load("en_core_web_sm")
            elif self.language == "es":
                self.nlp = spacy.load("es_core_news_sm")
            elif self.language == "fr":
                self.nlp = spacy.load("fr_core_news_sm")
            elif self.language == "de":
                self.nlp = spacy.load("de_core_news_sm")
            else:
                self.nlp = None
        except OSError:
            self.nlp = None
    
    def _normalize_time(self, match) -> str:
        """Normalize time format (12-hour)."""
        hour, minute, period = match.groups()
        hour_word = self.inflect_engine.number_to_words(int(hour))
        minute_word = self.inflect_engine.number_to_words(int(minute)) if int(minute) != 0 else ""
        
        if minute_word:
            return f"{hour_word} {minute_word} {period.lower()}"
        else:
            return f"{hour_word} {period.lower()}"
    
    def _normalize_24h_time(self, match) -> str:
        """Normalize 24-hour time format."""
        hour, minute = match.groups()
        hour_word = self.inflect_engine.number_to_words(int(hour))
        minute_word = self.inflect_engine.number_to_words(int(minute)) if int(minute) != 0 else ""
        
        if minute_word:
            return f"{hour_word} {minute_word}"
        else:
            return f"{hour_word} o'clock"
    
    def _normalize_date_slash(self, match) -> str:
        """Normalize date in MM/DD/YYYY format."""
        month, day, year = match.groups()
        month_names = ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
        
        month_name = month_names[int(month) - 1]
        day_word = self.inflect_engine.ordinal(int(day))
        
        return f"{month_name} {day_word} {year}"
    
    def _normalize_date_dash(self, match) -> str:
        """Normalize date in YYYY-MM-DD format."""
        year, month, day = match.groups()
        month_names = ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
        
        month_name = month_names[int(month) - 1]
        day_word = self.inflect_engine.ordinal(int(day))
        
        return f"{month_name} {day_word} {year}"
    
    def _normalize_dollars(self, match) -> str:
        """Normalize dollar amounts."""
        amount = match.group(1).replace(',', '')
        if '.' in amount:
            dollars, cents = amount.split('.')
            dollars_word = self.inflect_engine.number_to_words(int(dollars))
            cents_word = self.inflect_engine.number_to_words(int(cents))
            return f"{dollars_word} dollars and {cents_word} cents"
        else:
            dollars_word = self.inflect_engine.number_to_words(int(amount))
            return f"{dollars_word} dollars"
    
    def _normalize_euros(self, match) -> str:
        """Normalize euro amounts."""
        amount = match.group(1).replace(',', '')
        if '.' in amount:
            euros, cents = amount.split('.')
            euros_word = self.inflect_engine.number_to_words(int(euros))
            cents_word = self.inflect_engine.number_to_words(int(cents))
            return f"{euros_word} euros and {cents_word} cents"
        else:
            euros_word = self.inflect_engine.number_to_words(int(amount))
            return f"{euros_word} euros"
    
    def _normalize_pounds(self, match) -> str:
        """Normalize pound amounts."""
        amount = match.group(1).replace(',', '')
        if '.' in amount:
            pounds, pence = amount.split('.')
            pounds_word = self.inflect_engine.number_to_words(int(pounds))
            pence_word = self.inflect_engine.number_to_words(int(pence))
            return f"{pounds_word} pounds and {pence_word} pence"
        else:
            pounds_word = self.inflect_engine.number_to_words(int(amount))
            return f"{pounds_word} pounds"
    
    def _normalize_decimal(self, match) -> str:
        """Normalize decimal numbers."""
        integer_part, decimal_part = match.groups()
        integer_word = self.inflect_engine.number_to_words(int(integer_part))
        decimal_word = " ".join([self.inflect_engine.number_to_words(int(d)) for d in decimal_part])
        return f"{integer_word} point {decimal_word}"
    
    def _normalize_thousands(self, match) -> str:
        """Normalize numbers with thousand separators."""
        number = match.group(0).replace(',', '')
        return self.inflect_engine.number_to_words(int(number))
    
    def _normalize_percentage(self, match) -> str:
        """Normalize percentage."""
        number = match.group(1)
        number_word = self.inflect_engine.number_to_words(int(number))
        return f"{number_word} percent"
    
    def _normalize_integer(self, match) -> str:
        """Normalize integer numbers."""
        number = match.group(1)
        return self.inflect_engine.number_to_words(int(number))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle special characters
        text = unidecode(text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        return text
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        if not self.normalize_abbreviations:
            return text
        
        words = text.split()
        expanded_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.abbreviations:
                expanded_words.append(self.abbreviations[word_lower])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def normalize_numbers_and_dates(self, text: str) -> str:
        """Normalize numbers, dates, times, and currency."""
        if not self.normalize_numbers:
            return text
        
        # Apply all normalization patterns
        for pattern, normalizer in (self.time_patterns + self.date_patterns + 
                                   self.currency_patterns + self.number_patterns):
            text = pattern.sub(normalizer, text)
        
        return text
    
    def add_punctuation_pauses(self, text: str) -> str:
        """Add pauses for punctuation marks."""
        if not self.add_punctuation_pause:
            return text
        
        # Add pauses for major punctuation
        text = re.sub(r'[.!?]+', r'\g<0> <PAUSE>', text)
        text = re.sub(r'[,;:]+', r'\g<0> <SHORT_PAUSE>', text)
        
        return text
    
    def phonemize_text(self, text: str, preserve_punctuation: bool = True) -> str:
        """Convert text to phonemes."""
        try:
            phonemes = phonemize(
                text,
                language=self.language,
                backend=self.phoneme_backend,
                preserve_punctuation=preserve_punctuation,
                with_stress=True
            )
            return phonemes
        except Exception as e:
            # Fallback to original text if phonemization fails
            return text
    
    def process_text(self, text: str, 
                    use_phonemes: bool = False,
                    preserve_punctuation: bool = True) -> str:
        """
        Complete text processing pipeline.
        
        Args:
            text: Input text
            use_phonemes: Whether to convert to phonemes
            preserve_punctuation: Whether to preserve punctuation
            
        Returns:
            Processed text
        """
        # Step 1: Clean text
        text = self.clean_text(text)
        
        # Step 2: Expand abbreviations
        text = self.expand_abbreviations(text)
        
        # Step 3: Normalize numbers and dates
        text = self.normalize_numbers_and_dates(text)
        
        # Step 4: Add punctuation pauses
        text = self.add_punctuation_pauses(text)
        
        # Step 5: Phonemize if requested
        if use_phonemes:
            text = self.phonemize_text(text, preserve_punctuation)
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = sent_tokenize(text)
        return [self.process_text(sentence) for sentence in sentences]
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize processed text into units suitable for TTS."""
        # Process text first
        processed_text = self.process_text(text)
        
        # Simple word-level tokenization
        tokens = word_tokenize(processed_text)
        
        return tokens
    
    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """Get statistics about the text."""
        processed_text = self.process_text(text)
        tokens = self.tokenize_text(text)
        
        stats = {
            'original_length': len(text),
            'processed_length': len(processed_text),
            'num_sentences': len(self.split_into_sentences(text)),
            'num_tokens': len(tokens),
            'num_words': len([t for t in tokens if t.isalpha()]),
            'num_punctuation': len([t for t in tokens if t in string.punctuation])
        }
        
        return stats


# Factory function
def create_text_processor(config: Dict) -> TextProcessor:
    """Create text processor from configuration."""
    return TextProcessor(
        language=config.get('language', 'en'),
        phoneme_backend=config.get('phoneme_backend', 'espeak'),
        normalize_numbers=config.get('normalize_numbers', True),
        normalize_abbreviations=config.get('normalize_abbreviations', True),
        add_punctuation_pause=config.get('add_punctuation_pause', True)
    )
