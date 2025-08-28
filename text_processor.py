import re
import nltk
from typing import List, Tuple
import spacy

class TextProcessor:
    def __init__(self):
        """Initialize the text processor with necessary models."""
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Load spaCy model for better sentence splitting
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Normalize and clean raw text input.
        
        Args:
            text: Raw input text (story, transcript, article, etc.)
            
        Returns:
            Cleaned and normalized text
        """
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove timestamps (common in transcripts)
        text = re.sub(r'\d{1,2}:\d{2}(:\d{2})?', '', text)
        
        # Remove common filler words and phrases
        filler_words = [
            r'\b(um|uh|er|ah|like|you know|i mean|sort of|kind of)\b',
            r'\b(so|well|basically|actually|literally)\b'
        ]
        
        for pattern in filler_words:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Remove special characters but keep punctuation for sentence splitting
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        
        return text.strip()
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using spaCy for better accuracy.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of sentences
        """
        if self.nlp is None:
            # Fallback to NLTK if spaCy is not available
            return nltk.sent_tokenize(text)
        
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Filter out very short sentences (likely noise)
        sentences = [sent for sent in sentences if len(sent.split()) > 2]
        
        return sentences
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for LLM compatibility.
        
        Args:
            text: Input text
            
        Returns:
            Approximate token count
        """
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities for potential character tracking.
        
        Args:
            text: Input text
            
        Returns:
            List of named entities
        """
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]
        return list(set(entities))  # Remove duplicates
