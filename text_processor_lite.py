import re
import nltk
from typing import List

class TextProcessorLite:
    def __init__(self):
        """Initialize the lightweight text processor."""
        # Try to download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except:
                print("Warning: NLTK punkt not available, using fallback tokenization")
    
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
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities using simple regex patterns.
        This should be called BEFORE preprocessing to preserve capitalization.
        
        Args:
            text: Input text (before preprocessing)
            
        Returns:
            List of potential named entities
        """
        # Find capitalized words (potential names)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter out common words that aren't names
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'within', 'without',
            'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when',
            'why', 'how', 'what', 'who', 'which', 'whose', 'whom', 'she', 'he',
            'they', 'we', 'you', 'i', 'it', 'as', 'if', 'so', 'then', 'now',
            'just', 'very', 'much', 'more', 'most', 'some', 'any', 'all', 'each',
            'every', 'no', 'not', 'never', 'always', 'often', 'sometimes',
            'suddenly', 'slowly', 'quickly', 'carefully', 'gently', 'softly',
            'loudly', 'quietly', 'easily', 'hardly', 'nearly', 'almost', 'quite',
            'rather', 'too', 'also', 'only', 'even', 'still', 'yet', 'again',
            'once', 'twice', 'first', 'last', 'next', 'previous', 'current',
            'old', 'new', 'young', 'big', 'small', 'large', 'tiny', 'huge',
            'good', 'bad', 'great', 'terrible', 'wonderful', 'awful', 'beautiful',
            'ugly', 'clean', 'dirty', 'bright', 'dark', 'light', 'heavy',
            'hot', 'cold', 'warm', 'cool', 'fresh', 'stale', 'sweet', 'sour',
            'dust', 'air', 'water', 'fire', 'earth', 'wind', 'rain', 'snow',
            'sun', 'moon', 'star', 'sky', 'cloud', 'tree', 'flower', 'grass'
        }
        
        entities = [ent for ent in entities if ent.lower() not in common_words]
        
        return list(set(entities))  # Remove duplicates
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK or fallback method.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of sentences
        """
        try:
            # Try NLTK first
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback to simple regex-based tokenization
            sentences = self._fallback_tokenize_sentences(text)
        
        # Filter out very short sentences (likely noise)
        sentences = [sent for sent in sentences if len(sent.split()) > 2]
        
        return sentences
    
    def _fallback_tokenize_sentences(self, text: str) -> List[str]:
        """
        Fallback sentence tokenization using regex.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split by common sentence endings
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
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
