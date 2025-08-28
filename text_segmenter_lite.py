from typing import List, Dict, Any
import numpy as np
from text_processor_lite import TextProcessorLite

class TextSegmenterLite:
    def __init__(self, max_tokens_per_chunk: int = 512, similarity_threshold: float = 0.6):
        """
        Initialize the lightweight text segmenter.
        
        Args:
            max_tokens_per_chunk: Maximum tokens per chunk for LLM compatibility
            similarity_threshold: Threshold for semantic similarity (0.0 to 1.0)
        """
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.similarity_threshold = similarity_threshold
        self.text_processor = TextProcessorLite()
    
    def segment_text_rule_based(self, text: str) -> List[str]:
        """
        Rule-based text segmentation (MVP approach).
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of text chunks
        """
        sentences = self.text_processor.tokenize_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.text_processor.count_tokens(sentence)
            
            # If adding this sentence would exceed the limit, start a new chunk
            if current_tokens + sentence_tokens > self.max_tokens_per_chunk and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def segment_text_semantic_simple(self, text: str) -> List[str]:
        """
        Simple semantic-based text segmentation using word overlap.
        
        Args:
            text: Preprocessed text
            
        Returns:
            List of text chunks
        """
        sentences = self.text_processor.tokenize_sentences(text)
        
        if len(sentences) < 2:
            return [text]
        
        chunks = []
        current_chunk = [sentences[0]]
        current_tokens = self.text_processor.count_tokens(sentences[0])
        
        for i in range(1, len(sentences)):
            current_sentence = sentences[i]
            sentence_tokens = self.text_processor.count_tokens(current_sentence)
            
            # Check if adding this sentence would exceed token limit
            if current_tokens + sentence_tokens > self.max_tokens_per_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [current_sentence]
                current_tokens = sentence_tokens
                continue
            
            # Compute simple similarity using word overlap
            current_words = set(current_chunk[-1].lower().split())
            next_words = set(current_sentence.lower().split())
            
            if current_words and next_words:
                overlap = len(current_words.intersection(next_words))
                union = len(current_words.union(next_words))
                similarity = overlap / union if union > 0 else 0
            else:
                similarity = 0
            
            # If similarity is below threshold, start a new chunk
            if similarity < self.similarity_threshold:
                chunks.append(' '.join(current_chunk))
                current_chunk = [current_sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(current_sentence)
                current_tokens += sentence_tokens
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def segment_text(self, text: str, method: str = "rule_based") -> List[str]:
        """
        Main segmentation method that chooses between rule-based and semantic approaches.
        
        Args:
            text: Preprocessed text
            method: "rule_based" or "semantic"
            
        Returns:
            List of text chunks
        """
        if method == "rule_based":
            return self.segment_text_rule_based(text)
        elif method == "semantic":
            return self.segment_text_semantic_simple(text)
        else:
            raise ValueError("Method must be 'rule_based' or 'semantic'")
    
    def analyze_chunks(self, chunks: List[str]) -> Dict[str, Any]:
        """
        Analyze the generated chunks for quality metrics.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Dictionary with analysis metrics
        """
        analysis = {
            "total_chunks": len(chunks),
            "avg_chunk_length": np.mean([len(chunk.split()) for chunk in chunks]),
            "avg_chunk_tokens": np.mean([self.text_processor.count_tokens(chunk) for chunk in chunks]),
            "chunk_lengths": [len(chunk.split()) for chunk in chunks],
            "chunk_tokens": [self.text_processor.count_tokens(chunk) for chunk in chunks]
        }
        
        return analysis
