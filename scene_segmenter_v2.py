"""
Enhanced scene-based text segmentation module.

This module implements intelligent scene segmentation that groups text into scenes
based on either scene unfolding (progressive details within same setting) or 
scene changes (new environment, character, or narrative shift).

This is an improved version that better handles the requirements.
"""

import re
from typing import List, Dict, Set, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy


class SceneSegmenterV2:
    """
    Enhanced scene-based text segmentation.
    
    Segments text into scenes where each segment represents either:
    - A scene unfolding (progressive details within same setting)
    - A scene change (new environment, character, or narrative shift)
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.6,
                 max_tokens_per_chunk: int = 512,
                 model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the scene segmenter.
        
        Args:
            similarity_threshold: Threshold for scene context similarity (0.0 to 1.0)
            max_tokens_per_chunk: Maximum tokens per scene (fallback limit)
            model_name: Sentence transformer model name
        """
        self.similarity_threshold = similarity_threshold
        self.max_tokens_per_chunk = max_tokens_per_chunk
        
        # Initialize sentence transformer
        try:
            self.sentence_model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Warning: Could not load sentence transformer: {e}")
            self.sentence_model = None
        
        # Initialize spaCy for entity detection
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Entity detection disabled.")
            self.nlp = None
        
        # Scene change cue words - more comprehensive
        self.cue_words = {
            'temporal': [
                'suddenly', 'abruptly', 'instantly', 'immediately', 'then',
                'meanwhile', 'later', 'afterwards', 'next', 'finally',
                'eventually', 'soon', 'before', 'after', 'during'
            ],
            'spatial': [
                'meanwhile', 'elsewhere', 'across', 'beyond', 'nearby',
                'further', 'closer', 'away', 'inside', 'outside',
                'above', 'below', 'behind', 'ahead'
            ],
            'narrative': [
                'however', 'but', 'yet', 'although', 'despite',
                'conversely', 'alternatively', 'instead', 'rather',
                'moreover', 'furthermore', 'additionally', 'besides'
            ]
        }
        
        # Compile regex patterns for cue word detection
        self.cue_patterns = {}
        for category, words in self.cue_words.items():
            pattern = r'\b(' + '|'.join(words) + r')\b'
            self.cue_patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def segment_scenes(self, text: str, similarity_threshold: Optional[float] = None) -> List[str]:
        """
        Segment text into scenes based on scene unfolding and scene change logic.
        
        Args:
            text: Raw input text
            similarity_threshold: Override default similarity threshold
            
        Returns:
            List of scene segments (strings)
        """
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        if self.sentence_model is None:
            print("Warning: Sentence transformer not available. Using fallback segmentation.")
            return self._fallback_segmentation(text)
        
        # Split into sentences
        sentences = self._tokenize_sentences(text)
        
        if len(sentences) < 2:
            return [text]
        
        # Generate embeddings for all sentences
        embeddings = self.sentence_model.encode(sentences)
        
        # Initialize scene tracking
        scenes = []
        current_scene = [sentences[0]]
        current_tokens = self._count_tokens(sentences[0])
        current_entities = self._extract_entities(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = self._count_tokens(sentence)
            
            # Check token limit first (hard boundary)
            if current_tokens + sentence_tokens > self.max_tokens_per_chunk and current_scene:
                scenes.append(' '.join(current_scene))
                current_scene = [sentence]
                current_tokens = sentence_tokens
                current_entities = self._extract_entities(sentence)
                continue
            
            # Check for scene change indicators
            should_start_new_scene = self._should_start_new_scene(
                sentence, embeddings[i], embeddings, current_scene, i,
                current_entities, similarity_threshold
            )
            
            if should_start_new_scene:
                # Start new scene
                scenes.append(' '.join(current_scene))
                current_scene = [sentence]
                current_tokens = sentence_tokens
                current_entities = self._extract_entities(sentence)
            else:
                # Add to current scene
                current_scene.append(sentence)
                current_tokens += sentence_tokens
                
                # Update entities
                new_entities = self._extract_entities(sentence)
                current_entities.update(new_entities)
        
        # Add the last scene
        if current_scene:
            scenes.append(' '.join(current_scene))
        
        return scenes
    
    def _should_start_new_scene(self, 
                               sentence: str, 
                               sentence_embedding: np.ndarray,
                               all_embeddings: np.ndarray,
                               current_scene: List[str],
                               current_index: int,
                               current_entities: Set[str],
                               similarity_threshold: float) -> bool:
        """
        Determine if a sentence should start a new scene.
        
        Args:
            sentence: Current sentence
            sentence_embedding: Embedding of current sentence
            all_embeddings: All sentence embeddings
            current_scene: Current scene sentences
            current_index: Index of current sentence
            current_entities: Entities in current scene
            similarity_threshold: Similarity threshold for scene context
            
        Returns:
            True if should start new scene, False otherwise
        """
        # Check for strong cue words (hard boundary)
        if self._has_strong_cue_words(sentence):
            return True
        
        # Check for new entities (character/place not in current scene)
        sentence_entities = self._extract_entities(sentence)
        if sentence_entities and not sentence_entities.issubset(current_entities):
            return True
        
        # Calculate scene context vector as average of current scene
        scene_start = current_index - len(current_scene) + 1
        scene_embeddings = all_embeddings[scene_start:current_index]
        
        # Handle edge case where scene_embeddings might be empty
        if len(scene_embeddings) == 0:
            return True
        
        scene_context = np.mean(scene_embeddings, axis=0).reshape(1, -1)
        
        # Check for NaN values
        if np.isnan(scene_context).any() or np.isnan(sentence_embedding).any():
            return True
        
        # Check semantic similarity with scene context
        similarity = cosine_similarity(
            sentence_embedding.reshape(1, -1),
            scene_context
        )[0][0]
        
        return similarity < similarity_threshold
    
    def _has_strong_cue_words(self, sentence: str) -> bool:
        """Check if sentence contains strong scene change cue words."""
        # Only check for the most definitive scene change indicators
        strong_cues = [
            'suddenly', 'abruptly', 'instantly', 'immediately',
            'meanwhile', 'elsewhere', 'however', 'but', 'yet'
        ]
        
        sentence_lower = sentence.lower()
        for cue in strong_cues:
            if cue in sentence_lower:
                return True
        return False
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract named entities from text."""
        if self.nlp is None:
            return set()
        
        doc = self.nlp(text)
        entities = set()
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'FAC']:
                entities.add(ent.text.lower())
        
        return entities
    
    def _tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        if self.nlp is None:
            # Fallback to simple regex-based splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
        
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Filter out very short sentences
        sentences = [sent for sent in sentences if len(sent.split()) > 2]
        
        return sentences
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4
    
    def _fallback_segmentation(self, text: str) -> List[str]:
        """Fallback segmentation when sentence transformer is not available."""
        sentences = self._tokenize_sentences(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.max_tokens_per_chunk and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def analyze_scenes(self, scenes: List[str]) -> Dict[str, any]:
        """
        Analyze the generated scenes for quality metrics.
        
        Args:
            scenes: List of scene segments
            
        Returns:
            Dictionary with analysis metrics
        """
        if not scenes:
            return {"total_scenes": 0}
        
        scene_lengths = [len(scene.split()) for scene in scenes]
        scene_tokens = [self._count_tokens(scene) for scene in scenes]
        
        analysis = {
            "total_scenes": len(scenes),
            "avg_scene_length": np.mean(scene_lengths),
            "avg_scene_tokens": np.mean(scene_tokens),
            "scene_lengths": scene_lengths,
            "scene_tokens": scene_tokens,
            "min_scene_length": min(scene_lengths),
            "max_scene_length": max(scene_lengths)
        }
        
        return analysis


# Convenience function for easy usage
def segment_scenes_v2(text: str, similarity_threshold: float = 0.6) -> List[str]:
    """
    Convenience function to segment text into scenes.
    
    Args:
        text: Raw input text
        similarity_threshold: Threshold for scene context similarity
        
    Returns:
        List of scene segments
    """
    segmenter = SceneSegmenterV2(similarity_threshold=similarity_threshold)
    return segmenter.segment_scenes(text, similarity_threshold)


# Backward compatibility
def segment_scenes(text: str, similarity_threshold: float = 0.6) -> List[str]:
    """Backward compatible function."""
    return segment_scenes_v2(text, similarity_threshold)
