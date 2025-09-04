"""
Final intelligent scene-based text segmentation module.

This module implements the most intelligent scene segmentation that groups text into 
coherent scenes based on major narrative shifts, not minor transitions.
"""

import re
from typing import List, Dict, Set, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy


class FinalSceneSegmenter:
    """
    Final intelligent scene-based text segmentation.
    
    Focuses on major narrative shifts rather than minor transitions.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.2,
                 max_tokens_per_chunk: int = 512,
                 model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the final scene segmenter.
        
        Args:
            similarity_threshold: Threshold for semantic similarity (0.0 to 1.0)
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
        
        # Initialize spaCy for entity and subject detection
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Subject detection disabled.")
            self.nlp = None
        
        # Only major scene change indicators
        self.major_transitions = [
            'meanwhile', 'elsewhere', 'however', 'but', 'yet'
        ]
        
        # Compile regex pattern for major transition detection
        self.transition_pattern = re.compile(
            r'\b(' + '|'.join(self.major_transitions) + r')\b',
            re.IGNORECASE
        )
    
    def segment_scenes(self, text: str, similarity_threshold: Optional[float] = None) -> List[str]:
        """
        Segment text into coherent scenes based on major narrative shifts.
        
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
        
        # Track scene context
        scene_context = {
            'subjects': self._extract_subjects(sentences[0]),
            'entities': self._extract_entities(sentences[0]),
            'embedding_sum': embeddings[0],
            'sentence_count': 1
        }
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = self._count_tokens(sentence)
            
            # Check token limit first (hard boundary)
            if current_tokens + sentence_tokens > self.max_tokens_per_chunk and current_scene:
                scenes.append(' '.join(current_scene))
                current_scene = [sentence]
                current_tokens = sentence_tokens
                scene_context = {
                    'subjects': self._extract_subjects(sentence),
                    'entities': self._extract_entities(sentence),
                    'embedding_sum': embeddings[i],
                    'sentence_count': 1
                }
                continue
            
            # Check for scene break conditions
            should_break_scene = self._should_break_scene(
                sentence, embeddings[i], scene_context, similarity_threshold, i, sentences
            )
            
            if should_break_scene:
                # Start new scene
                scenes.append(' '.join(current_scene))
                current_scene = [sentence]
                current_tokens = sentence_tokens
                scene_context = {
                    'subjects': self._extract_subjects(sentence),
                    'entities': self._extract_entities(sentence),
                    'embedding_sum': embeddings[i],
                    'sentence_count': 1
                }
            else:
                # Add to current scene
                current_scene.append(sentence)
                current_tokens += sentence_tokens
                
                # Update scene context
                self._update_scene_context(scene_context, sentence, embeddings[i])
        
        # Add the last scene
        if current_scene:
            scenes.append(' '.join(current_scene))
        
        return scenes
    
    def _should_break_scene(self, 
                           sentence: str, 
                           sentence_embedding: np.ndarray,
                           scene_context: Dict,
                           similarity_threshold: float,
                           sentence_index: int,
                           all_sentences: List[str]) -> bool:
        """
        Determine if a sentence should start a new scene.
        More conservative approach focusing on major shifts.
        """
        # Check for major transition words (hard boundary)
        if self._has_major_transitions(sentence):
            return True
        
        # Check for character/entity introduction (new person appears)
        if self._has_new_character(sentence, scene_context['entities']):
            return True
        
        # Check for major location change
        if self._has_location_change(sentence, scene_context['entities']):
            return True
        
        # Check semantic similarity with scene context (more lenient)
        scene_embedding = scene_context['embedding_sum'] / scene_context['sentence_count']
        similarity = cosine_similarity(
            sentence_embedding.reshape(1, -1),
            scene_embedding.reshape(1, -1)
        )[0][0]
        
        # Only break if similarity is very low AND we have enough context
        if scene_context['sentence_count'] >= 2 and similarity < similarity_threshold:
            return True
        
        return False
    
    def _has_major_transitions(self, sentence: str) -> bool:
        """Check if sentence starts with major transition words."""
        sentence_lower = sentence.lower().strip()
        for transition in self.major_transitions:
            if sentence_lower.startswith(transition):
                return True
        return False
    
    def _has_new_character(self, sentence: str, current_entities: Set[str]) -> bool:
        """Check if a new character is introduced."""
        if self.nlp is None:
            return False
        
        doc = self.nlp(sentence)
        sentence_entities = set()
        
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                sentence_entities.add(ent.text.lower())
        
        # If we find new person entities not in current scene
        return sentence_entities and not sentence_entities.issubset(current_entities)
    
    def _has_location_change(self, sentence: str, current_entities: Set[str]) -> bool:
        """Check if there's a major location change."""
        if self.nlp is None:
            return False
        
        doc = self.nlp(sentence)
        sentence_entities = set()
        
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC', 'FAC']:  # Geographic, Location, Facility
                sentence_entities.add(ent.text.lower())
        
        # If we find new location entities not in current scene
        return sentence_entities and not sentence_entities.issubset(current_entities)
    
    def _extract_subjects(self, text: str) -> Set[str]:
        """Extract grammatical subjects from text using spaCy."""
        if self.nlp is None:
            return set()
        
        doc = self.nlp(text)
        subjects = set()
        
        for token in doc:
            # Look for subjects in dependency tree
            if token.dep_ in ['nsubj', 'nsubjpass']:  # nominal subject
                subjects.add(token.lemma_.lower())
            elif token.dep_ == 'pobj' and token.head.dep_ in ['prep']:  # prepositional object
                subjects.add(token.lemma_.lower())
            # Also include main entities as subjects
            elif token.ent_type_ in ['PERSON', 'ORG', 'GPE', 'LOC']:
                subjects.add(token.lemma_.lower())
        
        return subjects
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract named entities from text."""
        if self.nlp is None:
            return set()
        
        doc = self.nlp(text)
        entities = set()
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'FAC', 'EVENT']:
                entities.add(ent.text.lower())
        
        return entities
    
    def _update_scene_context(self, scene_context: Dict, sentence: str, embedding: np.ndarray) -> None:
        """Update scene context with new sentence."""
        # Update subjects and entities
        new_subjects = self._extract_subjects(sentence)
        new_entities = self._extract_entities(sentence)
        
        scene_context['subjects'].update(new_subjects)
        scene_context['entities'].update(new_entities)
        
        # Update embedding average
        scene_context['embedding_sum'] += embedding
        scene_context['sentence_count'] += 1
    
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
    
    def generate_scene_titles(self, scenes: List[str]) -> List[str]:
        """
        Generate scene titles based on main subjects and actions.
        
        Args:
            scenes: List of scene segments
            
        Returns:
            List of scene titles
        """
        titles = []
        
        for i, scene in enumerate(scenes, 1):
            if self.nlp is None:
                titles.append(f"Scene {i}")
                continue
            
            # Extract main subject and key action
            doc = self.nlp(scene)
            
            # Find main subject (first noun or pronoun)
            main_subject = None
            for token in doc:
                if token.dep_ in ['nsubj', 'nsubjpass'] and token.pos_ in ['NOUN', 'PROPN', 'PRON']:
                    main_subject = token.text
                    break
            
            # Find key action (main verb)
            main_action = None
            for token in doc:
                if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                    main_action = token.lemma_
                    break
            
            # Generate title
            if main_subject and main_action:
                title = f"Scene {i}: {main_subject} {main_action.title()}"
            elif main_subject:
                title = f"Scene {i}: {main_subject}"
            else:
                title = f"Scene {i}"
            
            titles.append(title)
        
        return titles
    
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
def segment_scenes(text: str, similarity_threshold: float = 0.2) -> List[str]:
    """
    Convenience function to segment text into coherent scenes.
    
    Args:
        text: Raw input text
        similarity_threshold: Threshold for semantic similarity
        
    Returns:
        List of scene segments
    """
    segmenter = FinalSceneSegmenter(similarity_threshold=similarity_threshold)
    return segmenter.segment_scenes(text, similarity_threshold)


def segment_scenes_with_titles(text: str, similarity_threshold: float = 0.2) -> Tuple[List[str], List[str]]:
    """
    Segment text into scenes and generate titles.
    
    Args:
        text: Raw input text
        similarity_threshold: Threshold for semantic similarity
        
    Returns:
        Tuple of (scene_segments, scene_titles)
    """
    segmenter = FinalSceneSegmenter(similarity_threshold=similarity_threshold)
    scenes = segmenter.segment_scenes(text, similarity_threshold)
    titles = segmenter.generate_scene_titles(scenes)
    return scenes, titles
