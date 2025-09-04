"""
Unlimited Scene Segmentation Module

Modified version that removes token limits from segmentation logic,
allowing for longer scenes based purely on semantic content.
Token limits are handled at the request/API level instead.
"""

import re
import math
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import Counter, defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from dataclasses import dataclass


@dataclass
class SceneContext:
    """Maintains context for the current scene being built."""
    ema_embedding: np.ndarray
    subjects: Set[str]
    entities: Set[str]
    visual_tokens: Set[str]
    sentence_count: int = 0


class UnlimitedSceneSegmenter:
    """Scene segmentation without token limits - purely semantic-based."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the segmenter with the specified model."""
        self.model = SentenceTransformer(model_name)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: en_core_web_sm not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Transition phrases for soft cue scoring
        self.transition_phrases = [
            "suddenly", "then", "next", "meanwhile", "afterwards", "later",
            "at that moment", "just then", "in the meantime", "subsequently",
            "meanwhile", "furthermore", "moreover", "however", "nevertheless",
            "on the other hand", "conversely", "in contrast", "meanwhile",
            "at the same time", "simultaneously", "concurrently"
        ]
    
    def _extract_sentence_features(self, sentence: str) -> Dict[str, Any]:
        """Extract linguistic features from a sentence."""
        if not self.nlp:
            return self._fallback_extract_features(sentence)
        
        doc = self.nlp(sentence)
        
        # Extract main subject
        main_subject = None
        for token in doc:
            if token.dep_ == "nsubj" and token.pos_ in ["PROPN", "NOUN"]:
                main_subject = token.lemma_.lower()
                break
        
        if not main_subject:
            # Fallback to first proper noun or noun
            for token in doc:
                if token.pos_ in ["PROPN", "NOUN"]:
                    main_subject = token.lemma_.lower()
                    break
        
        # Extract entities
        entities = set()
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "GPE", "LOC", "FAC", "ORG", "NORP", "DATE", "TIME"]:
                entities.add(ent.text.lower())
        
        # Extract noun chunks (objects)
        noun_chunks = [chunk.lemma_.lower() for chunk in doc.noun_chunks]
        
        # Extract verbs (actions)
        verbs = [token.lemma_.lower() for token in doc if token.pos_ == "VERB"]
        
        # Extract adjectives (tone)
        adjectives = [token.lemma_.lower() for token in doc if token.pos_ == "ADJ"]
        
        # Extract visual tokens (nouns, proper nouns, adjectives from noun chunks)
        visual_tokens = set()
        for chunk in doc.noun_chunks:
            for token in chunk:
                if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
                    visual_tokens.add(token.lemma_.lower())
        
        # Add prominent verbs to visual tokens
        for verb in verbs[:3]:  # Top 3 verbs
            visual_tokens.add(verb)
        
        return {
            "main_subject": main_subject,
            "entities": entities,
            "noun_chunks": noun_chunks,
            "verbs": verbs,
            "adjectives": adjectives,
            "visual_tokens": visual_tokens
        }
    
    def _fallback_extract_features(self, sentence: str) -> Dict[str, Any]:
        """Fallback feature extraction without spaCy."""
        # Simple regex-based extraction
        words = re.findall(r'\b\w+\b', sentence.lower())
        
        # Extract potential subjects (first few words)
        main_subject = words[0] if words else None
        
        # Simple entity detection (capitalized words)
        entities = set(re.findall(r'\b[A-Z][a-z]+\b', sentence))
        
        # Extract nouns (simple heuristic)
        noun_chunks = [word for word in words if len(word) > 3]
        
        # Extract verbs (simple heuristic - words ending in common verb patterns)
        verbs = [word for word in words if any(word.endswith(suffix) for suffix in ['ed', 'ing', 's'])]
        
        # Extract adjectives (simple heuristic)
        adjectives = [word for word in words if any(word.endswith(suffix) for suffix in ['ful', 'less', 'ous', 'ive', 'al'])]
        
        visual_tokens = set(noun_chunks + verbs[:3])
        
        return {
            "main_subject": main_subject,
            "entities": entities,
            "noun_chunks": noun_chunks,
            "verbs": verbs,
            "adjectives": adjectives,
            "visual_tokens": visual_tokens
        }
    
    def _compute_novelty_score(self, 
                             sentence_features: Dict[str, Any],
                             scene_context: SceneContext,
                             weights: Dict[str, float]) -> float:
        """Compute multi-signal novelty score for a sentence."""
        
        # 1. Semantic similarity (1 - cosine similarity)
        sentence_embedding = self.model.encode([sentence_features.get('text', '')])
        if scene_context.ema_embedding is not None:
            semantic_sim = cosine_similarity(
                sentence_embedding.reshape(1, -1),
                scene_context.ema_embedding.reshape(1, -1)
            )[0][0]
            semantic_novelty = 1 - semantic_sim
        else:
            semantic_novelty = 0.0
        
        # 2. Entity novelty (Jaccard distance)
        current_entities = sentence_features.get('entities', set())
        if scene_context.entities:
            intersection = len(current_entities & scene_context.entities)
            union = len(current_entities | scene_context.entities)
            entity_novelty = 1 - (intersection / union) if union > 0 else 0.0
        else:
            entity_novelty = 0.0
        
        # 3. Visual token novelty (Jaccard distance)
        current_visual = sentence_features.get('visual_tokens', set())
        if scene_context.visual_tokens:
            intersection = len(current_visual & scene_context.visual_tokens)
            union = len(current_visual | scene_context.visual_tokens)
            visual_novelty = 1 - (intersection / union) if union > 0 else 0.0
        else:
            visual_novelty = 0.0
        
        # 4. Cue word bonus (soft hint only)
        sentence_text = sentence_features.get('text', '').lower()
        cue_bonus = 0.0
        for phrase in self.transition_phrases:
            if sentence_text.startswith(phrase):
                cue_bonus = 0.05  # Small bonus, not a hard boundary
                break
        
        # Weighted combination
        novelty = (
            weights["semantic"] * semantic_novelty +
            weights["entity"] * entity_novelty +
            weights["visual"] * visual_novelty +
            weights["cue"] * cue_bonus
        )
        
        return novelty
    
    def _update_scene_context(self, 
                            scene_context: SceneContext,
                            sentence_features: Dict[str, Any],
                            sentence_embedding: np.ndarray,
                            alpha: float = 0.3) -> None:
        """Update scene context with new sentence information."""
        
        # Update EMA embedding
        if scene_context.ema_embedding is None:
            scene_context.ema_embedding = sentence_embedding.flatten()
        else:
            scene_context.ema_embedding = (
                alpha * sentence_embedding.flatten() + 
                (1 - alpha) * scene_context.ema_embedding
            )
        
        # Update entity sets
        scene_context.entities.update(sentence_features.get('entities', set()))
        
        # Update visual tokens
        scene_context.visual_tokens.update(sentence_features.get('visual_tokens', set()))
        
        # Update subjects
        if sentence_features.get('main_subject'):
            scene_context.subjects.add(sentence_features['main_subject'])
        
        scene_context.sentence_count += 1
    
    def _find_peak_boundaries(self, 
                            novelty_scores: List[float],
                            target_scenes: int,
                            min_sentences: int) -> List[int]:
        """Find peak boundaries for target scene count."""
        if len(novelty_scores) < min_sentences * target_scenes:
            return []
        
        # Smooth the novelty scores
        window_size = min(3, len(novelty_scores) // 4)
        if window_size > 1:
            smoothed = []
            for i in range(len(novelty_scores)):
                start = max(0, i - window_size // 2)
                end = min(len(novelty_scores), i + window_size // 2 + 1)
                smoothed.append(np.mean(novelty_scores[start:end]))
        else:
            smoothed = novelty_scores
        
        # Find local maxima
        peaks = []
        for i in range(min_sentences, len(smoothed) - min_sentences):
            if (smoothed[i] > smoothed[i-1] and 
                smoothed[i] > smoothed[i+1] and
                smoothed[i] > np.mean(smoothed)):
                peaks.append((i, smoothed[i]))
        
        # Sort by score and take top target_scenes-1
        peaks.sort(key=lambda x: x[1], reverse=True)
        boundaries = [peak[0] for peak in peaks[:target_scenes-1]]
        boundaries.sort()
        
        return boundaries
    
    def _create_scene_summary(self, sentences: List[str], features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of the scene for image generation with improved location detection."""
        
        # Aggregate features across all sentences in the scene
        all_subjects = []
        all_entities = []
        all_objects = []
        all_verbs = []
        all_adjectives = []
        
        for features in features_list:
            if features.get('main_subject'):
                all_subjects.append(features['main_subject'])
            all_entities.extend(features.get('entities', []))
            all_objects.extend(features.get('noun_chunks', []))
            all_verbs.extend(features.get('verbs', []))
            all_adjectives.extend(features.get('adjectives', []))
        
        # Get most frequent items
        subject_counter = Counter(all_subjects)
        entity_counter = Counter(all_entities)
        object_counter = Counter(all_objects)
        verb_counter = Counter(all_verbs)
        adj_counter = Counter(all_adjectives)
        
        # Extract top items
        subjects = [item for item, count in subject_counter.most_common(3)]
        entities = [item for item, count in entity_counter.most_common(5)]
        objects = [item for item, count in object_counter.most_common(5)]
        action_verbs = [item for item, count in verb_counter.most_common(3)]
        tone_adjs = [item for item, count in adj_counter.most_common(3)]
        
        # Improved location detection - prioritize GPE/LOC/FAC entities and common location nouns
        location_entities = []
        
        # First, look for GPE/LOC/FAC entities
        for ent in entities:
            if any(loc_type in ent for loc_type in ['GPE', 'LOC', 'FAC']):
                location_entities.append(ent)
        
        # Then look for common location nouns in entities and objects
        location_keywords = [
            'forest', 'room', 'house', 'street', 'mountain', 'river', 'city', 'village', 'castle',
            'building', 'park', 'beach', 'desert', 'valley', 'hill', 'bridge', 'tower', 'palace',
            'garden', 'field', 'path', 'road', 'square', 'plaza', 'market', 'shop', 'store',
            'office', 'library', 'school', 'hospital', 'church', 'temple', 'cave', 'cliff',
            'lake', 'ocean', 'sea', 'island', 'coast', 'shore', 'meadow', 'prairie', 'jungle'
        ]
        
        for item in entities + objects:
            if any(loc_word in item.lower() for loc_word in location_keywords):
                if item not in location_entities:
                    location_entities.append(item)
        
        # If no location found, set to empty list (will be excluded from prompt)
        if not location_entities:
            location_entities = []
        
        return {
            'subjects': subjects,
            'setting': location_entities,
            'objects': objects,
            'action_verbs': action_verbs,
            'tone_adjs': tone_adjs
        }
    
    def _generate_image_prompt(self, summary: Dict[str, Any], original_text: str = "") -> str:
        """Generate a concise image prompt for the scene and append to original text."""
        
        # Build the structured prompt components
        setting = ', '.join(summary['setting'][:2]) if summary['setting'] else None
        subjects = ', '.join(summary['subjects'][:2]) if summary['subjects'] else 'figures'
        actions = ', '.join(summary['action_verbs'][:2]) if summary['action_verbs'] else 'moving'
        objects = ', '.join(summary['objects'][:3]) if summary['objects'] else 'various items'
        mood = ', '.join(summary['tone_adjs'][:2]) if summary['tone_adjs'] else 'neutral'
        
        # Build the structured prompt - exclude setting if not found
        if setting:
            structured_prompt = f"{setting}; {subjects} {actions}; key objects: {objects}; mood: {mood}"
        else:
            structured_prompt = f"{subjects} {actions}; key objects: {objects}; mood: {mood}"
        
        # Keep structured prompt under 40 words
        words = structured_prompt.split()
        if len(words) > 40:
            structured_prompt = ' '.join(words[:40]) + '...'
        
        # Append the structured prompt to the original text
        if original_text:
            combined_prompt = f"{original_text} {structured_prompt}"
        else:
            combined_prompt = structured_prompt
        
        return combined_prompt
    
    def segment_scenes(self,
                      text: str,
                      target_scenes: Optional[int] = None,
                      similarity_threshold: Optional[float] = None,
                      min_sentences: int = 3,
                      cooldown: int = 1,
                      hysteresis: Tuple[float, float] = (0.58, 0.68),
                      weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Segment text into image-ready scenes WITHOUT token limits.
        
        Args:
            text: Input text to segment
            target_scenes: Target number of scenes (optional)
            similarity_threshold: Fallback threshold if target_scenes not provided
            min_sentences: Minimum sentences per scene
            cooldown: Minimum sentences between boundaries
            hysteresis: (stay_inside, enter_boundary) thresholds
            weights: Weights for novelty scoring components
            
        Returns:
            List of scene dictionaries with text, indices, summary, and prompt
        """
        
        if weights is None:
            weights = {"semantic": 0.55, "entity": 0.25, "visual": 0.15, "cue": 0.05}
        
        # Split into sentences
        if self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < min_sentences:
            return [{
                'text': text,
                'sent_indices': (0, len(sentences)),
                'summary': self._create_scene_summary(sentences, []),
                'prompt': self._generate_image_prompt(self._create_scene_summary(sentences, []))
            }]
        
        # Extract features for all sentences
        features_list = []
        embeddings = []
        
        for sentence in sentences:
            features = self._extract_sentence_features(sentence)
            features['text'] = sentence
            features_list.append(features)
            
            # Get embedding
            embedding = self.model.encode([sentence])
            embeddings.append(embedding.flatten())
        
        # Choose segmentation strategy
        if target_scenes is not None:
            # Use peak detection for target scene count
            novelty_scores = []
            for i, features in enumerate(features_list):
                # Create temporary scene context for scoring
                temp_context = SceneContext(None, set(), set(), set())
                for j in range(max(0, i-2), i):
                    self._update_scene_context(temp_context, features_list[j], embeddings[j])
                
                novelty = self._compute_novelty_score(features, temp_context, weights)
                novelty_scores.append(novelty)
            
            boundaries = self._find_peak_boundaries(novelty_scores, target_scenes, min_sentences)
            
        else:
            # Use hysteresis-based segmentation
            boundaries = []
            scene_context = SceneContext(None, set(), set(), set())
            last_boundary = -min_sentences
            
            for i, (features, embedding) in enumerate(zip(features_list, embeddings)):
                if i - last_boundary < min_sentences:
                    self._update_scene_context(scene_context, features, embedding)
                    continue
                
                novelty = self._compute_novelty_score(features, scene_context, weights)
                
                # Check for boundary
                if (novelty > hysteresis[1] and  # enter_boundary
                    i - last_boundary >= cooldown):
                    boundaries.append(i)
                    last_boundary = i
                    # Reset scene context
                    scene_context = SceneContext(None, set(), set(), set())
                
                self._update_scene_context(scene_context, features, embedding)
        
        # Create scenes from boundaries with character consistency tracking
        scenes = []
        start_idx = 0
        current_character = None
        
        for boundary in boundaries:
            if boundary > start_idx:
                scene_sentences = sentences[start_idx:boundary]
                scene_features = features_list[start_idx:boundary]
                scene_text = ' '.join(scene_sentences)
                
                summary = self._create_scene_summary(scene_sentences, scene_features)
                
                # Check for character change
                scene_subjects = summary.get('subjects', [])
                if scene_subjects:
                    primary_subject = scene_subjects[0]
                    if primary_subject != current_character:
                        current_character = primary_subject
                
                # Generate prompt with original text and character consistency
                prompt = self._generate_image_prompt(summary, scene_text)
                
                scenes.append({
                    'text': scene_text,
                    'sent_indices': (start_idx, boundary),
                    'summary': summary,
                    'prompt': prompt,
                    'character': current_character
                })
                
                start_idx = boundary
        
        # Add final scene
        if start_idx < len(sentences):
            scene_sentences = sentences[start_idx:]
            scene_features = features_list[start_idx:]
            scene_text = ' '.join(scene_sentences)
            
            summary = self._create_scene_summary(scene_sentences, scene_features)
            
            # Check for character change
            scene_subjects = summary.get('subjects', [])
            if scene_subjects:
                primary_subject = scene_subjects[0]
                if primary_subject != current_character:
                    current_character = primary_subject
            
            # Generate prompt with original text and character consistency
            prompt = self._generate_image_prompt(summary, scene_text)
            
            scenes.append({
                'text': scene_text,
                'sent_indices': (start_idx, len(sentences)),
                'summary': summary,
                'prompt': prompt,
                'character': current_character
            })
        
        return scenes


def segment_scenes_unlimited(text: str,
                           target_scenes: Optional[int] = None,
                           similarity_threshold: Optional[float] = None,
                           min_sentences: int = 3,
                           cooldown: int = 1,
                           hysteresis: Tuple[float, float] = (0.58, 0.68),
                           weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """
    Convenience function for unlimited scene segmentation.
    
    Returns a list of scenes. Each scene dict includes:
    - 'text': str
    - 'sent_indices': (start, end)
    - 'summary': { 'subjects', 'setting', 'objects', 'action_verbs', 'tone_adjs' }
    - 'prompt': str   # short image prompt for the scene
    """
    segmenter = UnlimitedSceneSegmenter()
    return segmenter.segment_scenes(
        text=text,
        target_scenes=target_scenes,
        similarity_threshold=similarity_threshold,
        min_sentences=min_sentences,
        cooldown=cooldown,
        hysteresis=hysteresis,
        weights=weights
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Segment text into image-ready scenes (unlimited)")
    parser.add_argument("input_file", help="Input text file")
    parser.add_argument("--target-scenes", type=int, help="Target number of scenes")
    parser.add_argument("--min-sentences", type=int, default=3, help="Minimum sentences per scene")
    parser.add_argument("--output", help="Output file (optional)")
    
    args = parser.parse_args()
    
    # Read input file
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Segment scenes
    scenes = segment_scenes_unlimited(
        text=text,
        target_scenes=args.target_scenes,
        min_sentences=args.min_sentences
    )
    
    # Output results
    output_text = ""
    for i, scene in enumerate(scenes, 1):
        output_text += f"--- Scene {i} ---\n"
        output_text += f"{scene['text']}\n\n"
        output_text += f"Prompt: {scene['prompt']}\n\n"
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
    else:
        print(output_text)
