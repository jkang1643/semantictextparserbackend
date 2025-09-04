"""
Request Manager for Handling Token Limits

This module handles token limits at the request/API level instead of during segmentation.
It splits long scenes into multiple requests while maintaining semantic coherence.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from text_processor import TextProcessor


class RequestManager:
    """Manages token limits by splitting long scenes into multiple requests."""
    
    def __init__(self, max_tokens_per_request: int = 512):
        """
        Initialize the request manager.
        
        Args:
            max_tokens_per_request: Maximum tokens per API request
        """
        self.max_tokens_per_request = max_tokens_per_request
        self.text_processor = TextProcessor()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (rough approximation)."""
        return self.text_processor.count_tokens(text)
    
    def split_scene_for_requests(self, scene: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a long scene into multiple request-sized chunks.
        
        Args:
            scene: Scene dictionary with 'text' and other metadata
            
        Returns:
            List of scene chunks, each suitable for a single request
        """
        scene_text = scene['text']
        scene_tokens = self.count_tokens(scene_text)
        
        # If scene is within token limit, return as-is
        if scene_tokens <= self.max_tokens_per_request:
            return [scene]
        
        # Split scene into sentences
        sentences = self.text_processor.tokenize_sentences(scene_text)
        
        # Create chunks that fit within token limit
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed limit, start new chunk
            if current_tokens + sentence_tokens > self.max_tokens_per_request and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                chunk_scene = self._create_chunk_scene(scene, chunk_text, len(chunks))
                chunks.append(chunk_scene)
                
                # Start new chunk
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_scene = self._create_chunk_scene(scene, chunk_text, len(chunks))
            chunks.append(chunk_scene)
        
        return chunks
    
    def _create_chunk_scene(self, original_scene: Dict[str, Any], chunk_text: str, chunk_index: int) -> Dict[str, Any]:
        """Create a scene chunk from original scene data."""
        return {
            'text': chunk_text,
            'sent_indices': original_scene.get('sent_indices', (0, 0)),
            'summary': original_scene.get('summary', {}),
            'prompt': self._generate_chunk_prompt(original_scene, chunk_text, chunk_index),
            'character': original_scene.get('character'),
            'chunk_index': chunk_index,
            'is_chunk': True,
            'original_scene_length': len(original_scene['text'].split())
        }
    
    def _generate_chunk_prompt(self, original_scene: Dict[str, Any], chunk_text: str, chunk_index: int) -> str:
        """Generate prompt for a scene chunk."""
        summary = original_scene.get('summary', {})
        
        # Build the structured prompt components
        setting = ', '.join(summary.get('setting', [])[:2]) if summary.get('setting') else None
        subjects = ', '.join(summary.get('subjects', [])[:2]) if summary.get('subjects') else 'figures'
        actions = ', '.join(summary.get('action_verbs', [])[:2]) if summary.get('action_verbs') else 'moving'
        objects = ', '.join(summary.get('objects', [])[:3]) if summary.get('objects') else 'various items'
        mood = ', '.join(summary.get('tone_adjs', [])[:2]) if summary.get('tone_adjs') else 'neutral'
        
        # Add chunk context
        chunk_context = f" (part {chunk_index + 1})" if chunk_index > 0 else ""
        
        # Build the structured prompt
        if setting:
            structured_prompt = f"{setting}; {subjects} {actions}{chunk_context}; key objects: {objects}; mood: {mood}"
        else:
            structured_prompt = f"{subjects} {actions}{chunk_context}; key objects: {objects}; mood: {mood}"
        
        # Keep structured prompt under 40 words
        words = structured_prompt.split()
        if len(words) > 40:
            structured_prompt = ' '.join(words[:40]) + '...'
        
        # Combine with chunk text
        combined_prompt = f"{chunk_text} {structured_prompt}"
        
        return combined_prompt
    
    def process_scenes_for_requests(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process all scenes and split long ones into request-sized chunks.
        
        Args:
            scenes: List of scene dictionaries
            
        Returns:
            List of request-ready scenes (may include chunks)
        """
        request_ready_scenes = []
        
        for scene in scenes:
            chunks = self.split_scene_for_requests(scene)
            request_ready_scenes.extend(chunks)
        
        return request_ready_scenes
    
    def get_request_summary(self, scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary of how scenes were split for requests.
        
        Args:
            scenes: List of processed scenes
            
        Returns:
            Summary dictionary with statistics
        """
        total_scenes = len(scenes)
        chunked_scenes = [s for s in scenes if s.get('is_chunk', False)]
        original_scenes = [s for s in scenes if not s.get('is_chunk', False)]
        
        return {
            'total_requests': total_scenes,
            'original_scenes': len(original_scenes),
            'chunked_scenes': len(chunked_scenes),
            'chunking_ratio': len(chunked_scenes) / len(original_scenes) if original_scenes else 0,
            'avg_tokens_per_request': sum(self.count_tokens(s['text']) for s in scenes) / total_scenes if scenes else 0
        }


def split_scenes_for_requests(scenes: List[Dict[str, Any]], 
                            max_tokens_per_request: int = 512) -> List[Dict[str, Any]]:
    """
    Convenience function to split scenes for requests.
    
    Args:
        scenes: List of scene dictionaries
        max_tokens_per_request: Maximum tokens per request
        
    Returns:
        List of request-ready scenes
    """
    manager = RequestManager(max_tokens_per_request)
    return manager.process_scenes_for_requests(scenes)


if __name__ == "__main__":
    # Example usage
    from scene_segmenter_unlimited import segment_scenes_unlimited
    
    # Example text
    text = """John entered the forest. The trees loomed overhead.
    The forest was alive with sounds. Strange markings appeared on the trees.
    Suddenly, the temperature dropped. Ahead, he heard a sound.
    A woman emerged, dressed in white. Her eyes glowed faintly.
    She spoke softly. Then she added another line.
    John felt a magical pull toward her."""
    
    # Segment without token limits
    scenes = segment_scenes_unlimited(text, target_scenes=2)
    
    print("Original scenes:")
    for i, scene in enumerate(scenes, 1):
        print(f"Scene {i}: {len(scene['text'])} chars, {len(scene['text'].split())} words")
    
    # Split for requests
    request_manager = RequestManager(max_tokens_per_request=100)  # Small limit for demo
    request_scenes = request_manager.process_scenes_for_requests(scenes)
    
    print("\nRequest-ready scenes:")
    for i, scene in enumerate(request_scenes, 1):
        print(f"Request {i}: {len(scene['text'])} chars, {len(scene['text'].split())} words")
        if scene.get('is_chunk'):
            print(f"  (Chunk {scene['chunk_index'] + 1} of original scene)")
    
    # Get summary
    summary = request_manager.get_request_summary(request_scenes)
    print(f"\nSummary: {summary}")
