"""
Tests for the robust scene segmenter to verify less-granular behavior.
"""

import pytest
from scene_segmenter import segment_scenes, SceneSegmenter


def test_demo_text_less_granular():
    """Test the demo text produces fewer, smarter scenes."""
    demo_text = """John entered the forest. The trees loomed overhead.
The forest was alive with sounds. Strange markings appeared on the trees.
Suddenly, the temperature dropped. Ahead, he heard a sound.
A woman emerged, dressed in white. Her eyes glowed faintly.
She spoke softly. Then she added another line.
John felt a magical pull toward her."""
    
    scenes = segment_scenes(demo_text, target_scenes=3, min_sentences=3)
    
    # Should produce approximately 3 scenes (not over-segmented)
    assert len(scenes) >= 2, f"Expected at least 2 scenes, got {len(scenes)}"
    assert len(scenes) <= 4, f"Expected at most 4 scenes, got {len(scenes)}"
    
    # Each scene should have reasonable length
    for i, scene in enumerate(scenes):
        assert len(scene['text']) > 20, f"Scene {i+1} too short: {scene['text']}"
        assert 'prompt' in scene, f"Scene {i+1} missing prompt"
        assert 'summary' in scene, f"Scene {i+1} missing summary"
        assert 'sent_indices' in scene, f"Scene {i+1} missing sent_indices"
    
    print("Demo text segmentation:")
    for i, scene in enumerate(scenes, 1):
        print(f"--- Scene {i} ---")
        print(f"Text: {scene['text']}")
        print(f"Prompt: {scene['prompt']}")
        print()


def test_no_split_on_cue_words_alone():
    """Test that cue words alone don't cause splits if context is stable."""
    text = """John walked into the room. Suddenly, he noticed something strange.
Then he realized what it was. Next, he decided to investigate.
Meanwhile, the clock kept ticking."""
    
    scenes = segment_scenes(text, min_sentences=2)
    
    # Should not over-segment on cue words alone
    assert len(scenes) <= 2, f"Over-segmented on cue words: {len(scenes)} scenes"
    
    print("Cue words test:")
    for i, scene in enumerate(scenes, 1):
        print(f"Scene {i}: {scene['text']}")


def test_split_on_subject_change():
    """Test that dominant subject changes cause splits even without cue words."""
    text = """John entered the forest. The trees were tall and dark.
The forest seemed mysterious. John felt uneasy.
Mary appeared from behind a tree. She smiled warmly.
Mary approached John slowly. She held out her hand."""
    
    scenes = segment_scenes(text, min_sentences=2)
    
    # Should split when subject changes from John to Mary
    assert len(scenes) >= 2, f"Should split on subject change: {len(scenes)} scenes"
    
    print("Subject change test:")
    for i, scene in enumerate(scenes, 1):
        print(f"Scene {i}: {scene['text']}")


def test_min_sentences_respected():
    """Test that minimum sentence requirements are respected."""
    text = """First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."""
    
    scenes = segment_scenes(text, min_sentences=3)
    
    # Should respect minimum sentences
    for scene in scenes:
        sentence_count = len(scene['text'].split('.')) - 1  # Rough count
        assert sentence_count >= 3, f"Scene too short: {scene['text']}"


if __name__ == "__main__":
    # Run the tests
    test_demo_text_less_granular()
    test_no_split_on_cue_words_alone()
    test_split_on_subject_change()
    test_min_sentences_respected()
    
    print("\nAll tests passed! The scene segmenter is working correctly.")