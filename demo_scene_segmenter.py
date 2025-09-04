#!/usr/bin/env python3
"""
Demo script for the robust scene segmenter.
"""

from scene_segmenter import segment_scenes

def main():
    # Demo text from the requirements
    demo_text = """John entered the forest. The trees loomed overhead.
The forest was alive with sounds. Strange markings appeared on the trees.
Suddenly, the temperature dropped. Ahead, he heard a sound.
A woman emerged, dressed in white. Her eyes glowed faintly.
She spoke softly. Then she added another line.
John felt a magical pull toward her."""
    
    print("=== Robust Scene Segmentation Demo ===\n")
    print("Original text:")
    print(demo_text)
    print("\n" + "="*50 + "\n")
    
    # Test with target scenes
    print("Segmentation with target_scenes=3:")
    scenes = segment_scenes(demo_text, target_scenes=3, min_sentences=3)
    
    for i, scene in enumerate(scenes, 1):
        print(f"--- Scene {i} ---")
        print(f"Text: {scene['text']}")
        print(f"Prompt: {scene['prompt']}")
        print(f"Summary: {scene['summary']}")
        print()
    
    print("="*50 + "\n")
    
    # Test without target scenes (hysteresis-based)
    print("Segmentation with hysteresis-based approach:")
    scenes_hyst = segment_scenes(demo_text, min_sentences=3)
    
    for i, scene in enumerate(scenes_hyst, 1):
        print(f"--- Scene {i} ---")
        print(f"Text: {scene['text']}")
        print(f"Prompt: {scene['prompt']}")
        print()
    
    print("="*50 + "\n")
    
    # Test cue words don't over-segment
    print("Testing cue words don't cause over-segmentation:")
    cue_text = """John walked into the room. Suddenly, he noticed something strange.
Then he realized what it was. Next, he decided to investigate.
Meanwhile, the clock kept ticking."""
    
    scenes_cue = segment_scenes(cue_text, min_sentences=2)
    print(f"Text with many cue words: {len(scenes_cue)} scenes")
    for i, scene in enumerate(scenes_cue, 1):
        print(f"Scene {i}: {scene['text']}")
    
    print("\n" + "="*50 + "\n")
    
    # Test subject change causes segmentation
    print("Testing subject change causes segmentation:")
    subject_text = """John entered the forest. The trees were tall and dark.
The forest seemed mysterious. John felt uneasy.
Mary appeared from behind a tree. She smiled warmly.
Mary approached John slowly. She held out her hand."""
    
    scenes_subj = segment_scenes(subject_text, min_sentences=2)
    print(f"Text with subject change: {len(scenes_subj)} scenes")
    for i, scene in enumerate(scenes_subj, 1):
        print(f"Scene {i}: {scene['text']}")

if __name__ == "__main__":
    main()
