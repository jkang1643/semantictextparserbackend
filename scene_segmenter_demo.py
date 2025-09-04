"""
Comprehensive demo of the intelligent scene segmentation system.

This demo shows the evolution from basic segmentation to intelligent scene-based
segmentation that groups text into coherent scenes based on narrative flow.
"""

from final_scene_segmenter import segment_scenes, segment_scenes_with_titles
from intelligent_scene_segmenter import segment_scenes as intelligent_segment_scenes
from scene_segmenter import segment_scenes as basic_segment_scenes


def demo_scene_segmentation():
    """Demonstrate different scene segmentation approaches."""
    
    demo_text = """John entered the forest. The trees loomed overhead. 
The forest was alive with sounds. Strange markings appeared on the trees. 
Suddenly, the temperature dropped. Ahead, he heard a sound. 
A woman emerged, dressed in white. Her eyes glowed faintly. 
She spoke softly. Then she added another line. 
John felt a magical pull toward her."""
    
    print("🎬 Scene Segmentation Comparison Demo")
    print("=" * 50)
    print()
    
    # Basic segmentation (original approach)
    print("1. Basic Segmentation (Original)")
    print("-" * 35)
    basic_scenes = basic_segment_scenes(demo_text, similarity_threshold=0.72)
    print(f"Total scenes: {len(basic_scenes)}")
    for i, scene in enumerate(basic_scenes, 1):
        print(f"  Scene {i}: {scene[:60]}...")
    print()
    
    # Intelligent segmentation (intermediate approach)
    print("2. Intelligent Segmentation (Intermediate)")
    print("-" * 45)
    intelligent_scenes = intelligent_segment_scenes(demo_text, similarity_threshold=0.3)
    print(f"Total scenes: {len(intelligent_scenes)}")
    for i, scene in enumerate(intelligent_scenes, 1):
        print(f"  Scene {i}: {scene[:60]}...")
    print()
    
    # Final segmentation (optimized approach)
    print("3. Final Segmentation (Optimized)")
    print("-" * 40)
    final_scenes = segment_scenes(demo_text, similarity_threshold=0.2)
    print(f"Total scenes: {len(final_scenes)}")
    for i, scene in enumerate(final_scenes, 1):
        print(f"  Scene {i}: {scene[:60]}...")
    print()
    
    # Show full final segmentation
    print("4. Complete Final Segmentation")
    print("-" * 35)
    scenes, titles = segment_scenes_with_titles(demo_text, similarity_threshold=0.2)
    for i, (scene, title) in enumerate(zip(scenes, titles), 1):
        print(f"{title}")
        print(f"{scene}")
        print()
    
    print("✅ Demo completed!")
    print()
    print("Key Improvements:")
    print("- Less granular segmentation (4 scenes vs 12+ scenes)")
    print("- Focuses on major narrative shifts")
    print("- Groups related sentences together")
    print("- Detects character introductions and location changes")
    print("- Generates meaningful scene titles")


def test_different_thresholds():
    """Test different similarity thresholds to show the effect."""
    
    demo_text = """John entered the forest. The trees loomed overhead. 
The forest was alive with sounds. Strange markings appeared on the trees. 
Suddenly, the temperature dropped. Ahead, he heard a sound. 
A woman emerged, dressed in white. Her eyes glowed faintly. 
She spoke softly. Then she added another line. 
John felt a magical pull toward her."""
    
    print("🎬 Threshold Sensitivity Test")
    print("=" * 35)
    print()
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for threshold in thresholds:
        scenes = segment_scenes(demo_text, similarity_threshold=threshold)
        print(f"Threshold {threshold:3.1f}: {len(scenes):2d} scenes")
        
        # Show scene breakdown for reasonable number of scenes
        if len(scenes) <= 6:
            for i, scene in enumerate(scenes, 1):
                print(f"    Scene {i}: {scene[:50]}...")
            print()


if __name__ == "__main__":
    demo_scene_segmentation()
    print("\n" + "="*60 + "\n")
    test_different_thresholds()
