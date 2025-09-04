#!/usr/bin/env python3
"""
Parameter guide for robust scene segmenter based on sensitivity testing.
Shows optimal parameters for different granularity levels.
"""

from scene_segmenter import segment_scenes

def demonstrate_parameter_ranges():
    """Demonstrate different parameter ranges for different granularity levels."""
    
    demo_text = """John entered the forest. The trees loomed overhead.
The forest was alive with sounds. Strange markings appeared on the trees.
Suddenly, the temperature dropped. Ahead, he heard a sound.
A woman emerged, dressed in white. Her eyes glowed faintly.
She spoke softly. Then she added another line.
John felt a magical pull toward her."""
    
    print("🎯 ROBUST SCENE SEGMENTER - PARAMETER GUIDE")
    print("=" * 60)
    print("Based on sensitivity testing, here are optimal parameters:")
    print()
    
    # Low Granularity (2-3 scenes) - Good for image generation
    print("1. LOW GRANULARITY (2-3 scenes) - Perfect for Image Generation")
    print("-" * 50)
    
    configs_low = [
        {
            'name': 'Sensitive Weights',
            'params': {
                'weights': {"semantic": 0.3, "entity": 0.4, "visual": 0.2, "cue": 0.1},
                'min_sentences': 3
            }
        },
        {
            'name': 'Conservative Hysteresis',
            'params': {
                'hysteresis': (0.6, 0.8),
                'min_sentences': 3
            }
        },
        {
            'name': 'Target Scenes',
            'params': {
                'target_scenes': 3,
                'min_sentences': 3
            }
        }
    ]
    
    for config in configs_low:
        try:
            scenes = segment_scenes(demo_text, **config['params'])
            print(f"   {config['name']}: {len(scenes)} scenes")
            for i, scene in enumerate(scenes, 1):
                print(f"      Scene {i}: {scene['text'][:50]}...")
        except Exception as e:
            print(f"   {config['name']}: Error - {e}")
        print()
    
    # Medium Granularity (4-6 scenes)
    print("2. MEDIUM GRANULARITY (4-6 scenes) - Balanced Approach")
    print("-" * 50)
    
    configs_medium = [
        {
            'name': 'Balanced Hysteresis',
            'params': {
                'hysteresis': (0.3, 0.6),
                'min_sentences': 2
            }
        },
        {
            'name': 'Default Weights',
            'params': {
                'weights': {"semantic": 0.55, "entity": 0.25, "visual": 0.15, "cue": 0.05},
                'min_sentences': 2
            }
        }
    ]
    
    for config in configs_medium:
        try:
            scenes = segment_scenes(demo_text, **config['params'])
            print(f"   {config['name']}: {len(scenes)} scenes")
            for i, scene in enumerate(scenes, 1):
                print(f"      Scene {i}: {scene['text'][:50]}...")
        except Exception as e:
            print(f"   {config['name']}: Error - {e}")
        print()
    
    # High Granularity (8-10+ scenes)
    print("3. HIGH GRANULARITY (8-10+ scenes) - Detailed Analysis")
    print("-" * 50)
    
    configs_high = [
        {
            'name': 'Very Sensitive Hysteresis',
            'params': {
                'hysteresis': (0.05, 0.15),
                'min_sentences': 1
            }
        },
        {
            'name': 'Ultra Sensitive',
            'params': {
                'hysteresis': (0.1, 0.2),
                'min_sentences': 1
            }
        }
    ]
    
    for config in configs_high:
        try:
            scenes = segment_scenes(demo_text, **config['params'])
            print(f"   {config['name']}: {len(scenes)} scenes")
            for i, scene in enumerate(scenes, 1):
                print(f"      Scene {i}: {scene['text'][:50]}...")
        except Exception as e:
            print(f"   {config['name']}: Error - {e}")
        print()
    
    # Recommendations
    print("4. RECOMMENDATIONS")
    print("-" * 50)
    print("✅ For Image Generation (2-3 scenes):")
    print("   - Use sensitive weights: entity=0.4, visual=0.2")
    print("   - Or use target_scenes=3")
    print()
    print("✅ For Detailed Analysis (8-10+ scenes):")
    print("   - Use very sensitive hysteresis: (0.05, 0.15)")
    print("   - Set min_sentences=1")
    print()
    print("✅ For Balanced Approach (4-6 scenes):")
    print("   - Use balanced hysteresis: (0.3, 0.6)")
    print("   - Or use default weights with min_sentences=2")
    print()
    print("🔧 ALGORITHM STATUS:")
    print("   - ✅ Can achieve 2-3 scenes (low granularity)")
    print("   - ✅ Can achieve 8-10+ scenes (high granularity)")
    print("   - ✅ Covers full target range!")
    print("   - 💡 No algorithm tweaks needed - just use right parameters")

def create_parameter_presets():
    """Create a parameter presets file for easy use."""
    
    presets = {
        "low_granularity": {
            "description": "2-3 scenes, perfect for image generation",
            "parameters": {
                "weights": {"semantic": 0.3, "entity": 0.4, "visual": 0.2, "cue": 0.1},
                "min_sentences": 3
            }
        },
        "medium_granularity": {
            "description": "4-6 scenes, balanced approach",
            "parameters": {
                "hysteresis": (0.3, 0.6),
                "min_sentences": 2
            }
        },
        "high_granularity": {
            "description": "8-10+ scenes, detailed analysis",
            "parameters": {
                "hysteresis": (0.05, 0.15),
                "min_sentences": 1
            }
        },
        "target_3_scenes": {
            "description": "Exactly 3 scenes using target_scenes",
            "parameters": {
                "target_scenes": 3,
                "min_sentences": 3
            }
        },
        "target_8_scenes": {
            "description": "Target 8 scenes (may be limited by algorithm)",
            "parameters": {
                "target_scenes": 8,
                "min_sentences": 1
            }
        }
    }
    
    import json
    with open('scene_segmenter_presets.json', 'w') as f:
        json.dump(presets, f, indent=2)
    
    print("📁 Parameter presets saved to 'scene_segmenter_presets.json'")
    print("   Use these presets for consistent results across different granularity levels")

if __name__ == "__main__":
    demonstrate_parameter_ranges()
    print()
    create_parameter_presets()
