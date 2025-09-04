#!/usr/bin/env python3
"""
Focused sensitivity test for robust scene segmenter.
Tests key parameter ranges to achieve 2-3 to 8-10 scenes.
"""

import json
from scene_segmenter import segment_scenes

def test_key_parameters():
    """Test key parameter combinations to find optimal ranges."""
    demo_text = """John entered the forest. The trees loomed overhead.
The forest was alive with sounds. Strange markings appeared on the trees.
Suddenly, the temperature dropped. Ahead, he heard a sound.
A woman emerged, dressed in white. Her eyes glowed faintly.
She spoke softly. Then she added another line.
John felt a magical pull toward her."""
    
    print("🔬 FOCUSED SENSITIVITY TEST")
    print("=" * 40)
    print("Testing key parameters to achieve 2-3 to 8-10 scenes")
    print()
    
    results = []
    
    # Test 1: Very sensitive hysteresis (should give more scenes)
    print("1. TESTING VERY SENSITIVE HYSTERESIS:")
    sensitive_configs = [
        (0.05, 0.15), (0.05, 0.20), (0.05, 0.25),
        (0.10, 0.20), (0.10, 0.25), (0.10, 0.30),
        (0.15, 0.25), (0.15, 0.30), (0.15, 0.35),
    ]
    
    for stay, enter in sensitive_configs:
        try:
            scenes = segment_scenes(
                demo_text,
                hysteresis=(stay, enter),
                min_sentences=1  # Lower minimum for more sensitivity
            )
            result = {
                'type': 'hysteresis_sensitive',
                'stay_inside': stay,
                'enter_boundary': enter,
                'min_sentences': 1,
                'num_scenes': len(scenes),
                'scenes': [scene['text'] for scene in scenes]
            }
            results.append(result)
            print(f"   ({stay:.2f}, {enter:.2f}) -> {len(scenes)} scenes")
        except Exception as e:
            print(f"   Error with ({stay:.2f}, {enter:.2f}): {e}")
    
    print()
    
    # Test 2: Different weight configurations for more sensitivity
    print("2. TESTING SENSITIVE WEIGHT CONFIGURATIONS:")
    sensitive_weights = [
        {"semantic": 0.3, "entity": 0.4, "visual": 0.2, "cue": 0.1},  # Entity + visual heavy
        {"semantic": 0.2, "entity": 0.5, "visual": 0.2, "cue": 0.1},  # Very entity heavy
        {"semantic": 0.4, "entity": 0.3, "visual": 0.2, "cue": 0.1},  # Balanced sensitive
        {"semantic": 0.5, "entity": 0.2, "visual": 0.2, "cue": 0.1},  # Semantic + visual
    ]
    
    for i, weights in enumerate(sensitive_weights):
        try:
            scenes = segment_scenes(
                demo_text,
                weights=weights,
                min_sentences=1
            )
            result = {
                'type': 'weights_sensitive',
                'weights': weights,
                'min_sentences': 1,
                'num_scenes': len(scenes),
                'scenes': [scene['text'] for scene in scenes]
            }
            results.append(result)
            print(f"   Config {i+1}: {weights} -> {len(scenes)} scenes")
        except Exception as e:
            print(f"   Error with weights {weights}: {e}")
    
    print()
    
    # Test 3: Very low minimum sentences
    print("3. TESTING VERY LOW MINIMUM SENTENCES:")
    for min_sent in [1, 2]:
        try:
            scenes = segment_scenes(
                demo_text,
                min_sentences=min_sent,
                hysteresis=(0.1, 0.3)  # Sensitive thresholds
            )
            result = {
                'type': 'min_sentences',
                'min_sentences': min_sent,
                'hysteresis': (0.1, 0.3),
                'num_scenes': len(scenes),
                'scenes': [scene['text'] for scene in scenes]
            }
            results.append(result)
            print(f"   Min sentences: {min_sent} -> {len(scenes)} scenes")
        except Exception as e:
            print(f"   Error with min_sentences {min_sent}: {e}")
    
    print()
    
    # Test 4: Target scenes to see if we can force more scenes
    print("4. TESTING TARGET SCENES (8-10):")
    for target in [8, 9, 10]:
        try:
            scenes = segment_scenes(
                demo_text,
                target_scenes=target,
                min_sentences=1
            )
            result = {
                'type': 'target_scenes',
                'target_scenes': target,
                'actual_scenes': len(scenes),
                'min_sentences': 1,
                'scenes': [scene['text'] for scene in scenes]
            }
            results.append(result)
            print(f"   Target: {target} -> Actual: {len(scenes)} scenes")
        except Exception as e:
            print(f"   Error with target {target}: {e}")
    
    return results

def analyze_results(results):
    """Analyze the test results."""
    print("\n" + "="*50)
    print("ANALYSIS")
    print("="*50)
    
    # Group by type
    hysteresis_results = [r for r in results if r['type'] == 'hysteresis_sensitive']
    weights_results = [r for r in results if r['type'] == 'weights_sensitive']
    min_sent_results = [r for r in results if r['type'] == 'min_sentences']
    target_results = [r for r in results if r['type'] == 'target_scenes']
    
    # Find the range
    all_scene_counts = [r['num_scenes'] for r in results]
    min_scenes = min(all_scene_counts)
    max_scenes = max(all_scene_counts)
    
    print(f"Scene count range: {min_scenes} - {max_scenes}")
    print()
    
    # Find configurations that give us our target ranges
    low_granularity = [r for r in results if 2 <= r['num_scenes'] <= 3]
    high_granularity = [r for r in results if 8 <= r['num_scenes'] <= 10]
    
    print("CONFIGURATIONS FOR TARGET RANGES:")
    print()
    
    if low_granularity:
        print("✅ 2-3 scenes (low granularity):")
        for r in low_granularity:
            if r['type'] == 'hysteresis_sensitive':
                print(f"   Hysteresis ({r['stay_inside']:.2f}, {r['enter_boundary']:.2f})")
            elif r['type'] == 'weights_sensitive':
                print(f"   Weights: {r['weights']}")
            elif r['type'] == 'min_sentences':
                print(f"   Min sentences: {r['min_sentences']}")
    else:
        print("❌ No configurations found for 2-3 scenes")
    
    print()
    
    if high_granularity:
        print("✅ 8-10 scenes (high granularity):")
        for r in high_granularity:
            if r['type'] == 'hysteresis_sensitive':
                print(f"   Hysteresis ({r['stay_inside']:.2f}, {r['enter_boundary']:.2f})")
            elif r['type'] == 'weights_sensitive':
                print(f"   Weights: {r['weights']}")
            elif r['type'] == 'target_scenes':
                print(f"   Target scenes: {r['target_scenes']} (actual: {r['actual_scenes']})")
    else:
        print("❌ No configurations found for 8-10 scenes")
        print("   This suggests the algorithm needs tweaking for higher sensitivity")
    
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    if max_scenes < 8:
        print("🔧 ALGORITHM TWEAKS NEEDED:")
        print("   1. Lower the default hysteresis thresholds")
        print("   2. Increase weight on entity and visual novelty")
        print("   3. Reduce minimum sentences requirement")
        print("   4. Add more aggressive peak detection for target scenes")
    else:
        print("✅ Algorithm can achieve target range with proper parameters")
    
    # Show some example scenes for analysis
    print("\nEXAMPLE SCENE BREAKDOWN:")
    if results:
        best_result = max(results, key=lambda x: x['num_scenes'])
        print(f"Most granular result ({best_result['num_scenes']} scenes):")
        for i, scene in enumerate(best_result['scenes'], 1):
            print(f"   Scene {i}: {scene}")

def main():
    """Run the focused sensitivity test."""
    results = test_key_parameters()
    analyze_results(results)
    
    # Save results
    with open('focused_sensitivity_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to 'focused_sensitivity_results.json'")

if __name__ == "__main__":
    main()
