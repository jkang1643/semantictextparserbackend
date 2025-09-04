#!/usr/bin/env python3
"""
Sensitivity test for robust scene segmenter parameters.
Tests different parameter combinations to find optimal ranges.
"""

import json
from scene_segmenter import segment_scenes
import numpy as np

def test_hysteresis_sensitivity():
    """Test hysteresis parameter sensitivity."""
    demo_text = """John entered the forest. The trees loomed overhead.
The forest was alive with sounds. Strange markings appeared on the trees.
Suddenly, the temperature dropped. Ahead, he heard a sound.
A woman emerged, dressed in white. Her eyes glowed faintly.
She spoke softly. Then she added another line.
John felt a magical pull toward her."""
    
    print("=== HYSTERESIS SENSITIVITY TEST ===")
    print("Testing different hysteresis thresholds...")
    print()
    
    results = []
    
    # Test different hysteresis values
    for stay_inside in np.arange(0.1, 1.0, 0.1):
        for enter_boundary in np.arange(stay_inside + 0.1, 1.0, 0.1):
            if enter_boundary <= stay_inside:
                continue
                
            try:
                scenes = segment_scenes(
                    demo_text,
                    hysteresis=(stay_inside, enter_boundary),
                    min_sentences=2
                )
                
                result = {
                    'stay_inside': round(stay_inside, 1),
                    'enter_boundary': round(enter_boundary, 1),
                    'num_scenes': len(scenes),
                    'scene_lengths': [len(scene['text'].split('.')) - 1 for scene in scenes],
                    'scenes': [scene['text'] for scene in scenes]
                }
                results.append(result)
                
                print(f"Thresholds: ({stay_inside:.1f}, {enter_boundary:.1f}) -> {len(scenes)} scenes")
                
            except Exception as e:
                print(f"Error with thresholds ({stay_inside:.1f}, {enter_boundary:.1f}): {e}")
    
    return results

def test_weights_sensitivity():
    """Test weight parameter sensitivity."""
    demo_text = """John entered the forest. The trees loomed overhead.
The forest was alive with sounds. Strange markings appeared on the trees.
Suddenly, the temperature dropped. Ahead, he heard a sound.
A woman emerged, dressed in white. Her eyes glowed faintly.
She spoke softly. Then she added another line.
John felt a magical pull toward her."""
    
    print("\n=== WEIGHTS SENSITIVITY TEST ===")
    print("Testing different weight combinations...")
    print()
    
    results = []
    
    # Test different weight combinations
    weight_configs = [
        {"semantic": 0.8, "entity": 0.1, "visual": 0.05, "cue": 0.05},  # Semantic heavy
        {"semantic": 0.5, "entity": 0.3, "visual": 0.15, "cue": 0.05},  # Balanced
        {"semantic": 0.3, "entity": 0.5, "visual": 0.15, "cue": 0.05},  # Entity heavy
        {"semantic": 0.4, "entity": 0.2, "visual": 0.3, "cue": 0.1},    # Visual heavy
        {"semantic": 0.6, "entity": 0.2, "visual": 0.1, "cue": 0.1},    # Cue heavy
        {"semantic": 0.7, "entity": 0.2, "visual": 0.05, "cue": 0.05},  # Default-like
    ]
    
    for i, weights in enumerate(weight_configs):
        try:
            scenes = segment_scenes(
                demo_text,
                weights=weights,
                min_sentences=2
            )
            
            result = {
                'weights': weights,
                'num_scenes': len(scenes),
                'scene_lengths': [len(scene['text'].split('.')) - 1 for scene in scenes],
                'scenes': [scene['text'] for scene in scenes]
            }
            results.append(result)
            
            print(f"Config {i+1}: {weights} -> {len(scenes)} scenes")
            
        except Exception as e:
            print(f"Error with weights {weights}: {e}")
    
    return results

def test_target_scenes_sensitivity():
    """Test target scenes parameter sensitivity."""
    demo_text = """John entered the forest. The trees loomed overhead.
The forest was alive with sounds. Strange markings appeared on the trees.
Suddenly, the temperature dropped. Ahead, he heard a sound.
A woman emerged, dressed in white. Her eyes glowed faintly.
She spoke softly. Then she added another line.
John felt a magical pull toward her."""
    
    print("\n=== TARGET SCENES SENSITIVITY TEST ===")
    print("Testing different target scene counts...")
    print()
    
    results = []
    
    # Test different target scene counts
    for target in range(2, 11):
        try:
            scenes = segment_scenes(
                demo_text,
                target_scenes=target,
                min_sentences=2
            )
            
            result = {
                'target_scenes': target,
                'actual_scenes': len(scenes),
                'scene_lengths': [len(scene['text'].split('.')) - 1 for scene in scenes],
                'scenes': [scene['text'] for scene in scenes]
            }
            results.append(result)
            
            print(f"Target: {target} -> Actual: {len(scenes)} scenes")
            
        except Exception as e:
            print(f"Error with target {target}: {e}")
    
    return results

def test_min_sentences_sensitivity():
    """Test minimum sentences parameter sensitivity."""
    demo_text = """John entered the forest. The trees loomed overhead.
The forest was alive with sounds. Strange markings appeared on the trees.
Suddenly, the temperature dropped. Ahead, he heard a sound.
A woman emerged, dressed in white. Her eyes glowed faintly.
She spoke softly. Then she added another line.
John felt a magical pull toward her."""
    
    print("\n=== MIN SENTENCES SENSITIVITY TEST ===")
    print("Testing different minimum sentence requirements...")
    print()
    
    results = []
    
    # Test different minimum sentence requirements
    for min_sent in range(1, 6):
        try:
            scenes = segment_scenes(
                demo_text,
                min_sentences=min_sent,
                hysteresis=(0.5, 0.7)
            )
            
            result = {
                'min_sentences': min_sent,
                'num_scenes': len(scenes),
                'scene_lengths': [len(scene['text'].split('.')) - 1 for scene in scenes],
                'scenes': [scene['text'] for scene in scenes]
            }
            results.append(result)
            
            print(f"Min sentences: {min_sent} -> {len(scenes)} scenes")
            
        except Exception as e:
            print(f"Error with min_sentences {min_sent}: {e}")
    
    return results

def analyze_results(hysteresis_results, weights_results, target_results, min_sent_results):
    """Analyze and summarize the test results."""
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*60)
    
    # Hysteresis analysis
    print("\n1. HYSTERESIS ANALYSIS:")
    scene_counts = [r['num_scenes'] for r in hysteresis_results]
    print(f"   Scene count range: {min(scene_counts)} - {max(scene_counts)}")
    print(f"   Most common count: {max(set(scene_counts), key=scene_counts.count)}")
    
    # Find parameter ranges that give 2-3 scenes
    low_scene_results = [r for r in hysteresis_results if 2 <= r['num_scenes'] <= 3]
    print(f"   Configurations giving 2-3 scenes: {len(low_scene_results)}")
    if low_scene_results:
        print("   Examples:")
        for r in low_scene_results[:3]:
            print(f"     Thresholds ({r['stay_inside']}, {r['enter_boundary']}) -> {r['num_scenes']} scenes")
    
    # Find parameter ranges that give 8-10 scenes
    high_scene_results = [r for r in hysteresis_results if 8 <= r['num_scenes'] <= 10]
    print(f"   Configurations giving 8-10 scenes: {len(high_scene_results)}")
    if high_scene_results:
        print("   Examples:")
        for r in high_scene_results[:3]:
            print(f"     Thresholds ({r['stay_inside']}, {r['enter_boundary']}) -> {r['num_scenes']} scenes")
    
    # Weights analysis
    print("\n2. WEIGHTS ANALYSIS:")
    for i, r in enumerate(weights_results):
        print(f"   Config {i+1}: {r['weights']} -> {r['num_scenes']} scenes")
    
    # Target scenes analysis
    print("\n3. TARGET SCENES ANALYSIS:")
    for r in target_results:
        print(f"   Target {r['target_scenes']} -> Actual {r['actual_scenes']} scenes")
    
    # Min sentences analysis
    print("\n4. MIN SENTENCES ANALYSIS:")
    for r in min_sent_results:
        print(f"   Min {r['min_sentences']} -> {r['num_scenes']} scenes")
    
    # Recommendations
    print("\n5. RECOMMENDATIONS:")
    
    # Find optimal ranges
    optimal_low = [r for r in hysteresis_results if 2 <= r['num_scenes'] <= 3]
    optimal_high = [r for r in hysteresis_results if 8 <= r['num_scenes'] <= 10]
    
    if optimal_low:
        avg_stay = np.mean([r['stay_inside'] for r in optimal_low])
        avg_enter = np.mean([r['enter_boundary'] for r in optimal_low])
        print(f"   For 2-3 scenes: Use hysteresis around ({avg_stay:.2f}, {avg_enter:.2f})")
    
    if optimal_high:
        avg_stay = np.mean([r['stay_inside'] for r in optimal_high])
        avg_enter = np.mean([r['enter_boundary'] for r in optimal_high])
        print(f"   For 8-10 scenes: Use hysteresis around ({avg_stay:.2f}, {avg_enter:.2f})")
    
    # Check if we need algorithm tweaks
    max_scenes = max(scene_counts)
    min_scenes = min(scene_counts)
    print(f"\n   Current range: {min_scenes} - {max_scenes} scenes")
    
    if max_scenes < 8:
        print("   ⚠️  WARNING: Maximum scene count is below target range (8-10)")
        print("   💡 RECOMMENDATION: Consider tweaking algorithm to be more sensitive")
    elif min_scenes > 3:
        print("   ⚠️  WARNING: Minimum scene count is above target range (2-3)")
        print("   💡 RECOMMENDATION: Consider tweaking algorithm to be less sensitive")
    else:
        print("   ✅ Algorithm covers target range well!")

def save_detailed_results(hysteresis_results, weights_results, target_results, min_sent_results):
    """Save detailed results to JSON file."""
    all_results = {
        'hysteresis_test': hysteresis_results,
        'weights_test': weights_results,
        'target_scenes_test': target_results,
        'min_sentences_test': min_sent_results,
        'test_info': {
            'demo_text': """John entered the forest. The trees loomed overhead.
The forest was alive with sounds. Strange markings appeared on the trees.
Suddenly, the temperature dropped. Ahead, he heard a sound.
A woman emerged, dressed in white. Her eyes glowed faintly.
She spoke softly. Then she added another line.
John felt a magical pull toward her.""",
            'total_sentences': 10
        }
    }
    
    with open('sensitivity_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📊 Detailed results saved to 'sensitivity_test_results.json'")

def main():
    """Run the complete sensitivity test suite."""
    print("🔬 ROBUST SCENE SEGMENTER SENSITIVITY TEST")
    print("=" * 50)
    print("Testing parameter sensitivity to find optimal ranges...")
    print("Target: 2-3 scenes (low granularity) to 8-10 scenes (high granularity)")
    print()
    
    # Run all tests
    hysteresis_results = test_hysteresis_sensitivity()
    weights_results = test_weights_sensitivity()
    target_results = test_target_scenes_sensitivity()
    min_sent_results = test_min_sentences_sensitivity()
    
    # Analyze results
    analyze_results(hysteresis_results, weights_results, target_results, min_sent_results)
    
    # Save detailed results
    save_detailed_results(hysteresis_results, weights_results, target_results, min_sent_results)
    
    print("\n✅ Sensitivity test completed!")

if __name__ == "__main__":
    main()
