#!/usr/bin/env python3
"""
Test script for the revised hysteresis algorithm with improved scene segmentation
and image prompt generation using the nano banana image generation model.
"""

import os
import sys
from scene_segmenter import SceneSegmenter
from image_generator import ImageGenerator
import json
from datetime import datetime

def test_revised_algorithm():
    """Test the revised algorithm with sample text."""
    
    # Sample text for testing
    sample_text = """
    John walked through the dark forest, his footsteps muffled by the thick layer of fallen leaves. 
    The ancient trees seemed to whisper secrets as the wind rustled through their branches. 
    He carried a worn leather satchel containing his most precious belongings.
    
    Sarah sat by the crackling fireplace in her cozy cottage, reading an old book by candlelight. 
    The warm glow illuminated her face as she turned the yellowed pages carefully. 
    Outside, rain pattered against the window panes.
    
    The old man stood on the mountain peak, gazing at the vast valley below. 
    His weathered hands gripped the wooden staff as he surveyed the landscape. 
    The morning sun cast long shadows across the rocky terrain.
    """
    
    print("🧪 Testing Revised Hysteresis Algorithm")
    print("=" * 50)
    
    # Initialize the scene segmenter
    print("📝 Initializing scene segmenter...")
    segmenter = SceneSegmenter()
    
    # Initialize the image generator (nano banana)
    print("🎨 Initializing nano banana image generator...")
    image_generator = ImageGenerator(service="nano_banana")
    
    # Segment the text into scenes
    print("🔍 Segmenting text into scenes...")
    scenes = segmenter.segment_scenes(
        text=sample_text,
        target_scenes=3,
        min_sentences=2,
        hysteresis=(0.58, 0.68)
    )
    
    print(f"✅ Found {len(scenes)} scenes")
    print()
    
    # Process each scene
    results = []
    for i, scene in enumerate(scenes):
        print(f"🎬 Scene {i+1}:")
        print(f"   Text: {scene['text'][:100]}...")
        print(f"   Character: {scene.get('character', 'Unknown')}")
        print(f"   Summary: {scene['summary']}")
        print(f"   Prompt: {scene['prompt']}")
        print()
        
        # Generate image for this scene
        print(f"   🎨 Generating image for scene {i+1}...")
        try:
            image_result = image_generator.generate_image(
                scene['prompt'],
                style="realistic"
            )
            
            if image_result.get('success'):
                print(f"   ✅ Image generated successfully")
                print(f"   📁 Saved to: {image_result.get('local_path', 'N/A')}")
            else:
                print(f"   ⚠️  Image generation failed: {image_result.get('metadata', {}).get('note', 'Unknown error')}")
            
            results.append({
                'scene_index': i + 1,
                'text': scene['text'],
                'character': scene.get('character'),
                'summary': scene['summary'],
                'prompt': scene['prompt'],
                'image_result': image_result
            })
            
        except Exception as e:
            print(f"   ❌ Error generating image: {e}")
            results.append({
                'scene_index': i + 1,
                'text': scene['text'],
                'character': scene.get('character'),
                'summary': scene['summary'],
                'prompt': scene['prompt'],
                'error': str(e)
            })
        
        print()
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"revised_algorithm_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {results_file}")
    
    # Print summary
    print("\n📊 Test Summary:")
    print(f"   Scenes processed: {len(scenes)}")
    print(f"   Images generated: {sum(1 for r in results if r.get('image_result', {}).get('success', False))}")
    print(f"   Errors: {sum(1 for r in results if 'error' in r)}")
    
    return results

def analyze_prompt_structure():
    """Analyze the structure of generated prompts."""
    
    print("\n🔍 Analyzing Prompt Structure:")
    print("=" * 40)
    
    # Test with different types of text
    test_cases = [
        {
            "name": "Location Present",
            "text": "John walked through the dark forest, carrying a wooden staff. The ancient trees towered above him."
        },
        {
            "name": "No Location",
            "text": "The old man sat quietly, reading his book by candlelight. The warm glow illuminated his weathered face."
        },
        {
            "name": "Multiple Characters",
            "text": "Sarah and John met in the garden. They discussed their plans for the journey ahead. The conversation was brief but meaningful."
        }
    ]
    
    segmenter = SceneSegmenter()
    
    for test_case in test_cases:
        print(f"\n📝 Test Case: {test_case['name']}")
        print(f"   Input: {test_case['text']}")
        
        scenes = segmenter.segment_scenes(test_case['text'], target_scenes=1)
        
        if scenes:
            scene = scenes[0]
            summary = scene['summary']
            prompt = scene['prompt']
            
            print(f"   Subjects: {summary.get('subjects', [])}")
            print(f"   Setting: {summary.get('setting', [])}")
            print(f"   Objects: {summary.get('objects', [])}")
            print(f"   Action Verbs: {summary.get('action_verbs', [])}")
            print(f"   Tone Adjectives: {summary.get('tone_adjs', [])}")
            print(f"   Generated Prompt: {prompt}")
            print(f"   Character: {scene.get('character', 'Unknown')}")
        else:
            print("   No scenes generated")

if __name__ == "__main__":
    print("🚀 Starting Revised Algorithm Test")
    print("=" * 50)
    
    try:
        # Run the main test
        results = test_revised_algorithm()
        
        # Analyze prompt structure
        analyze_prompt_structure()
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
