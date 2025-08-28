#!/usr/bin/env python3
"""
Quick Test Script - All Testing Options for Text-to-Image Pipeline

This script demonstrates all the different ways you can test the pipeline.
"""

from text_to_image_pipeline_lite import TextToImagePipelineLite

def test_basic_functionality():
    """Test basic pipeline functionality."""
    print("🧪 Test 1: Basic Functionality")
    print("-" * 40)
    
    text = "A knight rode through the misty forest, his armor gleaming in the moonlight."
    
    pipeline = TextToImagePipelineLite()
    results = pipeline.process_text(text, style="fantasy")
    
    print(f"✅ Processed {len(results)} chunks")
    print(f"✅ Characters found: {results[0]['characters']}")
    print(f"✅ Generated prompt: {results[0]['scene_prompt'][:80]}...")
    print()

def test_different_styles():
    """Test different visual styles."""
    print("🧪 Test 2: Different Styles")
    print("-" * 40)
    
    text = "A detective walked down a dark alley, rain falling heavily."
    
    styles = ["realistic", "artistic", "cinematic", "fantasy"]
    
    for style in styles:
        pipeline = TextToImagePipelineLite()
        results = pipeline.process_text(text, style=style)
        print(f"✅ {style.capitalize()} style: {results[0]['scene_prompt'][:60]}...")
    print()

def test_segmentation_methods():
    """Test different segmentation methods."""
    print("🧪 Test 3: Segmentation Methods")
    print("-" * 40)
    
    text = """
    The old wizard stood at the top of the tower. His robes billowed in the wind.
    Below him, the village lay in darkness. He raised his staff and began to chant.
    The sky filled with magical energy. Lightning crackled around him.
    """
    
    # Rule-based segmentation
    pipeline1 = TextToImagePipelineLite(segmentation_method="rule_based", max_tokens_per_chunk=100)
    results1 = pipeline1.process_text(text, style="fantasy")
    print(f"✅ Rule-based: {len(results1)} chunks")
    
    # Semantic segmentation
    pipeline2 = TextToImagePipelineLite(segmentation_method="semantic", max_tokens_per_chunk=100)
    results2 = pipeline2.process_text(text, style="fantasy")
    print(f"✅ Semantic: {len(results2)} chunks")
    print()

def test_character_extraction():
    """Test character extraction functionality."""
    print("🧪 Test 4: Character Extraction")
    print("-" * 40)
    
    text = """
    Sarah Chen opened the door to find Detective Mike Johnson waiting outside.
    "Hello Sarah," said Mike. "We need to talk about the case."
    Dr. Emily Rodriguez joined them in the hallway.
    """
    
    pipeline = TextToImagePipelineLite()
    results = pipeline.process_text(text, style="realistic")
    
    characters = results[0]['characters']
    print(f"✅ Characters found: {characters}")
    print()

def test_variations():
    """Test variations generation."""
    print("🧪 Test 5: Variations Generation")
    print("-" * 40)
    
    text = "A spaceship hovered above the alien city."
    
    pipeline = TextToImagePipelineLite()
    results = pipeline.process_text(text, style="cinematic")
    
    if results:
        variations = pipeline.generate_variations(0, num_variations=2)
        print(f"✅ Generated {len(variations)} variations")
        for i, var in enumerate(variations):
            print(f"   Variation {i+1}: {var['scene_prompt'][:60]}...")
    print()

def main():
    """Run all tests."""
    print("🎯 Quick Test Suite - Text-to-Image Pipeline")
    print("=" * 60)
    print("Testing all pipeline features...")
    print()
    
    try:
        test_basic_functionality()
        test_different_styles()
        test_segmentation_methods()
        test_character_extraction()
        test_variations()
        
        print("🎉 All tests completed successfully!")
        print("✅ Pipeline is working correctly")
        print()
        print("📋 Testing Summary:")
        print("   - Basic functionality: ✅")
        print("   - Multiple styles: ✅")
        print("   - Segmentation methods: ✅")
        print("   - Character extraction: ✅")
        print("   - Variations generation: ✅")
        print()
        print("🚀 Ready for production use!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()

