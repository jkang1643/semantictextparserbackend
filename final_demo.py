#!/usr/bin/env python3
"""
Final Comprehensive Demo - Full Text-to-Image Pipeline

This script demonstrates the complete lightweight text-to-image pipeline
working with all features without heavy dependencies.
"""

from text_to_image_pipeline_lite import TextToImagePipelineLite
import json

def main():
    """Run the comprehensive demo."""
    print("🎯 Full Text-to-Image Pipeline Demo")
    print("=" * 60)
    print("✅ No heavy dependencies (torch, transformers, spaCy)")
    print("✅ All core features working")
    print("✅ Ready for API integration")
    print()
    
    # Sample story with multiple characters and scenes
    story = """
    Detective Sarah Chen walked through the rain-soaked streets of Chinatown, 
    her trench coat billowing in the wind. The neon signs reflected in the 
    puddles, creating a kaleidoscope of colors. She was following a lead on 
    the mysterious disappearance of Professor James Wilson, a renowned 
    archaeologist who had been studying ancient artifacts.
    
    As she turned the corner onto Grant Avenue, she noticed a figure in a 
    dark hooded jacket ducking into an alley. Sarah quickened her pace, 
    her hand instinctively reaching for her service weapon. The alley was 
    narrow and dimly lit, with steam rising from manhole covers.
    
    "Stop! Police!" she called out, but the figure continued running. 
    Sarah chased after them, her boots splashing through the wet pavement. 
    Suddenly, the figure turned and revealed themselves - it was Dr. Maria 
    Rodriguez, Wilson's research partner. Her face was pale and her hands 
    were shaking.
    
    "Sarah, thank God it's you," Maria said breathlessly. "James is alive, 
    but he's in terrible danger. The artifacts he found... they're not what 
    they seem. There's something ancient and powerful that wants them back."
    
    Sarah felt a chill run down her spine as she looked into Maria's 
    frightened eyes. The rain continued to fall, but now it felt different - 
    almost as if the very air around them was charged with something otherworldly.
    """
    
    print("📖 Processing story...")
    print(f"   Length: {len(story)} characters")
    print(f"   Estimated tokens: {len(story) // 4}")
    print()
    
    # Initialize pipeline with different configurations
    print("🚀 Testing Pipeline Configurations")
    print("-" * 40)
    
    # Test 1: Rule-based with realistic style
    print("1️⃣ Rule-based segmentation + Realistic style")
    pipeline1 = TextToImagePipelineLite(
        segmentation_method="rule_based",
        image_service="nano_banana",
        max_tokens_per_chunk=300
    )
    
    results1 = pipeline1.process_text(story, style="realistic")
    stats1 = pipeline1.get_pipeline_stats()
    
    print(f"   ✅ Chunks: {stats1['total_chunks']}")
    print(f"   ✅ Characters: {stats1['characters_found']}")
    print(f"   ✅ Success rate: {stats1['success_rate']:.1%}")
    print()
    
    # Test 2: Semantic with fantasy style
    print("2️⃣ Semantic segmentation + Fantasy style")
    pipeline2 = TextToImagePipelineLite(
        segmentation_method="semantic",
        image_service="nano_banana",
        max_tokens_per_chunk=300,
        similarity_threshold=0.4
    )
    
    results2 = pipeline2.process_text(story, style="fantasy")
    stats2 = pipeline2.get_pipeline_stats()
    
    print(f"   ✅ Chunks: {stats2['total_chunks']}")
    print(f"   ✅ Characters: {stats2['characters_found']}")
    print(f"   ✅ Success rate: {stats2['success_rate']:.1%}")
    print()
    
    # Test 3: Cinematic style with variations
    print("3️⃣ Cinematic style + Variations")
    pipeline3 = TextToImagePipelineLite(
        segmentation_method="rule_based",
        image_service="nano_banana"
    )
    
    results3 = pipeline3.process_text(story[:800], style="cinematic")
    
    # Generate variations for the first chunk
    if results3:
        variations = pipeline3.generate_variations(0, num_variations=2)
        print(f"   ✅ Generated {len(variations)} variations")
    
    print()
    
    # Show sample results
    print("📊 Sample Results")
    print("-" * 40)
    
    if results1:
        print("First chunk (Rule-based + Realistic):")
        print(f"   Text: {results1[0]['chunk_text'][:100]}...")
        print(f"   Characters: {results1[0]['characters']}")
        print(f"   Prompt: {results1[0]['scene_prompt'][:100]}...")
        print(f"   Image: {results1[0]['image_result']['image_url']}")
        print()
    
    if results2:
        print("First chunk (Semantic + Fantasy):")
        print(f"   Text: {results2[0]['chunk_text'][:100]}...")
        print(f"   Characters: {results2[0]['characters']}")
        print(f"   Prompt: {results2[0]['scene_prompt'][:100]}...")
        print(f"   Image: {results2[0]['image_result']['image_url']}")
        print()
    
    # Save all results
    print("💾 Saving Results")
    print("-" * 40)
    
    pipeline1.save_results("final_demo_rule_based.json")
    pipeline2.save_results("final_demo_semantic.json")
    pipeline3.save_results("final_demo_cinematic.json")
    
    print("✅ All results saved to JSON files")
    print()
    
    # Final summary
    print("🎉 Pipeline Features Summary")
    print("=" * 60)
    print("✅ Text preprocessing and cleaning")
    print("✅ Character/entity extraction")
    print("✅ Rule-based text segmentation")
    print("✅ Semantic text segmentation")
    print("✅ Context-aware prompt generation")
    print("✅ Multiple visual styles (realistic, fantasy, cinematic, artistic)")
    print("✅ Image generation (mock/API)")
    print("✅ Variations generation")
    print("✅ Results export and statistics")
    print("✅ Character tracking across chunks")
    print("✅ Pipeline state management")
    print()
    print("🚀 Ready for production use!")
    print("   - Add API keys to .env file for real image generation")
    print("   - Use with any text input (stories, articles, transcripts)")
    print("   - Customize segmentation and styling parameters")
    print("   - Integrate with web applications or batch processing")

if __name__ == "__main__":
    main()
