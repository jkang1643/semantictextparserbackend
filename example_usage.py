#!/usr/bin/env python3
"""
Example Usage Script for Text-to-Image Pipeline

This script demonstrates various ways to use the text-to-image pipeline
programmatically, including different configurations and use cases.
"""

from text_to_image_pipeline import TextToImagePipeline
from text_processor import TextProcessor
from text_segmenter import TextSegmenter
from prompt_generator import PromptGenerator
from image_generator import ImageGenerator

def example_basic_usage():
    """Basic usage example with default settings."""
    print("=== Basic Usage Example ===")
    
    # Sample text
    text = """
    Sarah opened the door to her grandmother's attic. Dust motes danced in the 
    afternoon sunlight that streamed through the small window. The air was thick 
    with the scent of old books and memories. She carefully stepped across the 
    creaking floorboards, her eyes scanning the piles of forgotten treasures.
    
    In the corner, she spotted an old wooden chest covered in intricate carvings. 
    The brass lock was tarnished with age, but the keyhole seemed to glow with 
    an otherworldly light. As she reached out to touch it, the chest began to 
    hum softly, and the carvings seemed to move in the flickering light.
    """
    
    # Initialize pipeline with default settings
    pipeline = TextToImagePipeline()
    
    # Process the text
    results = pipeline.process_text(text, style="realistic")
    
    # Print results
    print(f"Generated {len(results)} chunks:")
    for i, result in enumerate(results):
        print(f"\nChunk {i+1}:")
        print(f"  Text: {result['chunk_text'][:80]}...")
        print(f"  Prompt: {result['scene_prompt'][:80]}...")
        print(f"  Image URL: {result['image_result'].get('image_url', 'No image')}")

def example_advanced_configuration():
    """Example with custom configuration."""
    print("\n=== Advanced Configuration Example ===")
    
    text = """
    The cyberpunk city stretched endlessly into the neon-lit horizon. Holographic 
    advertisements flickered in the perpetual twilight, casting colorful shadows 
    on the rain-slicked streets. A lone figure in a tattered trench coat moved 
    through the crowds, their face hidden beneath a hood. The air crackled with 
    the energy of a thousand digital connections, and somewhere in the distance, 
    sirens wailed their electronic song.
    """
    
    # Custom pipeline configuration
    pipeline = TextToImagePipeline(
        segmentation_method="semantic",
        image_service="nano_banana",
        max_tokens_per_chunk=256,  # Smaller chunks for more detailed scenes
        similarity_threshold=0.7   # Higher threshold for more cohesive chunks
    )
    
    # Process with cinematic style
    results = pipeline.process_text(text, style="cinematic", save_images=True)
    
    print(f"Generated {len(results)} chunks with custom settings:")
    for i, result in enumerate(results):
        print(f"\nChunk {i+1}:")
        print(f"  Characters: {result['characters']}")
        print(f"  Prompt: {result['scene_prompt'][:100]}...")

def example_component_usage():
    """Example using individual components."""
    print("\n=== Individual Component Usage ===")
    
    text = """
    The ancient library was a maze of towering bookshelves that seemed to reach 
    into infinity. The air was thick with the scent of leather and parchment, 
    and the only sound was the soft whisper of turning pages. A librarian with 
    silver-rimmed glasses moved silently between the aisles, her robes flowing 
    like shadows behind her.
    """
    
    # Use components individually
    processor = TextProcessor()
    segmenter = TextSegmenter()
    prompt_gen = PromptGenerator()
    image_gen = ImageGenerator()
    
    # Step 1: Preprocess
    clean_text = processor.preprocess_text(text)
    print(f"Cleaned text: {clean_text[:100]}...")
    
    # Step 2: Extract entities
    entities = processor.extract_entities(clean_text)
    print(f"Entities found: {entities}")
    
    # Step 3: Segment
    chunks = segmenter.segment_text(clean_text, "semantic")
    print(f"Segmented into {len(chunks)} chunks")
    
    # Step 4: Generate prompts
    for i, chunk in enumerate(chunks):
        prompt = prompt_gen.generate_scene_prompt(chunk, "fantasy")
        print(f"\nChunk {i+1} prompt: {prompt[:100]}...")
        
        # Step 5: Generate image (mock for demo)
        image_result = image_gen.generate_image(prompt)
        print(f"Image result: {image_result.get('image_url', 'Mock image')}")

def example_batch_processing():
    """Example processing multiple texts."""
    print("\n=== Batch Processing Example ===")
    
    texts = [
        "A knight in shining armor rode through the misty forest.",
        "The spaceship hovered above the alien landscape, its engines humming softly.",
        "A chef prepared a feast in a bustling kitchen filled with steam and spices."
    ]
    
    pipeline = TextToImagePipeline()
    
    all_results = []
    for i, text in enumerate(texts):
        print(f"\nProcessing text {i+1}/{len(texts)}...")
        results = pipeline.process_text(text, style="artistic")
        all_results.extend(results)
    
    print(f"\nTotal chunks generated: {len(all_results)}")
    
    # Get overall statistics
    stats = pipeline.get_pipeline_stats()
    print(f"Overall success rate: {stats['success_rate']:.1%}")

def example_variations():
    """Example generating multiple variations."""
    print("\n=== Variations Example ===")
    
    text = "A mysterious door appeared in the middle of the forest."
    
    pipeline = TextToImagePipeline()
    results = pipeline.process_text(text, style="realistic")
    
    if results:
        # Generate variations for the first chunk
        variations = pipeline.generate_variations(0, num_variations=3)
        
        print(f"Generated {len(variations)} variations:")
        for i, variation in enumerate(variations):
            print(f"\nVariation {i+1}:")
            print(f"  Prompt: {variation['scene_prompt'][:80]}...")
            print(f"  Image: {variation['image_result'].get('image_url', 'Mock image')}")

def example_error_handling():
    """Example with error handling."""
    print("\n=== Error Handling Example ===")
    
    # Test with empty text
    pipeline = TextToImagePipeline()
    
    try:
        results = pipeline.process_text("", style="realistic")
        print(f"Empty text processed: {len(results)} chunks")
    except Exception as e:
        print(f"Error with empty text: {e}")
    
    # Test with very long text
    long_text = "This is a very long text. " * 1000
    
    try:
        results = pipeline.process_text(long_text, style="realistic")
        print(f"Long text processed: {len(results)} chunks")
    except Exception as e:
        print(f"Error with long text: {e}")

if __name__ == "__main__":
    print("Text-to-Image Pipeline Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_advanced_configuration()
    example_component_usage()
    example_batch_processing()
    example_variations()
    example_error_handling()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
