#!/usr/bin/env python3
"""
Text-to-Image Pipeline - Main Script

This script demonstrates the complete text-to-image pipeline that:
1. Takes raw text input (story, transcript, article, etc.)
2. Preprocesses and segments the text into chunks
3. Generates visual scene prompts for each chunk
4. Creates images from the prompts using AI image generation services

Usage:
    python main.py --input "your text here"
    python main.py --file input.txt
    python main.py --demo
"""

import argparse
import sys
import os
from text_to_image_pipeline_lite import TextToImagePipelineLite

def load_text_from_file(filename: str) -> str:
    """Load text from a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def get_demo_text() -> str:
    """Get demo text for testing the pipeline."""
    return """
    John walked into the ancient forest, his footsteps muffled by the thick carpet of fallen leaves. 
    The towering trees created a natural cathedral, their branches intertwining overhead to form a 
    canopy that filtered the afternoon sunlight into golden shafts. The air was thick with the scent 
    of pine and earth, and somewhere in the distance, a bird sang a haunting melody.
    
    As he ventured deeper, the path became less defined, and the forest seemed to close in around him. 
    He noticed strange markings on some of the trees - symbols that looked both ancient and otherworldly. 
    The temperature dropped noticeably, and he pulled his jacket tighter around his shoulders.
    
    Suddenly, he heard a rustling sound ahead. A figure emerged from behind a massive oak tree - a woman 
    with silver hair that seemed to glow in the dappled light. She wore a flowing dress that seemed to 
    be made of mist and moonlight. Her eyes held an ancient wisdom that made John feel like a child in 
    comparison.
    
    "You've been expected," she said, her voice like wind through leaves. "The forest has been waiting 
    for someone like you." She extended her hand, and John felt an irresistible pull toward her, as if 
    the very air around them was charged with magic.
    """

def main():
    parser = argparse.ArgumentParser(
        description="Text-to-Image Pipeline (Lite) - Convert text into visual scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input "A man walks into a forest..."
  python main.py --file story.txt --style cinematic
  python main.py --demo --save-images
  python main.py --input "Your text" --segmentation rule_based --service stable_diffusion
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input text to process"
    )
    
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Input file containing text to process"
    )
    
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run with demo text"
    )
    
    parser.add_argument(
        "--style", "-s",
        type=str,
        default="realistic",
        choices=["realistic", "artistic", "cinematic", "fantasy"],
        help="Visual style for image generation (default: realistic)"
    )
    
    parser.add_argument(
        "--segmentation", "-seg",
        type=str,
        default="semantic",
        choices=["rule_based", "semantic"],
        help="Text segmentation method (default: semantic)"
    )
    
    parser.add_argument(
        "--service", "-svc",
        type=str,
        default="nano_banana",
        choices=["nano_banana", "stable_diffusion"],
        help="Image generation service (default: nano_banana)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per chunk (default: 512)"
    )
    
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.6,
        help="Similarity threshold for semantic segmentation (default: 0.6)"
    )
    
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save generated images locally"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for saved images (default: output)"
    )
    
    parser.add_argument(
        "--results-file",
        type=str,
        default="pipeline_results.json",
        help="Filename for saving results (default: pipeline_results.json)"
    )
    
    parser.add_argument(
        "--variations",
        type=int,
        default=0,
        help="Generate multiple variations for each chunk"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not any([args.input, args.file, args.demo]):
        print("Error: Must provide input text, file, or use --demo")
        parser.print_help()
        sys.exit(1)
    
    # Get input text
    if args.demo:
        text = get_demo_text()
        print("📖 Using demo text...")
    elif args.file:
        text = load_text_from_file(args.file)
        print(f"📄 Loading text from file: {args.file}")
    else:
        text = args.input
        print("📝 Using provided input text...")
    
    print(f"📊 Text length: {len(text)} characters")
    
    # Initialize pipeline
    print(f"🔧 Initializing pipeline with {args.segmentation} segmentation and {args.service} service...")
    pipeline = TextToImagePipelineLite(
        segmentation_method=args.segmentation,
        image_service=args.service,
        max_tokens_per_chunk=args.max_tokens,
        similarity_threshold=args.similarity_threshold
    )
    
    # Process text
    try:
        results = pipeline.process_text(
            text=text,
            style=args.style,
            save_images=args.save_images,
            output_dir=args.output_dir
        )
        
        # Generate variations if requested
        if args.variations > 0:
            print(f"🎨 Generating {args.variations} variations for each chunk...")
            for i in range(len(results)):
                variations = pipeline.generate_variations(i, args.variations)
                results[i]["variations"] = variations
        
        # Save results
        pipeline.save_results(args.results_file)
        
        # Print statistics
        stats = pipeline.get_pipeline_stats()
        print("\n📈 Pipeline Statistics:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Successful images: {stats['successful_images']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Characters found: {stats['characters_found']}")
        print(f"   Average prompt length: {stats['avg_prompt_length']:.1f} characters")
        print(f"   Average chunk length: {stats['avg_chunk_length']:.1f} characters")
        
        # Print sample results
        print("\n🎯 Sample Results:")
        for i, result in enumerate(results[:3]):  # Show first 3 results
            print(f"\nChunk {i+1}:")
            print(f"  Text: {result['chunk_text'][:100]}...")
            print(f"  Prompt: {result['scene_prompt'][:100]}...")
            print(f"  Image: {result['image_result'].get('image_url', 'No image generated')}")
        
        if len(results) > 3:
            print(f"\n... and {len(results) - 3} more chunks")
        
        print(f"\n✅ Pipeline completed! Results saved to {args.results_file}")
        if args.save_images:
            print(f"📁 Images saved to {args.output_dir}/")
        
    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
