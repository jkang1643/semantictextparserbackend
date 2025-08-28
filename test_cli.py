#!/usr/bin/env python3
"""
Lightweight Command Line Interface for Text-to-Image Pipeline Testing

This provides a simple CLI for testing the pipeline without heavy dependencies.
"""

import argparse
import sys
from text_to_image_pipeline_lite import TextToImagePipelineLite

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Test the lightweight text-to-image pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_cli.py --demo
  python test_cli.py --input "A knight rode through the forest..."
  python test_cli.py --file story.txt --style fantasy
  python test_cli.py --input "Detective story..." --segmentation semantic --style cinematic
        """
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Run with built-in demo story"
    )
    
    parser.add_argument(
        "--input", 
        type=str,
        help="Input text to process"
    )
    
    parser.add_argument(
        "--file", 
        type=str,
        help="Input file to process"
    )
    
    parser.add_argument(
        "--style", 
        type=str,
        choices=["realistic", "artistic", "cinematic", "fantasy"],
        default="realistic",
        help="Visual style for image generation"
    )
    
    parser.add_argument(
        "--segmentation", 
        type=str,
        choices=["rule_based", "semantic"],
        default="rule_based",
        help="Text segmentation method"
    )
    
    parser.add_argument(
        "--max-tokens", 
        type=int,
        default=512,
        help="Maximum tokens per chunk"
    )
    
    parser.add_argument(
        "--similarity-threshold", 
        type=float,
        default=0.6,
        help="Similarity threshold for semantic segmentation"
    )
    
    parser.add_argument(
        "--output", 
        type=str,
        default="pipeline_results.json",
        help="Output JSON file"
    )
    
    parser.add_argument(
        "--save-images", 
        action="store_true",
        help="Save images locally (mock images only)"
    )
    
    args = parser.parse_args()
    
    # Get input text
    if args.demo:
        text = """
        Detective Sarah Chen walked through the rain-soaked streets of Chinatown, 
        her trench coat billowing in the wind. The neon signs reflected in the 
        puddles, creating a kaleidoscope of colors. She was following a lead on 
        the mysterious disappearance of Professor James Wilson, a renowned 
        archaeologist who had been studying ancient artifacts.
        
        As she turned the corner onto Grant Avenue, she noticed a figure in a 
        dark hooded jacket ducking into an alley. Sarah quickened her pace, 
        her hand instinctively reaching for her service weapon. The alley was 
        narrow and dimly lit, with steam rising from manhole covers.
        """
        print("🎯 Running demo story...")
        
    elif args.input:
        text = args.input
        print(f"🎯 Processing input text ({len(text)} characters)...")
        
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"🎯 Processing file: {args.file} ({len(text)} characters)...")
        except FileNotFoundError:
            print(f"❌ Error: File '{args.file}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)
    else:
        print("❌ Error: Must specify --demo, --input, or --file")
        parser.print_help()
        sys.exit(1)
    
    # Initialize pipeline
    print(f"🚀 Initializing pipeline...")
    print(f"   Style: {args.style}")
    print(f"   Segmentation: {args.segmentation}")
    print(f"   Max tokens: {args.max_tokens}")
    
    pipeline = TextToImagePipelineLite(
        segmentation_method=args.segmentation,
        image_service="dalle",
        max_tokens_per_chunk=args.max_tokens,
        similarity_threshold=args.similarity_threshold
    )
    
    # Process text
    try:
        results = pipeline.process_text(
            text, 
            style=args.style,
            save_images=args.save_images
        )
        
        # Save results
        pipeline.save_results(args.output)
        
        # Print statistics
        stats = pipeline.get_pipeline_stats()
        print("\n📊 Pipeline Statistics:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Characters found: {stats['characters_found']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Average prompt length: {stats['avg_prompt_length']:.1f} characters")
        print(f"   Average chunk length: {stats['avg_chunk_length']:.1f} characters")
        
        # Show sample results
        if results:
            print(f"\n🎯 Sample Results:")
            for i, result in enumerate(results[:3]):
                print(f"\nChunk {i+1}:")
                print(f"  Text: {result['chunk_text'][:100]}...")
                print(f"  Characters: {result['characters']}")
                print(f"  Prompt: {result['scene_prompt'][:100]}...")
                print(f"  Image: {result['image_result']['image_url']}")
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"💾 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
