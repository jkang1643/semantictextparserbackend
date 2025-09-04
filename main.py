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
from text_to_image_pipeline import TextToImagePipeline

def get_complexity_parameters(complexity: int) -> dict:
    """
    Convert complexity level (0-10) to scene segmentation parameters.
    
    Args:
        complexity: Integer from 0-10 where:
            0-2: Very simple (2-3 scenes) - Image generation focused
            3-4: Simple (3-4 scenes) - Fewer scenes
            5-6: Balanced (4-6 scenes) - Default balanced approach
            7-8: Detailed (6-8 scenes) - More granular
            9-10: Very detailed (8-10+ scenes) - Maximum granularity
    
    Returns:
        Dictionary with hysteresis, min_sentences, and weights parameters
    """
    if complexity <= 2:
        # Very simple - 2-3 scenes, perfect for image generation
        return {
            "hysteresis": (0.6, 0.8),
            "min_sentences": 3,
            "weights": {"semantic": 0.3, "entity": 0.4, "visual": 0.2, "cue": 0.1}
        }
    elif complexity <= 4:
        # Simple - 3-4 scenes
        return {
            "hysteresis": (0.5, 0.7),
            "min_sentences": 3,
            "weights": {"semantic": 0.4, "entity": 0.35, "visual": 0.2, "cue": 0.05}
        }
    elif complexity <= 6:
        # Balanced - 4-6 scenes (default)
        return {
            "hysteresis": (0.3, 0.6),
            "min_sentences": 2,
            "weights": {"semantic": 0.55, "entity": 0.25, "visual": 0.15, "cue": 0.05}
        }
    elif complexity <= 8:
        # Detailed - 6-8 scenes
        return {
            "hysteresis": (0.2, 0.4),
            "min_sentences": 2,
            "weights": {"semantic": 0.6, "entity": 0.2, "visual": 0.15, "cue": 0.05}
        }
    else:
        # Very detailed - 8-10+ scenes
        return {
            "hysteresis": (0.05, 0.15),
            "min_sentences": 1,
            "weights": {"semantic": 0.65, "entity": 0.15, "visual": 0.15, "cue": 0.05}
        }

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
  python main.py --input "Your text" --complexity 3 --style fantasy
  python main.py --file story.txt --complexity 8 --segmentation robust_scene
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
        default="robust_scene",
        choices=["rule_based", "semantic", "robust_scene"],
        help="Text segmentation method (default: robust_scene)"
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
        "--complexity",
        type=int,
        default=5,
        choices=range(0, 11),
        help="Scene complexity level (0-10): 0=very simple (2-3 scenes), 5=balanced (4-6 scenes), 10=very detailed (8-10+ scenes) (default: 5)"
    )
    
    parser.add_argument(
        "--target-scenes",
        type=int,
        help="Target number of scenes for robust_scene segmentation (optional, overrides complexity)"
    )
    
    parser.add_argument(
        "--min-sentences",
        type=int,
        default=3,
        help="Minimum sentences per scene for robust_scene segmentation (default: 3)"
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
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Maximum time to wait for pipeline completion in seconds (default: 300)"
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
    
    # Get complexity parameters if using robust_scene segmentation
    complexity_params = {}
    if args.segmentation == "robust_scene" and not args.target_scenes:
        complexity_params = get_complexity_parameters(args.complexity)
        print(f"🎯 Using complexity level {args.complexity}: {complexity_params['hysteresis']} hysteresis, {complexity_params['min_sentences']} min sentences")
        # Use complexity min_sentences instead of args.min_sentences
        min_sentences = complexity_params.pop('min_sentences')
    else:
        min_sentences = args.min_sentences
    
    # Initialize pipeline
    print(f"🔧 Initializing pipeline with {args.segmentation} segmentation and {args.service} service...")
    pipeline = TextToImagePipeline(
        segmentation_method=args.segmentation,
        image_service=args.service,
        max_tokens_per_chunk=args.max_tokens,
        similarity_threshold=args.similarity_threshold,
        target_scenes=args.target_scenes,
        min_sentences=min_sentences,
        timeout_seconds=args.timeout,
        use_unlimited_segmentation=True,  # Enable unlimited segmentation by default
        **complexity_params  # Pass complexity parameters
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
                try:
                    variations = pipeline.generate_variations(i, args.variations)
                    results[i]["variations"] = variations
                except Exception as e:
                    print(f"⚠️ Error generating variations for chunk {i+1}: {e}")
                    results[i]["variations"] = []
        
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
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user (Ctrl+C)")
        if pipeline.results:
            print(f"📊 Partial results: {len(pipeline.results)} chunks processed")
            pipeline.save_results(f"interrupted_{args.results_file}")
            print(f"💾 Partial results saved to interrupted_{args.results_file}")
        sys.exit(0)
    except TimeoutError as e:
        print(f"\n⏰ Pipeline timeout: {e}")
        if pipeline.results:
            print(f"📊 Partial results: {len(pipeline.results)} chunks processed")
            pipeline.save_results(f"timeout_{args.results_file}")
            print(f"💾 Partial results saved to timeout_{args.results_file}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        if pipeline.results:
            print(f"📊 Partial results: {len(pipeline.results)} chunks processed")
            pipeline.save_results(f"error_{args.results_file}")
            print(f"💾 Partial results saved to error_{args.results_file}")
        sys.exit(1)

if __name__ == "__main__":
    main()
