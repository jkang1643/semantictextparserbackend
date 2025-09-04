"""
Unlimited Text-to-Image Pipeline

Modified pipeline that removes token limits from segmentation and handles them
at the request level instead.
"""

import os
import json
import signal
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from text_processor import TextProcessor
from text_segmenter import TextSegmenter
from scene_segmenter_unlimited import UnlimitedSceneSegmenter
from request_manager import RequestManager
from prompt_generator import PromptGenerator
from image_generator import ImageGenerator


class UnlimitedTextToImagePipeline:
    def __init__(self, 
                 segmentation_method: str = "robust_scene",
                 image_service: str = "nano_banana",
                 max_tokens_per_request: int = 512,  # Changed from max_tokens_per_chunk
                 similarity_threshold: float = 0.6,
                 target_scenes: Optional[int] = None,
                 min_sentences: int = 3,
                 timeout_seconds: int = 300,
                 hysteresis: Optional[Tuple[float, float]] = None,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize the unlimited text-to-image pipeline.
        
        Args:
            segmentation_method: "rule_based", "semantic", or "robust_scene"
            image_service: "nano_banana" or "stable_diffusion"
            max_tokens_per_request: Maximum tokens per API request (not segmentation)
            similarity_threshold: Threshold for semantic similarity
            target_scenes: Target number of scenes for robust_scene method
            min_sentences: Minimum sentences per scene for robust_scene method
            timeout_seconds: Maximum time to wait for operations (default: 5 minutes)
            hysteresis: (stay_inside, enter_boundary) thresholds for robust_scene
            weights: Weights for novelty scoring components
        """
        self.segmentation_method = segmentation_method
        self.image_service = image_service
        self.target_scenes = target_scenes
        self.min_sentences = min_sentences
        self.timeout_seconds = timeout_seconds
        self.hysteresis = hysteresis
        self.weights = weights
        self.start_time = None
        self.interrupted = False
        
        # Set up signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Initialize all components with timeout protection
        try:
            print("🔧 Initializing unlimited pipeline components...")
            self.text_processor = TextProcessor()
            print("✅ Text processor initialized")
            
            self.text_segmenter = TextSegmenter(
                max_tokens_per_chunk=10000,  # High limit, not used for segmentation
                similarity_threshold=similarity_threshold
            )
            print("✅ Text segmenter initialized")
            
            if segmentation_method == "robust_scene":
                print("🧠 Loading unlimited scene segmenter (this may take a moment)...")
                self.scene_segmenter = UnlimitedSceneSegmenter()
                print("✅ Unlimited scene segmenter initialized")
            else:
                self.scene_segmenter = None
                
            self.request_manager = RequestManager(max_tokens_per_request=max_tokens_per_request)
            print("✅ Request manager initialized")
            
            self.prompt_generator = PromptGenerator()
            print("✅ Prompt generator initialized")
            
            self.image_generator = ImageGenerator(service=image_service)
            print("✅ Image generator initialized")
            
        except Exception as e:
            print(f"❌ Error initializing pipeline: {e}")
            raise
        
        # Pipeline state
        self.results = []
        self.characters = []
        self.previous_chunks = []
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print("\n⚠️ Interrupt received. Attempting graceful shutdown...")
        self.interrupted = True
        sys.exit(1)
    
    def _check_timeout(self):
        """Check if we've exceeded the timeout limit."""
        if self.start_time and time.time() - self.start_time > self.timeout_seconds:
            raise TimeoutError(f"Pipeline timeout after {self.timeout_seconds} seconds")
    
    def _check_interrupted(self):
        """Check if the process has been interrupted."""
        if self.interrupted:
            raise KeyboardInterrupt("Pipeline interrupted by user")
    
    def process_text(self, text: str, style: str = "realistic", 
                    save_images: bool = False, output_dir: str = "output") -> List[Dict[str, Any]]:
        """
        Process text through the unlimited pipeline.
        
        Args:
            text: Input text to process
            style: Image generation style
            save_images: Whether to save generated images
            output_dir: Directory to save images
            
        Returns:
            List of results with generated content
        """
        self.start_time = time.time()
        self.results = []
        
        try:
            print(f"🚀 Starting unlimited text processing...")
            print(f"📊 Text length: {len(text)} characters")
            
            # Step 1: Preprocess text
            self._check_interrupted()
            self._check_timeout()
            
            print("📝 Preprocessing text...")
            clean_text = self.text_processor.preprocess_text(text)
            print(f"✅ Text preprocessed: {len(clean_text)} characters")
            
            # Step 2: Segment text (NO TOKEN LIMITS)
            self._check_interrupted()
            self._check_timeout()
            
            if self.segmentation_method == "robust_scene":
                # Use unlimited scene segmentation
                if not self.scene_segmenter:
                    raise ValueError("Scene segmenter not initialized for robust_scene method")
                
                # Prepare parameters for segment_scenes
                segment_params = {
                    "text": clean_text,
                    "target_scenes": self.target_scenes,
                    "min_sentences": self.min_sentences
                }
                
                # Add complexity parameters if provided
                if self.hysteresis is not None:
                    segment_params["hysteresis"] = self.hysteresis
                if self.weights is not None:
                    segment_params["weights"] = self.weights
                
                scenes = self.scene_segmenter.segment_scenes(**segment_params)
                print(f"📊 Unlimited scene segmentation: {len(scenes)} scenes")
                
                # Show scene lengths
                for i, scene in enumerate(scenes, 1):
                    tokens = self.request_manager.count_tokens(scene['text'])
                    print(f"   Scene {i}: {tokens} tokens, {len(scene['text'].split())} words")
                
            else:
                # Use traditional segmentation methods
                chunks = self.text_segmenter.segment_text(clean_text, self.segmentation_method)
                
                # Analyze chunks
                chunk_analysis = self.text_segmenter.analyze_chunks(chunks)
                print(f"📊 Segmentation analysis: {chunk_analysis['total_chunks']} chunks, "
                      f"avg {chunk_analysis['avg_chunk_tokens']:.1f} tokens per chunk")
                
                # Convert chunks to scene format
                scenes = []
                for i, chunk in enumerate(chunks):
                    scenes.append({
                        'text': chunk,
                        'sent_indices': (0, 0),
                        'summary': {},
                        'prompt': chunk,
                        'character': None
                    })
            
            # Step 3: Split scenes for requests (handle token limits here)
            self._check_interrupted()
            self._check_timeout()
            
            print("🔄 Splitting scenes for requests...")
            request_scenes = self.request_manager.process_scenes_for_requests(scenes)
            
            # Show request summary
            summary = self.request_manager.get_request_summary(request_scenes)
            print(f"📊 Request summary: {summary['total_requests']} requests from {summary['original_scenes']} scenes")
            if summary['chunked_scenes'] > 0:
                print(f"   {summary['chunked_scenes']} scenes were split into multiple requests")
            
            # Step 4: Process each request
            self.results = []
            scene_data = None
            
            if self.segmentation_method == "robust_scene":
                # Store scene data for processing
                scene_data = scenes
            
            total_requests = len(request_scenes)
            for i, request_scene in enumerate(request_scenes, 1):
                self._check_interrupted()
                self._check_timeout()
                
                print(f"🎨 Processing request {i}/{total_requests}...")
                
                # Generate prompt
                if self.segmentation_method == "robust_scene":
                    prompt = request_scene.get('prompt', request_scene['text'])
                else:
                    prompt = self.prompt_generator.generate_prompt(
                        request_scene['text'], 
                        style=style,
                        characters=self.characters,
                        previous_chunks=self.previous_chunks
                    )
                
                # Generate image
                try:
                    image_data = self.image_generator.generate_image(prompt)
                    
                    result = {
                        'request_index': i,
                        'scene_index': request_scene.get('chunk_index', 0) if request_scene.get('is_chunk') else i,
                        'is_chunk': request_scene.get('is_chunk', False),
                        'text': request_scene['text'],
                        'prompt': prompt,
                        'image_data': image_data,
                        'tokens': self.request_manager.count_tokens(request_scene['text']),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Add scene metadata if available
                    if 'summary' in request_scene:
                        result['summary'] = request_scene['summary']
                    if 'character' in request_scene:
                        result['character'] = request_scene['character']
                    
                    self.results.append(result)
                    
                    # Update tracking
                    if 'character' in request_scene and request_scene['character']:
                        if request_scene['character'] not in self.characters:
                            self.characters.append(request_scene['character'])
                    
                    self.previous_chunks.append(request_scene['text'])
                    
                    print(f"✅ Request {i} completed: {len(request_scene['text'])} chars, {result['tokens']} tokens")
                    
                except Exception as e:
                    print(f"❌ Error processing request {i}: {e}")
                    # Continue with next request
                    continue
            
            print(f"🎉 Pipeline completed! Generated {len(self.results)} results")
            
            # Save results if requested
            if save_images:
                self._save_results(output_dir)
            
            return self.results
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            raise
    
    def _save_results(self, output_dir: str):
        """Save results to files."""
        if not self.results:
            print("⚠️ No results to save")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save images
        for i, result in enumerate(self.results):
            if 'image_data' in result and result['image_data']:
                filename = f"image_{i+1:03d}.png"
                filepath = os.path.join(output_dir, filename)
                
                try:
                    with open(filepath, 'wb') as f:
                        f.write(result['image_data'])
                    print(f"💾 Saved image: {filepath}")
                except Exception as e:
                    print(f"❌ Error saving image {filename}: {e}")
        
        # Save metadata
        metadata_file = os.path.join(output_dir, "results_metadata.json")
        try:
            # Remove image_data from metadata (too large for JSON)
            metadata = []
            for result in self.results:
                metadata_result = {k: v for k, v in result.items() if k != 'image_data'}
                metadata.append(metadata_result)
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved metadata: {metadata_file}")
        except Exception as e:
            print(f"❌ Error saving metadata: {e}")


def main():
    """Example usage of the unlimited pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unlimited Text-to-Image Pipeline")
    parser.add_argument("--input", help="Input text")
    parser.add_argument("--file", help="Input file")
    parser.add_argument("--demo", action="store_true", help="Use demo text")
    parser.add_argument("--segmentation", choices=["rule_based", "semantic", "robust_scene"], 
                       default="robust_scene", help="Segmentation method")
    parser.add_argument("--service", choices=["nano_banana", "stable_diffusion"], 
                       default="nano_banana", help="Image generation service")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per request")
    parser.add_argument("--target-scenes", type=int, help="Target number of scenes")
    parser.add_argument("--complexity", type=int, default=5, help="Complexity level (0-10)")
    parser.add_argument("--save", action="store_true", help="Save generated images")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    # Get input text
    if args.demo:
        text = """John entered the forest. The trees loomed overhead.
        The forest was alive with sounds. Strange markings appeared on the trees.
        Suddenly, the temperature dropped. Ahead, he heard a sound.
        A woman emerged, dressed in white. Her eyes glowed faintly.
        She spoke softly. Then she added another line.
        John felt a magical pull toward her."""
        print("📖 Using demo text...")
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"📄 Loading text from file: {args.file}")
    else:
        text = args.input
        print("📝 Using provided input text...")
    
    if not text:
        print("❌ No input text provided")
        return
    
    # Get complexity parameters
    from main import get_complexity_parameters
    complexity_params = get_complexity_parameters(args.complexity)
    
    # Initialize pipeline
    print(f"🔧 Initializing unlimited pipeline...")
    pipeline = UnlimitedTextToImagePipeline(
        segmentation_method=args.segmentation,
        image_service=args.service,
        max_tokens_per_request=args.max_tokens,
        target_scenes=args.target_scenes,
        **complexity_params
    )
    
    # Process text
    try:
        results = pipeline.process_text(
            text=text,
            save_images=args.save,
            output_dir=args.output_dir
        )
        
        print(f"\n🎉 Success! Generated {len(results)} results")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
