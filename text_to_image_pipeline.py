import os
import json
import signal
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from text_processor import TextProcessor
from text_segmenter import TextSegmenter
from scene_segmenter import SceneSegmenter
from scene_segmenter_unlimited import UnlimitedSceneSegmenter
from request_manager import RequestManager
from prompt_generator import PromptGenerator
from image_generator import ImageGenerator

class TextToImagePipeline:
    def __init__(self, 
                 segmentation_method: str = "robust_scene",
                 image_service: str = "nano_banana",
                 max_tokens_per_chunk: int = 512,
                 similarity_threshold: float = 0.6,
                 target_scenes: Optional[int] = None,
                 min_sentences: int = 3,
                 timeout_seconds: int = 300,
                 hysteresis: Optional[Tuple[float, float]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 use_unlimited_segmentation: bool = True):
        """
        Initialize the complete text-to-image pipeline.
        
        Args:
            segmentation_method: "rule_based", "semantic", or "robust_scene"
            image_service: "nano_banana" or "stable_diffusion"
            max_tokens_per_chunk: Maximum tokens per chunk (used for request splitting)
            similarity_threshold: Threshold for semantic similarity
            target_scenes: Target number of scenes for robust_scene method
            min_sentences: Minimum sentences per scene for robust_scene method
            timeout_seconds: Maximum time to wait for operations (default: 5 minutes)
            hysteresis: (stay_inside, enter_boundary) thresholds for robust_scene
            weights: Weights for novelty scoring components
            use_unlimited_segmentation: Use unlimited segmentation (no token limits)
        """
        self.segmentation_method = segmentation_method
        self.image_service = image_service
        self.target_scenes = target_scenes
        self.min_sentences = min_sentences
        self.timeout_seconds = timeout_seconds
        self.hysteresis = hysteresis
        self.weights = weights
        self.use_unlimited_segmentation = use_unlimited_segmentation
        self.start_time = None
        self.interrupted = False
        
        # Set up signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Initialize all components with timeout protection
        try:
            print("🔧 Initializing pipeline components...")
            self.text_processor = TextProcessor()
            print("✅ Text processor initialized")
            
            self.text_segmenter = TextSegmenter(
                max_tokens_per_chunk=max_tokens_per_chunk,
                similarity_threshold=similarity_threshold
            )
            print("✅ Text segmenter initialized")
            
            if segmentation_method == "robust_scene":
                if use_unlimited_segmentation:
                    print("🧠 Loading unlimited scene segmenter (this may take a moment)...")
                    self.scene_segmenter = UnlimitedSceneSegmenter()
                    print("✅ Unlimited scene segmenter initialized")
                else:
                    print("🧠 Loading scene segmenter (this may take a moment)...")
                    self.scene_segmenter = SceneSegmenter()
                    print("✅ Scene segmenter initialized")
            else:
                self.scene_segmenter = None
                
            self.request_manager = RequestManager(max_tokens_per_request=max_tokens_per_chunk)
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
        Complete pipeline: text → chunks → prompts → images.
        
        Args:
            text: Raw input text
            style: Visual style for image generation
            save_images: Whether to save images locally
            output_dir: Directory to save images
            
        Returns:
            List of results with chunks, prompts, and images
        """
        self.start_time = time.time()
        print("🚀 Starting text-to-image pipeline...")
        print(f"⏱️ Timeout set to {self.timeout_seconds} seconds")
        
        try:
            # Step 1: Preprocess text
            print("📝 Preprocessing text...")
            self._check_interrupted()
            self._check_timeout()
            
            clean_text = self.text_processor.preprocess_text(text)
            
            # Extract entities for character tracking
            print("👥 Extracting characters...")
            self._check_interrupted()
            self._check_timeout()
            
            self.characters = self.text_processor.extract_entities(clean_text)
            print(f"👥 Found characters: {self.characters}")
            
            # Step 2: Segment text into chunks
            print(f"✂️ Segmenting text using {self.segmentation_method} method...")
            self._check_interrupted()
            self._check_timeout()
            
            if self.segmentation_method == "robust_scene":
                # Use robust scene segmentation
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
                print(f"📊 Robust scene segmentation: {len(scenes)} scenes")
                
                # Show scene lengths
                for i, scene in enumerate(scenes, 1):
                    tokens = self.request_manager.count_tokens(scene['text'])
                    print(f"   Scene {i}: {tokens} tokens, {len(scene['text'].split())} words")
                
                # Split scenes for requests if using unlimited segmentation
                if self.use_unlimited_segmentation:
                    print("🔄 Splitting scenes for requests...")
                    request_scenes = self.request_manager.process_scenes_for_requests(scenes)
                    
                    # Show request summary
                    summary = self.request_manager.get_request_summary(request_scenes)
                    print(f"📊 Request summary: {summary['total_requests']} requests from {summary['original_scenes']} scenes")
                    if summary['chunked_scenes'] > 0:
                        print(f"   {summary['chunked_scenes']} scenes were split into multiple requests")
                    
                    chunks = [scene['text'] for scene in request_scenes]
                    scene_data = request_scenes
                else:
                    chunks = [scene['text'] for scene in scenes]
                    scene_data = scenes
            else:
                # Use traditional segmentation methods
                chunks = self.text_segmenter.segment_text(clean_text, self.segmentation_method)
                
                # Analyze chunks
                chunk_analysis = self.text_segmenter.analyze_chunks(chunks)
                print(f"📊 Segmentation analysis: {chunk_analysis['total_chunks']} chunks, "
                      f"avg {chunk_analysis['avg_chunk_tokens']:.1f} tokens per chunk")
            
            # Step 3: Process each chunk
            self.results = []
            
            total_chunks = len(chunks)
            for i, chunk in enumerate(chunks):
                # Check for interruption and timeout before each chunk
                self._check_interrupted()
                self._check_timeout()
                
                # Show progress
                progress = (i / total_chunks) * 100
                print(f"🎨 Processing chunk {i+1}/{total_chunks} ({progress:.1f}%)...")
                
                result = self._process_chunk(chunk, i, style, save_images, output_dir, scene_data)
                self.results.append(result)
                
                # Update context for next iteration
                self.previous_chunks.append(chunk)
                if len(self.previous_chunks) > 3:  # Keep last 3 chunks for context
                    self.previous_chunks.pop(0)
                
                # Brief pause to prevent overwhelming the system
                time.sleep(0.1)
            
            elapsed_time = time.time() - self.start_time
            print(f"✅ Pipeline completed successfully! (took {elapsed_time:.1f} seconds)")
            return self.results
            
        except (TimeoutError, KeyboardInterrupt) as e:
            print(f"⚠️ Pipeline interrupted: {e}")
            if self.results:
                print(f"📊 Partial results: {len(self.results)} chunks processed")
            raise
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            if self.results:
                print(f"📊 Partial results: {len(self.results)} chunks processed")
            raise
    
    def _process_chunk(self, chunk: str, chunk_index: int, style: str, 
                      save_images: bool, output_dir: str, scene_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Process a single text chunk through the pipeline.
        
        Args:
            chunk: Text chunk to process
            chunk_index: Index of the chunk
            style: Visual style
            save_images: Whether to save images
            output_dir: Output directory
            scene_data: Scene data from robust scene segmentation (optional)
            
        Returns:
            Dictionary with chunk, prompt, and image data
        """
        try:
            # Check for interruption and timeout
            self._check_interrupted()
            self._check_timeout()
            
            # Generate scene prompt
            print(f"  📝 Generating prompt for chunk {chunk_index + 1}...")
            if scene_data and chunk_index < len(scene_data):
                # Use robust scene data for prompt generation
                scene = scene_data[chunk_index]
                scene_prompt = scene.get('prompt', chunk)  # Use pre-generated prompt
            else:
                # Use traditional prompt generation
                scene_prompt = self.prompt_generator.enhance_prompt_with_context(
                    chunk, 
                    previous_chunks=self.previous_chunks,
                    characters=self.characters
                )
            
            # Check again before image generation
            self._check_interrupted()
            self._check_timeout()
            
            # Generate image
            print(f"  🎨 Generating image for chunk {chunk_index + 1}...")
            image_result = self.image_generator.generate_image(scene_prompt, style=style)
            
            # Save image locally if requested
            if save_images and image_result.get("success", False):
                print(f"  💾 Saving image for chunk {chunk_index + 1}...")
                os.makedirs(output_dir, exist_ok=True)
                filename = f"{output_dir}/chunk_{chunk_index:03d}.png"
                self.image_generator.save_image_locally(image_result["image_url"], filename)
                image_result["local_path"] = filename
            
            return {
                "chunk_index": chunk_index,
                "chunk_text": chunk,
                "scene_prompt": scene_prompt,
                "image_result": image_result,
                "characters": self.characters,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"  ❌ Error processing chunk {chunk_index + 1}: {e}")
            # Return a partial result with error information
            return {
                "chunk_index": chunk_index,
                "chunk_text": chunk,
                "scene_prompt": f"Error generating prompt: {e}",
                "image_result": {"success": False, "error": str(e)},
                "characters": self.characters,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def generate_variations(self, chunk_index: int, num_variations: int = 3) -> List[Dict[str, Any]]:
        """
        Generate multiple variations for a specific chunk.
        
        Args:
            chunk_index: Index of the chunk to generate variations for
            num_variations: Number of variations to generate
            
        Returns:
            List of variation results
        """
        if chunk_index >= len(self.results):
            raise ValueError(f"Chunk index {chunk_index} out of range")
        
        chunk_data = self.results[chunk_index]
        chunk_text = chunk_data["chunk_text"]
        
        # Generate multiple prompt variations
        prompt_variations = self.prompt_generator.generate_multiple_variations(
            chunk_text, num_variations
        )
        
        variations = []
        for i, prompt in enumerate(prompt_variations):
            image_result = self.image_generator.generate_image(prompt)
            variations.append({
                "variation_index": i,
                "original_chunk_index": chunk_index,
                "scene_prompt": prompt,
                "image_result": image_result,
                "timestamp": datetime.now().isoformat()
            })
        
        return variations
    
    def save_results(self, filename: str = "pipeline_results.json"):
        """
        Save pipeline results to JSON file.
        
        Args:
            filename: Output filename
        """
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_result = result.copy()
            # Remove any non-serializable objects
            if "image_result" in serializable_result:
                serializable_result["image_result"] = {
                    k: v for k, v in serializable_result["image_result"].items()
                    if isinstance(v, (str, int, float, bool, dict, list))
                }
            serializable_results.append(serializable_result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to {filename}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the pipeline execution.
        
        Returns:
            Dictionary with pipeline statistics
        """
        if not self.results:
            return {"error": "No results available"}
        
        total_chunks = len(self.results)
        successful_images = sum(1 for r in self.results if r["image_result"].get("success", False))
        
        avg_prompt_length = sum(len(r["scene_prompt"]) for r in self.results) / total_chunks
        avg_chunk_length = sum(len(r["chunk_text"]) for r in self.results) / total_chunks
        
        return {
            "total_chunks": total_chunks,
            "successful_images": successful_images,
            "success_rate": successful_images / total_chunks,
            "avg_prompt_length": avg_prompt_length,
            "avg_chunk_length": avg_chunk_length,
            "characters_found": len(self.characters),
            "segmentation_method": self.segmentation_method,
            "image_service": self.image_service
        }
