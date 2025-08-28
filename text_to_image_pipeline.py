import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from text_processor import TextProcessor
from text_segmenter import TextSegmenter
from prompt_generator import PromptGenerator
from image_generator import ImageGenerator

class TextToImagePipeline:
    def __init__(self, 
                 segmentation_method: str = "semantic",
                 image_service: str = "dalle",
                 max_tokens_per_chunk: int = 512,
                 similarity_threshold: float = 0.6):
        """
        Initialize the complete text-to-image pipeline.
        
        Args:
            segmentation_method: "rule_based" or "semantic"
            image_service: "dalle" or "stable_diffusion"
            max_tokens_per_chunk: Maximum tokens per chunk
            similarity_threshold: Threshold for semantic similarity
        """
        self.segmentation_method = segmentation_method
        self.image_service = image_service
        
        # Initialize all components
        self.text_processor = TextProcessor()
        self.text_segmenter = TextSegmenter(
            max_tokens_per_chunk=max_tokens_per_chunk,
            similarity_threshold=similarity_threshold
        )
        self.prompt_generator = PromptGenerator()
        self.image_generator = ImageGenerator(service=image_service)
        
        # Pipeline state
        self.results = []
        self.characters = []
        self.previous_chunks = []
    
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
        print("🚀 Starting text-to-image pipeline...")
        
        # Step 1: Preprocess text
        print("📝 Preprocessing text...")
        clean_text = self.text_processor.preprocess_text(text)
        
        # Extract entities for character tracking
        self.characters = self.text_processor.extract_entities(clean_text)
        print(f"👥 Found characters: {self.characters}")
        
        # Step 2: Segment text into chunks
        print(f"✂️ Segmenting text using {self.segmentation_method} method...")
        chunks = self.text_segmenter.segment_text(clean_text, self.segmentation_method)
        
        # Analyze chunks
        chunk_analysis = self.text_segmenter.analyze_chunks(chunks)
        print(f"📊 Segmentation analysis: {chunk_analysis['total_chunks']} chunks, "
              f"avg {chunk_analysis['avg_chunk_tokens']:.1f} tokens per chunk")
        
        # Step 3: Process each chunk
        self.results = []
        for i, chunk in enumerate(chunks):
            print(f"🎨 Processing chunk {i+1}/{len(chunks)}...")
            
            result = self._process_chunk(chunk, i, style, save_images, output_dir)
            self.results.append(result)
            
            # Update context for next iteration
            self.previous_chunks.append(chunk)
            if len(self.previous_chunks) > 3:  # Keep last 3 chunks for context
                self.previous_chunks.pop(0)
        
        print("✅ Pipeline completed successfully!")
        return self.results
    
    def _process_chunk(self, chunk: str, chunk_index: int, style: str, 
                      save_images: bool, output_dir: str) -> Dict[str, Any]:
        """
        Process a single text chunk through the pipeline.
        
        Args:
            chunk: Text chunk to process
            chunk_index: Index of the chunk
            style: Visual style
            save_images: Whether to save images
            output_dir: Output directory
            
        Returns:
            Dictionary with chunk, prompt, and image data
        """
        # Generate scene prompt
        scene_prompt = self.prompt_generator.enhance_prompt_with_context(
            chunk, 
            previous_chunks=self.previous_chunks,
            characters=self.characters
        )
        
        # Generate image
        image_result = self.image_generator.generate_image(scene_prompt, style=style)
        
        # Save image locally if requested
        if save_images and image_result.get("success", False):
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
