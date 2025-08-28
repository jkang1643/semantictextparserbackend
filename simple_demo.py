#!/usr/bin/env python3
"""
Simplified Text-to-Image Pipeline Demo

This script demonstrates the pipeline functionality without requiring
heavy dependencies like spaCy, transformers, etc.
"""

import re
import json
from typing import List, Dict, Any
from datetime import datetime

class SimpleTextProcessor:
    """Simplified text processor for demo purposes."""
    
    def __init__(self):
        pass
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove timestamps
        text = re.sub(r'\d{1,2}:\d{2}(:\d{2})?', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Simple sentence tokenization."""
        # Split by common sentence endings
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4
    
    def extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction using regex."""
        # Find capitalized words (potential names)
        entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        return list(set(entities))

class SimpleTextSegmenter:
    """Simplified text segmenter."""
    
    def __init__(self, max_tokens_per_chunk: int = 512):
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.text_processor = SimpleTextProcessor()
    
    def segment_text_rule_based(self, text: str) -> List[str]:
        """Rule-based text segmentation."""
        sentences = self.text_processor.tokenize_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.text_processor.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.max_tokens_per_chunk and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

class SimplePromptGenerator:
    """Simplified prompt generator."""
    
    def __init__(self):
        pass
    
    def generate_scene_prompt(self, text_chunk: str, style: str = "realistic") -> str:
        """Generate a basic scene prompt."""
        # Simple rule-based prompt generation
        prompt = text_chunk.lower()
        
        # Add style-specific enhancements
        if style == "realistic":
            prompt += ", high quality, detailed, realistic photography"
        elif style == "artistic":
            prompt += ", artistic illustration, vibrant colors"
        elif style == "cinematic":
            prompt += ", cinematic lighting, dramatic composition"
        elif style == "fantasy":
            prompt += ", fantasy art style, magical atmosphere"
        
        prompt += ", high resolution, professional photography"
        return prompt
    
    def enhance_prompt_with_context(self, text_chunk: str, previous_chunks: List[str] = None, 
                                  characters: List[str] = None) -> str:
        """Generate prompt with context."""
        prompt = self.generate_scene_prompt(text_chunk, "realistic")
        
        if characters:
            prompt += f", featuring characters: {', '.join(characters)}"
        
        return prompt

class SimpleImageGenerator:
    """Simplified image generator that returns mock results."""
    
    def __init__(self, service: str = "dalle"):
        self.service = service
    
    def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate mock image result."""
        return {
            "success": True,
            "image_url": f"https://via.placeholder.com/512x512/cccccc/666666?text=Mock+{self.service.title()}+Image",
            "service": self.service.title(),
            "prompt": prompt,
            "metadata": {
                "note": "Mock image - API not available",
                "original_prompt": prompt
            }
        }

class SimpleTextToImagePipeline:
    """Simplified text-to-image pipeline for demo."""
    
    def __init__(self, segmentation_method: str = "rule_based", image_service: str = "dalle"):
        self.segmentation_method = segmentation_method
        self.image_service = image_service
        
        # Initialize components
        self.text_processor = SimpleTextProcessor()
        self.text_segmenter = SimpleTextSegmenter()
        self.prompt_generator = SimplePromptGenerator()
        self.image_generator = SimpleImageGenerator(service=image_service)
        
        # Pipeline state
        self.results = []
        self.characters = []
        self.previous_chunks = []
    
    def process_text(self, text: str, style: str = "realistic", 
                    save_images: bool = False, output_dir: str = "output") -> List[Dict[str, Any]]:
        """Complete pipeline: text → chunks → prompts → images."""
        print("🚀 Starting simplified text-to-image pipeline...")
        
        # Step 1: Preprocess text
        print("📝 Preprocessing text...")
        clean_text = self.text_processor.preprocess_text(text)
        
        # Extract entities for character tracking
        self.characters = self.text_processor.extract_entities(clean_text)
        print(f"👥 Found characters: {self.characters}")
        
        # Step 2: Segment text into chunks
        print(f"✂️ Segmenting text using {self.segmentation_method} method...")
        chunks = self.text_segmenter.segment_text_rule_based(clean_text)
        print(f"📊 Segmentation analysis: {len(chunks)} chunks")
        
        # Step 3: Process each chunk
        self.results = []
        for i, chunk in enumerate(chunks):
            print(f"🎨 Processing chunk {i+1}/{len(chunks)}...")
            
            result = self._process_chunk(chunk, i, style, save_images, output_dir)
            self.results.append(result)
            
            # Update context for next iteration
            self.previous_chunks.append(chunk)
            if len(self.previous_chunks) > 3:
                self.previous_chunks.pop(0)
        
        print("✅ Pipeline completed successfully!")
        return self.results
    
    def _process_chunk(self, chunk: str, chunk_index: int, style: str, 
                      save_images: bool, output_dir: str) -> Dict[str, Any]:
        """Process a single text chunk through the pipeline."""
        # Generate scene prompt
        scene_prompt = self.prompt_generator.enhance_prompt_with_context(
            chunk, 
            previous_chunks=self.previous_chunks,
            characters=self.characters
        )
        
        # Generate image
        image_result = self.image_generator.generate_image(scene_prompt, style=style)
        
        return {
            "chunk_index": chunk_index,
            "chunk_text": chunk,
            "scene_prompt": scene_prompt,
            "image_result": image_result,
            "characters": self.characters,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_results(self, filename: str = "pipeline_results.json"):
        """Save pipeline results to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to {filename}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline execution."""
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

def main():
    """Demo the simplified pipeline."""
    print("🎯 Simplified Text-to-Image Pipeline Demo")
    print("=" * 50)
    
    # Sample text
    text = """
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
    
    # Initialize pipeline
    pipeline = SimpleTextToImagePipeline(
        segmentation_method="rule_based",
        image_service="dalle"
    )
    
    # Process text
    results = pipeline.process_text(text, style="fantasy")
    
    # Save results
    pipeline.save_results("demo_results.json")
    
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
    for i, result in enumerate(results[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  Text: {result['chunk_text'][:100]}...")
        print(f"  Prompt: {result['scene_prompt'][:100]}...")
        print(f"  Image: {result['image_result'].get('image_url', 'No image generated')}")
    
    print(f"\n✅ Demo completed! Results saved to demo_results.json")

if __name__ == "__main__":
    main()
