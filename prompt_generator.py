import os
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PromptGenerator:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize the prompt generator with OpenAI API.
        
        Args:
            model: OpenAI model to use for prompt generation
        """
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            openai.api_key = self.api_key
        else:
            print("Warning: OPENAI_API_KEY not found in environment variables")
    
    def generate_scene_prompt(self, text_chunk: str, style: str = "realistic") -> str:
        """
        Transform a text chunk into a visual scene description for image generation.
        
        Args:
            text_chunk: Input text chunk
            style: Visual style preference ("realistic", "artistic", "cinematic", etc.)
            
        Returns:
            Enhanced scene prompt for image generation
        """
        if not self.api_key:
            # Fallback to rule-based prompt generation
            return self._generate_fallback_prompt(text_chunk, style)
        
        system_prompt = f"""You are an expert at creating visual scene descriptions for AI image generation. 
        Transform the given text into a detailed, visual scene description that would work well with 
        text-to-image models like Google Nano Banana, Stable Diffusion, or Midjourney.
        
        Style: {style}
        
        Guidelines:
        - Focus on visual elements: setting, lighting, colors, composition
        - Include character descriptions if present
        - Add atmospheric details (mood, weather, time of day)
        - Make it cinematic and visually appealing
        - Keep it under 200 words
        - Avoid abstract concepts, focus on concrete visual elements
        """
        
        user_prompt = f"Transform this text into a visual scene description: {text_chunk}"
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating prompt with OpenAI: {e}")
            return self._generate_fallback_prompt(text_chunk, style)
    
    def _generate_fallback_prompt(self, text_chunk: str, style: str) -> str:
        """
        Fallback rule-based prompt generation when OpenAI API is not available.
        
        Args:
            text_chunk: Input text chunk
            style: Visual style preference
            
        Returns:
            Basic scene prompt
        """
        # Simple rule-based transformations
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
        
        # Add general quality improvements
        prompt += ", high resolution, professional photography"
        
        return prompt
    
    def enhance_prompt_with_context(self, text_chunk: str, previous_chunks: List[str] = None, 
                                  characters: List[str] = None) -> str:
        """
        Generate a scene prompt with context from previous chunks and character information.
        
        Args:
            text_chunk: Current text chunk
            previous_chunks: List of previous chunks for context
            characters: List of character names for consistency
            
        Returns:
            Enhanced scene prompt with context
        """
        if not self.api_key:
            return self._generate_fallback_prompt(text_chunk, "realistic")
        
        context = ""
        if previous_chunks:
            context += f"Previous scene context: {' '.join(previous_chunks[-2:])} "
        
        if characters:
            context += f"Characters in the story: {', '.join(characters)} "
        
        system_prompt = """You are an expert at creating visual scene descriptions for AI image generation.
        Create a detailed scene description that maintains visual consistency with previous scenes
        and accurately represents the characters and setting described in the text.
        
        Focus on:
        - Visual continuity with previous scenes
        - Character consistency and appearance
        - Atmospheric details and mood
        - Cinematic composition and lighting
        """
        
        user_prompt = f"{context}Current scene text: {text_chunk}"
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.6
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating contextual prompt: {e}")
            return self._generate_fallback_prompt(text_chunk, "realistic")
    
    def generate_multiple_variations(self, text_chunk: str, num_variations: int = 3) -> List[str]:
        """
        Generate multiple prompt variations for the same text chunk.
        
        Args:
            text_chunk: Input text chunk
            num_variations: Number of variations to generate
            
        Returns:
            List of prompt variations
        """
        variations = []
        
        for i in range(num_variations):
            style = ["realistic", "artistic", "cinematic"][i % 3]
            prompt = self.generate_scene_prompt(text_chunk, style)
            variations.append(prompt)
        
        return variations
