import os
import requests
import base64
from typing import Dict, Any, Optional
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ImageGenerator:
    def __init__(self, service: str = "dalle"):
        """
        Initialize the image generator with specified service.
        
        Args:
            service: Image generation service ("dalle", "stable_diffusion", "custom")
        """
        self.service = service
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            openai.api_key = self.api_key
        
        # Service-specific configurations
        self.service_configs = {
            "dalle": {
                "model": "dall-e-3",
                "size": "1024x1024",
                "quality": "standard",
                "style": "natural"
            },
            "stable_diffusion": {
                "api_url": os.getenv("STABLE_DIFFUSION_API_URL"),
                "api_key": os.getenv("STABLE_DIFFUSION_API_KEY")
            }
        }
    
    def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text prompt for image generation
            **kwargs: Additional parameters for the specific service
            
        Returns:
            Dictionary containing image data and metadata
        """
        if self.service == "dalle":
            return self._generate_dalle_image(prompt, **kwargs)
        elif self.service == "stable_diffusion":
            return self._generate_stable_diffusion_image(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported service: {self.service}")
    
    def _generate_dalle_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate image using DALL-E API.
        
        Args:
            prompt: Text prompt for image generation
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with image data and metadata
        """
        if not self.api_key:
            return self._generate_mock_image(prompt, "DALL-E")
        
        config = self.service_configs["dalle"]
        
        # Override defaults with kwargs
        size = kwargs.get("size", config["size"])
        quality = kwargs.get("quality", config["quality"])
        style = kwargs.get("style", config["style"])
        
        try:
            response = openai.Image.create(
                model=config["model"],
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=1
            )
            
            return {
                "success": True,
                "image_url": response.data[0].url,
                "service": "DALL-E",
                "prompt": prompt,
                "metadata": {
                    "size": size,
                    "quality": quality,
                    "style": style
                }
            }
            
        except Exception as e:
            print(f"Error generating DALL-E image: {e}")
            return self._generate_mock_image(prompt, "DALL-E")
    
    def _generate_stable_diffusion_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate image using Stable Diffusion API.
        
        Args:
            prompt: Text prompt for image generation
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with image data and metadata
        """
        config = self.service_configs["stable_diffusion"]
        
        if not config["api_url"] or not config["api_key"]:
            return self._generate_mock_image(prompt, "Stable Diffusion")
        
        # Default parameters for Stable Diffusion
        params = {
            "prompt": prompt,
            "negative_prompt": kwargs.get("negative_prompt", ""),
            "width": kwargs.get("width", 512),
            "height": kwargs.get("height", 512),
            "steps": kwargs.get("steps", 20),
            "cfg_scale": kwargs.get("cfg_scale", 7.5),
            "seed": kwargs.get("seed", -1)
        }
        
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                config["api_url"],
                json=params,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "image_url": result.get("image_url", ""),
                    "image_data": result.get("image_data", ""),
                    "service": "Stable Diffusion",
                    "prompt": prompt,
                    "metadata": params
                }
            else:
                print(f"Stable Diffusion API error: {response.status_code}")
                return self._generate_mock_image(prompt, "Stable Diffusion")
                
        except Exception as e:
            print(f"Error generating Stable Diffusion image: {e}")
            return self._generate_mock_image(prompt, "Stable Diffusion")
    
    def _generate_mock_image(self, prompt: str, service: str) -> Dict[str, Any]:
        """
        Generate a mock image response when API is not available.
        
        Args:
            prompt: Text prompt for image generation
            service: Service name for metadata
            
        Returns:
            Mock image response
        """
        return {
            "success": False,
            "image_url": f"https://via.placeholder.com/512x512/cccccc/666666?text=Mock+{service}+Image",
            "service": service,
            "prompt": prompt,
            "metadata": {
                "note": "Mock image - API not available",
                "original_prompt": prompt
            }
        }
    
    def generate_image_variations(self, prompt: str, num_variations: int = 3, **kwargs) -> list:
        """
        Generate multiple image variations from the same prompt.
        
        Args:
            prompt: Text prompt for image generation
            num_variations: Number of variations to generate
            **kwargs: Additional parameters
            
        Returns:
            List of image generation results
        """
        variations = []
        
        for i in range(num_variations):
            # Add variation to prompt
            variation_prompt = f"{prompt} (variation {i+1})"
            result = self.generate_image(variation_prompt, **kwargs)
            variations.append(result)
        
        return variations
    
    def save_image_locally(self, image_url: str, filename: str) -> bool:
        """
        Download and save an image from URL to local file.
        
        Args:
            image_url: URL of the image to download
            filename: Local filename to save as
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            return True
            
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
