import os
import requests
import base64
from typing import Dict, Any, Optional
import openai
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

class ImageGenerator:
    def __init__(self, service: str = "nano_banana"):
        """
        Initialize the image generator with specified service.
        
        Args:
            service: Image generation service ("nano_banana", "stable_diffusion", "custom")
        """
        self.service = service
        self.nano_banana_api_key = os.getenv("NANO_BANANA_API_KEY")
        self.nano_banana_api_url = os.getenv("NANO_BANANA_API_URL", "https://api.nanobanana.ai/v1/images/generations")
        
        # OpenAI API key for prompt generation (optional)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Create images directory if it doesn't exist
        self.images_dir = "generated_images"
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        
        # Service-specific configurations
        self.service_configs = {
            "nano_banana": {
                "api_url": self.nano_banana_api_url,
                "api_key": self.nano_banana_api_key,
                "model": "imagen-3",
                "size": "1024x1024",
                "quality": "standard",
                "style": "natural"
            },
            "stable_diffusion": {
                "api_url": os.getenv("STABLE_DIFFUSION_API_URL"),
                "api_key": os.getenv("STABLE_DIFFUSION_API_KEY")
            }
        }
    
    def save_base64_image(self, base64_data: str, prompt: str, service: str = "unknown") -> str:
        """
        Save a base64 encoded image to disk.
        
        Args:
            base64_data: Base64 encoded image data (with or without data URL prefix)
            prompt: The prompt used to generate the image
            service: The service that generated the image
            
        Returns:
            Path to the saved image file
        """
        try:
            # Remove data URL prefix if present
            if base64_data.startswith('data:'):
                base64_data = base64_data.split(',')[1]
            
            # Decode base64 data
            image_bytes = base64.b64decode(base64_data)
            
            # Create filename from prompt and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Clean prompt for filename (remove special characters)
            clean_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            clean_prompt = clean_prompt.replace(' ', '_')
            
            filename = f"{service}_{timestamp}_{clean_prompt}.png"
            filepath = os.path.join(self.images_dir, filename)
            
            # Save image to disk
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            print(f"💾 Image saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving image: {e}")
            return ""

    def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text prompt for image generation
            **kwargs: Additional parameters for the specific service
            
        Returns:
            Dictionary containing image data and metadata
        """
        if self.service == "nano_banana":
            return self._generate_nano_banana_image(prompt, **kwargs)
        elif self.service == "stable_diffusion":
            return self._generate_stable_diffusion_image(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported service: {self.service}")
    
    def _generate_nano_banana_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate image using Google Nano Banana API.
        
        Args:
            prompt: Text prompt for image generation
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with image data and metadata
        """
        if not self.nano_banana_api_key:
            return self._generate_mock_image(prompt, "Nano Banana")
        
        config = self.service_configs["nano_banana"]
        
        # Override defaults with kwargs
        size = kwargs.get("size", config["size"])
        quality = kwargs.get("quality", config["quality"])
        style = kwargs.get("style", config["style"])
        
        # Prepare the request payload for Google Imagen API
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"Generate a high-quality image: {prompt}. Style: {style}, Quality: {quality}, Size: {size}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95
            }
        }
        
        headers = {
            "x-goog-api-key": config['api_key'],
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                config["api_url"],
                json=payload,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract data from Google Imagen API response
                try:
                    # Imagen should return image data in the response
                    candidates = result.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        
                        # Look for image data in the response
                        for part in parts:
                            if "inlineData" in part:
                                # This contains the actual image data
                                image_data = part["inlineData"]
                                mime_type = image_data.get("mimeType", "image/png")
                                
                                # Save the image to disk
                                image_path = self.save_base64_image(
                                    image_data.get("data", ""),
                                    prompt,
                                    "Google Imagen"
                                )
                                
                                return {
                                    "success": True,
                                    "image_url": "data:" + mime_type + ";base64," + image_data.get("data", ""),
                                    "local_path": image_path,
                                    "service": "Google Imagen",
                                    "prompt": prompt,
                                    "metadata": {
                                        "size": size,
                                        "quality": quality,
                                        "style": style,
                                        "model": config["model"],
                                        "mime_type": mime_type,
                                        "note": "Image generated by Google Imagen API"
                                    }
                                }
                        
                        # If no image data found, return the text response
                        text_content = parts[0].get("text", "") if parts else ""
                        return {
                            "success": True,
                            "image_url": "https://via.placeholder.com/512x512/cccccc/666666?text=Imagen+Response",
                            "service": "Google Imagen",
                            "prompt": prompt,
                            "generated_text": text_content,
                            "metadata": {
                                "size": size,
                                "quality": quality,
                                "style": style,
                                "model": config["model"],
                                "note": "Imagen response received but no image data found"
                            }
                        }
                    else:
                        return self._generate_mock_image(prompt, "Google Imagen")
                except Exception as parse_error:
                    print(f"Error parsing response: {parse_error}")
                    return self._generate_mock_image(prompt, "Google Generative AI")
            else:
                print(f"Google Imagen API error: {response.status_code} - {response.text}")
                return self._generate_mock_image(prompt, "Google Imagen")
                
        except Exception as e:
            print(f"Error generating Google Imagen image: {e}")
            return self._generate_mock_image(prompt, "Google Imagen")
    
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
                
                # Try to save image locally if we have image data
                local_path = ""
                if result.get("image_data"):
                    local_path = self.save_base64_image(
                        result.get("image_data", ""),
                        prompt,
                        "Stable Diffusion"
                    )
                elif result.get("image_url"):
                    # Download and save from URL
                    local_path = self.save_image_from_url(
                        result.get("image_url", ""),
                        prompt,
                        "Stable Diffusion"
                    )
                
                return {
                    "success": True,
                    "image_url": result.get("image_url", ""),
                    "image_data": result.get("image_data", ""),
                    "local_path": local_path,
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
        # Create a simple mock image (1x1 pixel PNG)
        mock_image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        # Save the mock image locally
        image_path = self.save_base64_image(mock_image_data, prompt, f"Mock_{service}")
        
        return {
            "success": False,
            "image_url": f"https://via.placeholder.com/512x512/cccccc/666666?text=Mock+{service}+Image",
            "local_path": image_path,
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
    
    def save_image_from_url(self, image_url: str, prompt: str, service: str = "unknown") -> str:
        """
        Download and save an image from URL to local file.
        
        Args:
            image_url: URL of the image to download
            prompt: The prompt used to generate the image
            service: The service that generated the image
            
        Returns:
            Path to the saved image file
        """
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Create filename from prompt and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Clean prompt for filename (remove special characters)
            clean_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            clean_prompt = clean_prompt.replace(' ', '_')
            
            # Determine file extension from content type
            content_type = response.headers.get('content-type', 'image/png')
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            else:
                ext = '.png'  # default
            
            filename = f"{service}_{timestamp}_{clean_prompt}{ext}"
            filepath = os.path.join(self.images_dir, filename)
            
            # Save image to disk
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"💾 Image downloaded and saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error downloading image: {e}")
            return ""
    
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
