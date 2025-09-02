#!/usr/bin/env python3
"""
Test script for Google Nano Banana integration

This script tests the updated ImageGenerator class with Nano Banana support.
"""

import os
from dotenv import load_dotenv
from image_generator import ImageGenerator

# Load environment variables
load_dotenv()

def test_nano_banana_integration():
    """Test the Google Imagen integration."""
    print("🧪 Testing Google Imagen Integration")
    print("=" * 50)
    
    # Check environment variables
    nano_banana_key = os.getenv("NANO_BANANA_API_KEY")
    nano_banana_url = os.getenv("NANO_BANANA_API_URL")
    
    print(f"🔑 Google Imagen API Key: {'✅ Set' if nano_banana_key else '❌ Not set'}")
    print(f"🌐 Google Imagen API URL: {nano_banana_url or 'Using default'}")
    print()
    
    # Test ImageGenerator initialization
    print("🚀 Testing ImageGenerator initialization...")
    try:
        image_gen = ImageGenerator(service="nano_banana")
        print("✅ ImageGenerator initialized successfully")
        print(f"   Service: {image_gen.service}")
        print(f"   API Key available: {'Yes' if image_gen.nano_banana_api_key else 'No'}")
        print(f"   API URL: {image_gen.nano_banana_api_url}")
    except Exception as e:
        print(f"❌ Error initializing ImageGenerator: {e}")
        return
    
    print()
    
    # Test image generation
    print("🎨 Testing image generation...")
    test_prompt = "A beautiful serene forest scene with golden sunlight filtering through tall trees, photorealistic style"
    
    try:
        result = image_gen.generate_image(test_prompt, size="1024x1024")
        print("✅ Image generation completed")
        print(f"   Success: {result['success']}")
        print(f"   Service: {result['service']}")
        print(f"   Image URL: {result['image_url']}")
        
        if 'revised_prompt' in result:
            print(f"   Revised Prompt: {result['revised_prompt']}")
        
        print(f"   Metadata: {result['metadata']}")
        
    except Exception as e:
        print(f"❌ Error during image generation: {e}")
    
    print()
    
    # Test service configuration
    print("⚙️ Testing service configuration...")
    config = image_gen.service_configs.get("nano_banana", {})
    print(f"   Model: {config.get('model', 'Not set')}")
    print(f"   Default Size: {config.get('size', 'Not set')}")
    print(f"   Default Quality: {config.get('quality', 'Not set')}")
    print(f"   Default Style: {config.get('style', 'Not set')}")
    
    print()
    
    # Test fallback to mock image
    print("🔄 Testing fallback to mock image...")
    if not nano_banana_key:
        print("   No API key set, testing mock image generation...")
        mock_result = image_gen._generate_mock_image(test_prompt, "Nano Banana")
        print(f"   Mock image generated: {mock_result['success']}")
        print(f"   Mock service: {mock_result['service']}")
    else:
        print("   API key available, skipping mock test")
    
    print()
    print("🎉 Google Imagen integration test completed!")

def test_service_switching():
    """Test switching between different services."""
    print("\n🔄 Testing Service Switching")
    print("=" * 30)
    
    services = ["nano_banana", "stable_diffusion"]
    
    for service in services:
        print(f"\nTesting {service} service...")
        try:
            image_gen = ImageGenerator(service=service)
            print(f"✅ {service} service initialized")
            
            # Test a simple prompt
            result = image_gen.generate_image("A simple test image")
            print(f"   Generation successful: {result['success']}")
            print(f"   Service used: {result['service']}")
            
        except Exception as e:
            print(f"❌ Error with {service}: {e}")

if __name__ == "__main__":
    test_nano_banana_integration()
    test_service_switching()
