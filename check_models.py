#!/usr/bin/env python3
"""
Check available Google Generative AI models
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_available_models():
    """Check what models are available in the Google Generative AI API."""
    api_key = os.getenv("NANO_BANANA_API_KEY")
    
    if not api_key:
        print("❌ No API key found")
        return
    
    # List models endpoint
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        print("🔍 Checking available models...")
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            models = response.json()
            print("✅ Available models:")
            print("=" * 50)
            
            for model in models.get("models", []):
                name = model.get("name", "Unknown")
                display_name = model.get("displayName", "No display name")
                description = model.get("description", "No description")
                supported_generation_methods = model.get("supportedGenerationMethods", [])
                
                print(f"\n📋 Model: {name}")
                print(f"   Display Name: {display_name}")
                print(f"   Description: {description}")
                print(f"   Supported Methods: {supported_generation_methods}")
                
                # Check if it supports image generation
                if "generateContent" in supported_generation_methods:
                    print("   🎨 Supports: generateContent")
                if "generateText" in supported_generation_methods:
                    print("   📝 Supports: generateText")
                if "embedText" in supported_generation_methods:
                    print("   🔗 Supports: embedText")
                    
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error checking models: {e}")

if __name__ == "__main__":
    check_available_models()
