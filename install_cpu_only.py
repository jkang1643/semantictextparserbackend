#!/usr/bin/env python3
"""
CPU-Only Installation Script for Text-to-Image Pipeline
This script ensures ONLY CPU versions are installed - NO GPU/CUDA dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"   Command: {cmd}")
        print(f"   Error: {e.stderr}")
        return False

def check_virtual_env():
    """Check if we're in a virtual environment."""
    if not os.environ.get('VIRTUAL_ENV'):
        print("❌ Error: Please activate a virtual environment first!")
        print("   Run: python -m venv venv && source venv/bin/activate")
        return False
    
    print(f"✅ Virtual environment detected: {os.environ['VIRTUAL_ENV']}")
    return True

def verify_cpu_only():
    """Verify that only CPU versions are installed."""
    print("🔍 Verifying CPU-only installation...")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print("❌ WARNING: CUDA is available - this should not happen!")
            return False
        else:
            print("✅ Confirmed: CPU-only installation successful")
            return True
    except ImportError:
        print("❌ PyTorch not found - installation may have failed")
        return False

def test_imports():
    """Test that both pipelines can be imported."""
    print("🧪 Testing pipeline imports...")
    try:
        from text_to_image_pipeline import TextToImagePipeline
        from text_to_image_pipeline_lite import TextToImagePipelineLite
        print("✅ Both pipelines import successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main installation process."""
    print("🚀 Installing CPU-only dependencies for Text-to-Image Pipeline...")
    print("⚠️  This installation will ONLY use CPU versions - NO GPU support")
    
    # Check virtual environment
    if not check_virtual_env():
        sys.exit(1)
    
    # Installation steps
    steps = [
        ("pip install -r requirements.txt", "Installing core dependencies"),
        ("pip install torch --index-url https://download.pytorch.org/whl/cpu", "Installing PyTorch CPU-only version"),
        ("pip install spacy sentence-transformers transformers scikit-learn", "Installing ML dependencies"),
        ("python -m spacy download en_core_web_sm", "Downloading spaCy English model")
    ]
    
    # Execute installation steps
    for cmd, description in steps:
        if not run_command(cmd, description):
            print(f"❌ Installation failed at: {description}")
            sys.exit(1)
    
    # Verify installation
    if not verify_cpu_only():
        print("❌ CPU-only verification failed")
        sys.exit(1)
    
    if not test_imports():
        print("❌ Import testing failed")
        sys.exit(1)
    
    print("🎉 Installation complete! CPU-only dependencies installed successfully.")
    print("📝 You can now use both lite and full pipelines with CPU-only processing.")

if __name__ == "__main__":
    main()
