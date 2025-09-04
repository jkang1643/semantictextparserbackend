#!/bin/bash
# CPU-Only Installation Script for Text-to-Image Pipeline
# This script ensures ONLY CPU versions are installed - NO GPU/CUDA dependencies

set -e  # Exit on any error

echo "🚀 Installing CPU-only dependencies for Text-to-Image Pipeline..."
echo "⚠️  This installation will ONLY use CPU versions - NO GPU support"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Error: Please activate a virtual environment first!"
    echo "   Run: python -m venv venv && source venv/bin/activate"
    exit 1
fi

echo "✅ Virtual environment detected: $VIRTUAL_ENV"

# Install core dependencies first
echo "📦 Installing core dependencies..."
pip install -r requirements.txt

# Install PyTorch CPU-only version explicitly
echo "🧠 Installing PyTorch CPU-only version..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other ML dependencies
echo "🤖 Installing ML dependencies..."
pip install spacy sentence-transformers transformers scikit-learn

# Download spaCy model
echo "📚 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Verify CPU-only installation
echo "🔍 Verifying CPU-only installation..."
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print('❌ WARNING: CUDA is available - this should not happen!')
    exit(1)
else:
    print('✅ Confirmed: CPU-only installation successful')
"

# Test imports
echo "🧪 Testing pipeline imports..."
python -c "
from text_to_image_pipeline import TextToImagePipeline
from text_to_image_pipeline_lite import TextToImagePipelineLite
print('✅ Both pipelines import successfully')
"

echo "🎉 Installation complete! CPU-only dependencies installed successfully."
echo "📝 You can now use both lite and full pipelines with CPU-only processing."
