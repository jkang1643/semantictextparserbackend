# Quick Setup Guide

## 🚀 Fast Start (5 minutes)

### 1. Prerequisites
- Python 3.8+ installed
- Git (optional, for cloning)

### 2. Environment Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install minimal dependencies (no heavy ML libraries)
pip install -r requirements_minimal.txt
```

### 3. Test the Full Pipeline
```bash
# Run the comprehensive demo (all features working)
python final_demo.py

# Or run the simplified demo
python simple_demo.py
```

This will process a sample text and generate mock image results, demonstrating the complete pipeline workflow.

## 🔧 Full Installation (Advanced Features)

### 1. Install All Dependencies (Optional)
```bash
# Activate virtual environment first
source venv/bin/activate

# Install all dependencies (may take 5-10 minutes)
pip install -r requirements.txt

# Install spaCy model (only if using full version)
python -m spacy download en_core_web_sm
```

### 2. Set Up API Keys (Optional)
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
STABLE_DIFFUSION_API_URL=your_stable_diffusion_api_url
STABLE_DIFFUSION_API_KEY=your_stable_diffusion_api_key
```

### 3. Test Full Pipeline
```bash
# Run with demo text
python main.py --demo

# Run with custom text
python main.py --input "Your story text here..." --style fantasy

# Run with file input
python main.py --file story.txt --save-images
```

## 📊 What You Get

### Lightweight Pipeline (Recommended)
- ✅ **Text preprocessing** - Cleaning and normalization
- ✅ **Character extraction** - Named entity detection
- ✅ **Rule-based segmentation** - Token-aware chunking
- ✅ **Semantic segmentation** - Word overlap similarity
- ✅ **Context-aware prompts** - Character and scene tracking
- ✅ **Multiple styles** - Realistic, fantasy, cinematic, artistic
- ✅ **Image generation** - Mock/API integration
- ✅ **Variations generation** - Multiple prompt versions
- ✅ **Results export** - JSON with full metadata
- ✅ **Statistics** - Processing metrics and analysis
- ✅ **No heavy dependencies** - Works without torch/transformers

### Full Pipeline (Optional)
- ✅ All lightweight features
- ✅ Advanced semantic segmentation (sentence-transformers)
- ✅ LLM-powered prompt enhancement (OpenAI GPT)
- ✅ Real AI image generation (with API keys)
- ✅ Advanced entity extraction (spaCy)

## 🎯 Quick Examples

### Basic Usage
```python
from text_to_image_pipeline_lite import TextToImagePipelineLite

# Initialize pipeline
pipeline = TextToImagePipelineLite(
    segmentation_method="rule_based",
    image_service="dalle"
)

# Process text
results = pipeline.process_text("Your story here...", style="fantasy")

# Save results
pipeline.save_results("my_results.json")
```

### Command Line
```bash
# Comprehensive demo
python final_demo.py

# Simple demo
python simple_demo.py

# Full pipeline demo
python main.py --demo --style cinematic

# Custom text
python main.py --input "A knight rode through the misty forest..." --style fantasy
```

## 🔍 Expected Output

The pipeline will generate:
1. **Text chunks** - Segmented portions of your story
2. **Scene prompts** - Enhanced descriptions for image generation
3. **Image results** - URLs or local files of generated images
4. **Statistics** - Processing metrics and analysis
5. **JSON file** - Complete results for further processing

## 🚨 Troubleshooting

### Common Issues
1. **Import errors**: Make sure virtual environment is activated
2. **API errors**: Check your `.env` file and API keys (optional)
3. **Memory issues**: Use rule-based segmentation for large texts
4. **Slow processing**: Start with lightweight pipeline first

### Getting Help
1. Check the `README.md` for detailed documentation
2. Review `example_usage.py` for code examples
3. Run `python main.py --help` for command options
4. Check `final_demo_rule_based.json` for output format

## 🎉 Success!

Once you see output like this, you're ready to go:
```
✅ Pipeline completed successfully!
📈 Pipeline Statistics:
   Total chunks: 3
   Characters found: 5
   Success rate: 100.0%
💾 Results saved to final_demo_rule_based.json
```

Your text-to-image pipeline is now working! 🚀

## 🆕 What's New

- **Lightweight Pipeline**: No heavy dependencies required
- **Better Character Extraction**: Improved entity detection
- **Semantic Segmentation**: Word overlap similarity method
- **Comprehensive Demo**: Full feature demonstration
- **Robust Error Handling**: Graceful fallbacks for missing dependencies
