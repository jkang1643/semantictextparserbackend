# Text-to-Image Pipeline

A sophisticated Python pipeline that converts raw text (stories, transcripts, articles) into visual scenes using AI image generation. The system intelligently segments text, generates descriptive scene prompts, and creates images using services like DALL-E or Stable Diffusion.

## 🚀 Features

### Core Pipeline
- **Text Preprocessing**: Noise removal, normalization, and entity extraction
- **Intelligent Segmentation**: Rule-based and semantic-based text chunking
- **Prompt Enhancement**: LLM-powered scene description generation
- **Image Generation**: Support for multiple AI image services
- **Character Tracking**: Maintains character consistency across scenes

### Advanced Capabilities
- **Multiple Segmentation Methods**: Rule-based (fast) and semantic-based (smart)
- **Context-Aware Prompts**: Uses previous chunks for scene continuity
- **Style Variations**: Realistic, artistic, cinematic, and fantasy styles
- **Batch Processing**: Handle multiple texts efficiently
- **Image Variations**: Generate multiple versions of each scene
- **Local Image Saving**: Download and save generated images

## 📋 Requirements

- Python 3.8+
- OpenAI API key (for DALL-E and GPT prompt generation)
- Optional: Stable Diffusion API access

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd backendtextparser
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install spaCy model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   STABLE_DIFFUSION_API_URL=your_stable_diffusion_api_url
   STABLE_DIFFUSION_API_KEY=your_stable_diffusion_api_key
   ```

## 🎯 Quick Start

### Command Line Usage

**Basic demo:**
```bash
python main.py --demo
```

**Process text from file:**
```bash
python main.py --file story.txt --style cinematic --save-images
```

**Process custom text:**
```bash
python main.py --input "A knight rode through the misty forest..." --style fantasy
```

**Advanced configuration:**
```bash
python main.py --file novel.txt \
    --segmentation semantic \
    --service dalle \
    --max-tokens 256 \
    --similarity-threshold 0.7 \
    --style artistic \
    --save-images \
    --output-dir my_images
```

### Programmatic Usage

```python
from text_to_image_pipeline import TextToImagePipeline

# Initialize pipeline
pipeline = TextToImagePipeline(
    segmentation_method="semantic",
    image_service="dalle",
    max_tokens_per_chunk=512,
    similarity_threshold=0.6
)

# Process text
text = "Your story text here..."
results = pipeline.process_text(
    text=text,
    style="realistic",
    save_images=True,
    output_dir="output"
)

# Access results
for result in results:
    print(f"Chunk: {result['chunk_text']}")
    print(f"Prompt: {result['scene_prompt']}")
    print(f"Image: {result['image_result']['image_url']}")
```

## 📁 Project Structure

```
backendtextparser/
├── text_processor.py          # Text preprocessing and entity extraction
├── text_segmenter.py          # Text segmentation (rule-based & semantic)
├── prompt_generator.py        # LLM-powered scene prompt generation
├── image_generator.py         # AI image generation services
├── text_to_image_pipeline.py  # Main pipeline orchestrator
├── main.py                   # Command-line interface
├── example_usage.py          # Usage examples
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .env                     # Environment variables (create this)
```

## 🔧 Configuration Options

### Segmentation Methods

**Rule-based (Fast):**
- Simple paragraph/sentence-based chunking
- Good for quick processing
- Less intelligent but reliable

**Semantic (Smart):**
- Uses sentence embeddings for similarity
- Groups related sentences together
- Better scene coherence
- Requires more computational resources

### Image Services

**DALL-E:**
- High-quality images
- Good prompt understanding
- Requires OpenAI API key

**Stable Diffusion:**
- More control over generation parameters
- Lower cost
- Requires Stable Diffusion API access

### Visual Styles

- **Realistic**: Photographic quality
- **Artistic**: Illustrated/painted style
- **Cinematic**: Movie-like composition
- **Fantasy**: Magical/fantastical elements

## 📊 Pipeline Statistics

The system provides detailed statistics about processing:

```python
stats = pipeline.get_pipeline_stats()
print(f"Total chunks: {stats['total_chunks']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Characters found: {stats['characters_found']}")
print(f"Average prompt length: {stats['avg_prompt_length']:.1f}")
```

## 🎨 Advanced Features

### Character Consistency
The pipeline automatically extracts and tracks characters across scenes:

```python
# Characters are automatically detected and maintained
results = pipeline.process_text(text)
print(f"Characters: {results[0]['characters']}")
```

### Context-Aware Prompts
Uses previous chunks for scene continuity:

```python
# The pipeline maintains context across chunks
# Each prompt considers previous scenes for consistency
```

### Multiple Variations
Generate different interpretations of the same scene:

```python
# Generate 3 variations of the first chunk
variations = pipeline.generate_variations(0, num_variations=3)
```

## 🔍 Example Output

**Input Text:**
```
John walked into the ancient forest. The towering trees created a natural 
cathedral, their branches intertwining overhead. A woman with silver hair 
emerged from behind a massive oak tree.
```

**Generated Chunks:**
1. "John walked into the ancient forest. The towering trees created a natural cathedral..."
2. "A woman with silver hair emerged from behind a massive oak tree."

**Scene Prompts:**
1. "A man named John walks into a dark ancient forest with towering trees forming a natural cathedral-like canopy overhead, cinematic lighting, atmospheric"
2. "A mysterious woman with glowing silver hair emerges from behind a massive ancient oak tree in a mystical forest setting, ethereal lighting"

**Images:** Generated AI images matching each scene description

## 🚨 Error Handling

The pipeline includes robust error handling:

- **API Failures**: Falls back to mock images when services are unavailable
- **Empty Text**: Graceful handling of empty or invalid input
- **Long Texts**: Automatic chunking for texts of any length
- **Network Issues**: Retry logic and timeout handling

## 🔧 Customization

### Adding New Image Services

Extend the `ImageGenerator` class:

```python
class CustomImageGenerator(ImageGenerator):
    def _generate_custom_image(self, prompt: str, **kwargs):
        # Implement your custom image generation logic
        pass
```

### Custom Prompt Styles

Modify the `PromptGenerator` class:

```python
def generate_scene_prompt(self, text_chunk: str, style: str = "custom"):
    if style == "custom":
        # Add your custom prompt generation logic
        pass
```

## 📈 Performance Tips

1. **Use rule-based segmentation** for faster processing of large texts
2. **Adjust token limits** based on your image service requirements
3. **Batch process** multiple texts for efficiency
4. **Cache embeddings** for repeated semantic segmentation
5. **Use appropriate similarity thresholds** (0.6-0.8 for most use cases)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the example usage scripts
2. Review the error handling section
3. Ensure all dependencies are installed
4. Verify your API keys are correctly set

## 🔮 Future Enhancements

- [ ] Coreference resolution for better character tracking
- [ ] Scene continuity analysis
- [ ] Style transfer between images
- [ ] Video generation from sequential images
- [ ] Web interface for easy usage
- [ ] Support for more image generation services
