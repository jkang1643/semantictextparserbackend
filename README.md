# Text-to-Image Pipeline

A sophisticated Python pipeline that converts raw text (stories, transcripts, articles) into visual scenes using AI image generation. The system intelligently segments text, generates descriptive scene prompts, and creates images using services like Google Nano Banana or Stable Diffusion.

## 🚀 Features

### Core Pipeline
- **Text Preprocessing**: Noise removal, normalization, and entity extraction
- **Robust Scene Segmentation**: Multi-signal analysis for meaningful scene boundaries
- **Intelligent Scene Segmentation**: Advanced narrative-aware text chunking
- **Prompt Enhancement**: LLM-powered scene description generation
- **Image Generation**: Support for Google Nano Banana and other AI image services
- **Character Tracking**: Maintains character consistency across scenes

### Advanced Capabilities
- **Multi-Signal Scene Analysis**: Combines semantic similarity, entity tracking, visual tokens, and cue words
- **Hysteresis Control**: Prevents over-segmentation with dual thresholds
- **Target Scene Count**: Optional peak detection for consistent output
- **Intelligent Scene Detection**: Groups text into coherent scenes based on narrative flow
- **Character & Location Tracking**: Detects character introductions and location changes
- **Semantic Context Awareness**: Uses sentence embeddings for intelligent grouping
- **Context-Aware Prompts**: Uses previous chunks for scene continuity
- **Style Variations**: Realistic, artistic, cinematic, and fantasy styles
- **Batch Processing**: Handle multiple texts efficiently
- **Image Variations**: Generate multiple versions of each scene
- **Local Image Saving**: Download and save generated images

## 📋 Requirements

- Python 3.8+
- Google Nano Banana API key (for image generation)
- Optional: OpenAI API key (for GPT prompt generation)
- Optional: Stable Diffusion API access

## 🛠️ Installation

### ⚠️ CPU-Only Installation (Recommended)

This project uses **CPU-only dependencies** to avoid GPU/CUDA requirements [[memory:8056467]]. Use the provided installation scripts:

**Option 1: Automated Script (Recommended)**
```bash
# Linux/Mac
./install_cpu_only.sh

# Windows
install_cpu_only.bat

# Cross-platform Python script
python install_cpu_only.py
```

**Option 2: Manual Installation**
```bash
# 1. Clone the repository
git clone <repository-url>
cd backendtextparser

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements_cpu_only.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm
```

### 📦 Pipeline Options

**Lite Pipeline (Fast, Basic):**
```bash
pip install -r requirements_minimal.txt
python test_cli.py --demo
```

**Full Pipeline (Advanced, CPU-only):**
```bash
pip install -r requirements_cpu_only.txt
python main.py --demo
```

### 🔑 Environment Setup

Create a `.env` file in the project root:
```env
NANO_BANANA_API_KEY=your_nano_banana_api_key_here
NANO_BANANA_API_URL=https://api.nanobanana.ai/v1/images/generations
OPENAI_API_KEY=your_openai_api_key_here
STABLE_DIFFUSION_API_URL=your_stable_diffusion_api_url
STABLE_DIFFUSION_API_KEY=your_stable_diffusion_api_key
```

## 🎯 Quick Start

### Simple Usage (Recommended)

**Basic demo:**
```bash
python main.py --demo
```

**Process a story file:**
```bash
python main.py --file story.txt --complexity 5 --style cinematic --save-images
```

**Process custom text:**
```bash
python main.py --input "A knight rode through the misty forest..." --complexity 3 --style fantasy
```

### Complexity Levels (Choose One)

| Complexity | Scenes | Best For |
|------------|--------|----------|
| **2-3** | 2-3 scenes | Image generation, storyboards |
| **5** | 4-6 scenes | **Default - most use cases** |
| **8** | 6-8 scenes | Detailed analysis |
| **10** | 8-10+ scenes | Maximum detail |

### Visual Styles (Choose One)

- `realistic` - Photographic quality
- `artistic` - Illustrated/painted style  
- `cinematic` - Movie-like composition
- `fantasy` - Magical/fantastical elements

### Segmentation Methods (Choose One)

- `robust_scene` - **Default** - Advanced NLP analysis with character tracking
- `semantic` - Smart grouping using sentence embeddings
- `rule_based` - Fast, simple paragraph/sentence chunking

### Complete Examples

```bash
# Simple story processing (recommended)
python main.py --file my_story.txt --complexity 5 --style cinematic --save-images

# Quick image generation
python main.py --file story.txt --complexity 3 --style realistic --save-images

# Detailed analysis with specific segmentation
python main.py --file novel.txt --complexity 8 --style artistic --segmentation robust_scene --save-images

# Fast processing with rule-based segmentation
python main.py --file story.txt --complexity 5 --style cinematic --segmentation rule_based --save-images
```

### Testing

```bash
# Test the system
python test_revised_algorithm.py

# Lightweight testing
python test_cli.py --demo --style fantasy
```

### Programmatic Usage

```python
from text_to_image_pipeline import TextToImagePipeline

# Initialize pipeline
pipeline = TextToImagePipeline(
    segmentation_method="semantic",
    image_service="nano_banana",
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
├── main.py                       # Main command-line interface
├── test_cli.py                   # Lightweight CLI for testing
├── text_to_image_pipeline.py     # Full pipeline orchestrator
├── text_to_image_pipeline_lite.py # Lightweight pipeline
├── scene_segmenter.py            # Robust scene segmentation
├── text_processor.py             # Text preprocessing
├── prompt_generator.py           # Scene prompt generation
├── image_generator.py            # AI image generation services
├── parameter_guide.py            # Parameter optimization guide
├── demo.txt                      # Demo text for testing
├── scene_segmenter_presets.json  # Parameter presets
├── requirements_cpu_only.txt     # CPU-only dependencies
├── requirements_minimal.txt      # Minimal dependencies
└── .env                         # Environment variables (create this)
```

## 🔧 Configuration Options

### **Segmentation Methods**
- **Rule-based**: Fast, simple paragraph/sentence chunking
- **Semantic**: Smart grouping using sentence embeddings  
- **Robust Scene**: spaCy parsing + all-MiniLM-L6-v2 embeddings + multi-signal analysis
- **Intelligent Scene**: Narrative-aware with character/location tracking

### **Image Services**
- **Google Nano Banana**: High-quality images, good prompt understanding
- **Stable Diffusion**: More control, lower cost

### **Visual Styles**
- **Realistic**: Photographic quality
- **Artistic**: Illustrated/painted style  
- **Cinematic**: Movie-like composition
- **Fantasy**: Magical/fantastical elements

## 🎭 Scene Segmentation Comparison

| Method | Granularity | Intelligence | Speed | Use Case |
|--------|-------------|--------------|-------|----------|
| Rule-based | High (many small chunks) | Low | Fast | Quick processing |
| Semantic | Medium | Medium | Medium | Balanced approach |
| **Robust Scene** | **Low (few coherent scenes)** | **Very High** | **Medium** | **Image generation** |
| **Intelligent Scene** | **Low (few coherent scenes)** | **High** | **Medium** | **Story visualization** |

### **Robust Scene Segmentation Implementation**

The robust scene segmenter uses advanced NLP techniques for precise text analysis:

#### **1. spaCy Parsing & Feature Extraction**
- **Sentence splitting**: Uses spaCy's sentence tokenizer for accurate sentence boundaries
- **Main subject detection**: Extracts subjects via dependency parsing (`nsubj`), with fallback to first `PROPN`/`NOUN`
- **Entity extraction**: Focuses on key entity types: `PERSON`, `GPE`/`LOC`, `FAC`, `ORG`, `NORP`, `DATE`/`TIME`
- **Linguistic features**: Extracts noun chunks (objects), verbs (actions), adjectives (tone)
- **Visual tokens**: Combines nouns, proper nouns, adjectives, and key verbs for scene context

#### **2. Sentence Embeddings**
- **Model**: Uses `all-MiniLM-L6-v2` sentence transformer for semantic understanding
- **Context tracking**: Maintains exponential moving average (EMA) of scene embeddings
- **Similarity analysis**: Computes cosine similarity between new sentences and scene context

#### **3. Multi-Signal Novelty Scoring**
- **Semantic novelty**: 1 - cosine similarity with scene context
- **Entity novelty**: Jaccard distance of entity sets
- **Visual novelty**: Jaccard distance of visual token sets  
- **Cue bonus**: Small boost for transition phrases (soft hint only)

#### **4. Segmentation Strategies**
- **Target scenes mode**: Peak detection for exact scene count
- **Hysteresis mode**: Dual thresholds prevent over-segmentation
- **Cooldown periods**: Minimum sentences between boundaries

### **Revised Hysteresis Algorithm (Latest Update)**

The algorithm has been enhanced with improved feature extraction and character consistency:

#### **Enhanced Scene Summary Building**
- **subjects**: Dominant nsubj/PERSON entities with character consistency tracking
- **setting**: GPE/LOC/FAC entities + frequent location nouns (excluded if not detected)
- **objects**: Top noun chunks for visual context
- **action_verbs**: Top verb lemmas for scene actions
- **tone_adjs**: Frequent adjectives for mood and atmosphere

#### **Improved Image Prompt Generation**
- **Append strategy**: Structured prompt is appended to original text instead of replacing it
- **Format**: `"{original_text} {setting}; {subjects} {action_verbs}; key objects: {objects}; mood: {tone_adjs}"`
- **Location handling**: Excludes setting from prompt if no location is detected
- **Character consistency**: Maintains same character across scenes until next character change

#### **Advanced Location Detection**
- **Entity prioritization**: First looks for GPE/LOC/FAC entities from spaCy NER
- **Keyword matching**: Falls back to comprehensive location noun detection
- **Smart exclusion**: Removes location from prompt if not found rather than using "unknown location"

#### **Character Consistency Tracking**
- **Character detection**: Automatically identifies primary subject in each scene
- **Consistency maintenance**: Keeps same character until next character change
- **Metadata tracking**: Each scene includes 'character' field for consistency

### Best Practices

- **For Image Generation**: Use `target_scenes=3` for consistent 3-scene output
- **For Story Analysis**: Use `hysteresis=(0.3, 0.6)` for balanced segmentation  
- **For Detailed Processing**: Use `hysteresis=(0.05, 0.15)` with `min_sentences=1`
- **For Quick Testing**: Use `test_cli.py` for lightweight processing
- **For Consistent Results**: Use parameter presets from `scene_segmenter_presets.json`


## 🎛️ Parameter Guide

The robust scene segmenter offers flexible parameters to achieve different granularity levels. Based on extensive testing, here are the recommended settings:

### **Quick Settings by Use Case**

#### **🖼️ Image Generation (2-3 scenes)**
Perfect for creating visual storyboards with fewer, more meaningful scenes:

```bash
# Command line - Simple complexity levels
python main.py --file story.txt --complexity 2

# Command line - Target specific number of scenes
python main.py --file story.txt --target-scenes 3

# Programmatic
scenes = segment_scenes(text, target_scenes=3, min_sentences=3)
```

#### **📊 Balanced Analysis (4-6 scenes)**
Good for detailed text analysis while maintaining scene coherence:

```bash
# Command line - Balanced complexity
python main.py --file story.txt --complexity 5

# Command line - Custom target scenes
python main.py --file story.txt --target-scenes 5

# Programmatic
scenes = segment_scenes(text, hysteresis=(0.3, 0.6), min_sentences=2)
```

#### **🔍 Detailed Analysis (8-10+ scenes)**
For fine-grained text segmentation and detailed scene analysis:

```bash
# Command line - High complexity
python main.py --file story.txt --complexity 9

# Command line - Maximum detail
python main.py --file story.txt --complexity 10

# Programmatic
scenes = segment_scenes(text, hysteresis=(0.05, 0.15), min_sentences=1)
```

### **Complexity Scale (0-10)**

The new `--complexity` parameter simplifies scene segmentation with intuitive levels:

| Complexity | Description | Expected Scenes | Use Case |
|------------|-------------|-----------------|----------|
| **0-2** | Very Simple | 2-3 scenes | Image generation, storyboards |
| **3-4** | Simple | 3-4 scenes | Fewer scenes, focused content |
| **5-6** | Balanced | 4-6 scenes | Default, general purpose |
| **7-8** | Detailed | 6-8 scenes | More granular analysis |
| **9-10** | Very Detailed | 8-10+ scenes | Maximum detail, fine-grained |

### **Parameter Reference**

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `target_scenes` | Exact number of scenes to create | 3 (image gen), 6 (balanced), 10 (detailed) |
| `min_sentences` | Minimum sentences per scene | 3 (coarse), 2 (balanced), 1 (fine) |
| `hysteresis` | (stay_inside, enter_boundary) thresholds | (0.6,0.8) coarse, (0.3,0.6) balanced, (0.05,0.15) fine |
| `weights` | Novelty scoring weights | Default: semantic=0.55, entity=0.25, visual=0.15, cue=0.05 |

### **Testing & Validation**

```bash
# Run parameter guide to see all options
python parameter_guide.py

# Test scene segmentation only
python -m scene_segmenter demo.txt --target-scenes 4

# Run full test suite
python test_scene_segmenter.py
```

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

### **Traditional Pipeline Output**

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

### **Revised Algorithm Output (Latest)**

**Input Text:**
```
John walked through the dark forest, his footsteps muffled by the thick layer of fallen leaves. 
The ancient trees seemed to whisper secrets as the wind rustled through their branches. 
He carried a worn leather satchel containing his most precious belongings.
```

**Generated Scene Summary:**
- **subjects**: ['john', 'tree', 'leather']
- **setting**: ['the dark forest'] 
- **objects**: ['john', 'the dark forest', 'his footstep', 'the thick layer', 'fall leave']
- **action_verbs**: ['walk', 'muffle', 'fall']
- **tone_adjs**: ['dark', 'thick', 'ancient']
- **character**: 'john'

**Generated Prompt:**
```
John walked through the dark forest, his footsteps muffled by the thick layer of fallen leaves. 
The ancient trees seemed to whisper secrets as the wind rustled through their branches. 
He carried a worn leather satchel containing his most precious belongings. 
the dark forest; john, tree walk, muffle; key objects: john, the dark forest, his footstep; mood: dark, thick
```

**Images:** Generated AI images using nano banana model with combined original text + structured summary

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

- [x] **Intelligent Scene Segmentation** - Advanced narrative-aware text chunking
- [x] **Character & Location Tracking** - Automatic detection of scene changes
- [x] **Scene Title Generation** - Automatic scene description generation
- [x] **Robust Scene Segmentation** - Multi-signal analysis for meaningful scene boundaries
- [x] **Revised Hysteresis Algorithm** - Enhanced feature extraction with character consistency
- [x] **Improved Image Prompt Generation** - Append strategy with structured summaries
- [x] **Advanced Location Detection** - Smart location handling with exclusion logic
- [x] **Character Consistency Tracking** - Maintains character continuity across scenes
- [ ] Coreference resolution for better character tracking
- [ ] Scene continuity analysis
- [ ] Multi-language scene segmentation support
- [ ] Custom transition word configuration
- [ ] Scene type classification (dialogue, action, description)
- [ ] Style transfer between images
- [ ] Video generation from sequential images
- [ ] Web interface for easy usage
- [ ] Support for more image generation services
