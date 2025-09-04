# Pipeline Comparison: Lite vs Full

This document outlines the differences between the lightweight and full text-to-image pipelines.

## 🚀 Lite Pipeline (`text_to_image_pipeline_lite.py`)

### Dependencies
- **Minimal**: Only essential packages
- **Size**: ~30 packages, ~200MB
- **Installation**: Fast (< 1 minute)

### Components
- `TextProcessorLite`: Basic text preprocessing with NLTK
- `TextSegmenterLite`: Rule-based and simple semantic segmentation
- `PromptGenerator`: OpenAI-based prompt enhancement
- `ImageGenerator`: Image generation service

### Segmentation Methods
1. **Rule-based**: Simple sentence-based chunking
2. **Semantic (Simple)**: Word overlap similarity

### Pros
- ✅ Fast installation and startup
- ✅ Low memory usage
- ✅ No heavy ML dependencies
- ✅ Works on any system

### Cons
- ❌ Less sophisticated text analysis
- ❌ Basic semantic understanding
- ❌ Limited entity recognition

---

## 🧠 Full Pipeline (`text_to_image_pipeline.py`)

### Dependencies
- **Heavy**: Full ML stack with PyTorch, spaCy, transformers
- **Size**: ~60+ packages, ~2GB+
- **Installation**: Slower (5-10 minutes)

### Components
- `TextProcessor`: Advanced NLP with spaCy
- `TextSegmenter`: Sophisticated semantic segmentation with sentence-transformers
- `PromptGenerator`: OpenAI-based prompt enhancement
- `ImageGenerator`: Image generation service

### Segmentation Methods
1. **Rule-based**: Advanced sentence-based chunking
2. **Semantic (Advanced)**: Sentence transformer embeddings with cosine similarity

### Pros
- ✅ Advanced text understanding
- ✅ Better entity recognition
- ✅ Sophisticated semantic segmentation
- ✅ Higher quality text analysis

### Cons
- ❌ Heavy dependencies
- ❌ Slower startup time
- ❌ Higher memory usage
- ❌ Requires more system resources

---

## 📊 Performance Comparison

| Feature | Lite Pipeline | Full Pipeline |
|---------|---------------|---------------|
| **Installation Time** | < 1 minute | 5-10 minutes |
| **Startup Time** | < 1 second | 3-5 seconds |
| **Memory Usage** | ~100MB | ~500MB+ |
| **Text Analysis Quality** | Basic | Advanced |
| **Semantic Understanding** | Simple | Sophisticated |
| **Entity Recognition** | Regex-based | NLP-based |

---

## 🎯 When to Use Which

### Use Lite Pipeline When:
- Quick prototyping
- Limited system resources
- Simple text processing needs
- Fast deployment required
- Basic segmentation is sufficient

### Use Full Pipeline When:
- High-quality text analysis needed
- Complex semantic understanding required
- Advanced entity recognition important
- System resources are abundant
- Maximum segmentation quality desired

---

## 🔧 Installation Commands

### Lite Pipeline
```bash
pip install -r requirements.txt
```

### Full Pipeline
```bash
pip install -r requirements_full.txt
python -m spacy download en_core_web_sm
```

---

## 🧪 Testing Commands

### Test Lite Pipeline
```python
from text_to_image_pipeline_lite import TextToImagePipelineLite
pipeline = TextToImagePipelineLite(segmentation_method='rule_based')
```

### Test Full Pipeline
```python
from text_to_image_pipeline import TextToImagePipeline
pipeline = TextToImagePipeline(segmentation_method='semantic')
```

---

## 📝 Notes

- Both pipelines use the same `PromptGenerator` and `ImageGenerator`
- The main difference is in text processing and segmentation capabilities
- Full pipeline requires CPU-only PyTorch (no GPU dependencies)
- Both pipelines support the same image generation services
- Results quality depends on the complexity of your input text
