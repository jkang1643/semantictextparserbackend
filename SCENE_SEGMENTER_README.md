# Robust Scene Segmenter

A Python module that segments text into image-ready scenes based on meaningful shifts in subject, setting, and semantics. Transition words act as soft hints, not hard boundaries.

## Features

- **Multi-signal analysis**: Combines semantic similarity, entity tracking, visual tokens, and cue words
- **Hysteresis control**: Prevents over-segmentation with dual thresholds
- **Target scene count**: Optional peak detection for consistent output
- **Image-ready prompts**: Generates concise prompts for each scene
- **Robust fallback**: Works with or without spaCy

## Installation

### CPU-Only Installation (Recommended)

```bash
# Install PyTorch CPU-only first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install sentence-transformers scikit-learn spacy numpy

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Alternative: Use requirements file

```bash
pip install -r requirements_cpu_only.txt
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

```python
from scene_segmenter import segment_scenes

text = """John entered the forest. The trees loomed overhead.
The forest was alive with sounds. Strange markings appeared on the trees.
Suddenly, the temperature dropped. Ahead, he heard a sound.
A woman emerged, dressed in white. Her eyes glowed faintly.
She spoke softly. Then she added another line.
John felt a magical pull toward her."""

# Segment with target scene count
scenes = segment_scenes(text, target_scenes=3, min_sentences=3)

for i, scene in enumerate(scenes, 1):
    print(f"--- Scene {i} ---")
    print(f"Text: {scene['text']}")
    print(f"Prompt: {scene['prompt']}")
    print(f"Summary: {scene['summary']}")
```

### CLI Usage

```bash
# Basic segmentation
python -m scene_segmenter demo.txt

# With target scenes
python -m scene_segmenter demo.txt --target-scenes 4

# With custom minimum sentences
python -m scene_segmenter demo.txt --min-sentences 2

# Save to file
python -m scene_segmenter demo.txt --output results.txt
```

### Advanced Parameters

```python
scenes = segment_scenes(
    text=text,
    target_scenes=3,                    # Target number of scenes
    min_sentences=3,                   # Minimum sentences per scene
    cooldown=1,                        # Minimum sentences between boundaries
    hysteresis=(0.58, 0.68),          # (stay_inside, enter_boundary) thresholds
    weights={                          # Novelty scoring weights
        "semantic": 0.55,              # Semantic similarity weight
        "entity": 0.25,                # Entity change weight
        "visual": 0.15,                # Visual token weight
        "cue": 0.05                    # Transition word weight
    }
)
```

## How It Works

### 1. Multi-Signal Novelty Scoring

For each sentence, the system computes:

- **Semantic novelty**: 1 - cosine similarity with scene context
- **Entity novelty**: Jaccard distance of entities (PERSON, GPE, LOC, etc.)
- **Visual novelty**: Jaccard distance of visual tokens (nouns, objects, verbs)
- **Cue bonus**: Small boost for transition phrases (soft hint only)

### 2. Scene Context Tracking

- **EMA embedding**: Exponential moving average of sentence embeddings
- **Entity sets**: Tracks active entities in current scene
- **Visual tokens**: Tracks objects and visual elements
- **Subject tracking**: Monitors dominant subjects

### 3. Segmentation Strategies

**Target Scenes Mode**:
- Computes novelty scores for all positions
- Smooths with moving average
- Finds local maxima peaks
- Selects top N-1 peaks respecting spacing

**Hysteresis Mode**:
- Uses dual thresholds (stay_inside, enter_boundary)
- Prevents over-segmentation with cooldown periods
- Respects minimum sentence requirements

### 4. Image Prompt Generation

Each scene gets a concise prompt:
```
"{setting}; {subjects} {actions}; key objects: {objects}; mood: {mood}"
```

## Testing

Run the test suite:

```bash
python test_scene_segmenter.py
```

Run the demo:

```bash
python demo_scene_segmenter.py
```

## Expected Behavior

### Less-Granular Segmentation

The system is designed to produce **fewer, smarter scenes** by:

- **Ignoring cue words alone**: "Suddenly" doesn't split if context is stable
- **Focusing on real shifts**: Subject changes, location changes, semantic drift
- **Using hysteresis**: Prevents micro-segmentation
- **Respecting spacing**: Minimum sentences between boundaries

### Example Output

**Input**: 10 sentences with multiple transition words

**Expected**: 2-4 scenes (not 8-10 micro-scenes)

**Scene 1**: Forest exploration (4-5 sentences)
**Scene 2**: Temperature drop and sound (2-3 sentences)  
**Scene 3**: Woman appearance and interaction (2-3 sentences)

## Dependencies

- `sentence-transformers>=2.2.0` (all-MiniLM-L6-v2 model)
- `scikit-learn>=1.3.0` (cosine similarity)
- `spacy>=3.7.0` (linguistic analysis)
- `numpy>=1.24.0` (numerical operations)
- `torch>=2.0.0` (CPU-only version)

## License

This module is part of the backend text parser project.
