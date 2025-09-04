# CPU-Only Setup Guide

This document ensures that the Text-to-Image Pipeline uses **ONLY CPU versions** of all dependencies, with **NO GPU/CUDA support**.

## 🎯 Why CPU-Only?

- **Compatibility**: Works on any system without GPU requirements
- **Simplicity**: No CUDA driver installation needed
- **Reliability**: Avoids GPU memory issues and driver conflicts
- **Portability**: Easy deployment across different environments

## 📦 Installation Methods

### Method 1: Automated Scripts (Recommended)

**Linux/Mac:**
```bash
./install_cpu_only.sh
```

**Windows:**
```cmd
install_cpu_only.bat
```

**Cross-platform:**
```bash
python install_cpu_only.py
```

### Method 2: Manual Installation

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Install PyTorch CPU-only (CRITICAL STEP)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install ML dependencies
pip install spacy sentence-transformers transformers scikit-learn

# 5. Download spaCy model
python -m spacy download en_core_web_sm
```

## 🔍 Verification

Run the verification script to ensure CPU-only installation:

```bash
python verify_cpu_only.py
```

**Expected output:**
```
🎉 All checks passed! CPU-only installation verified.
✅ You can safely use both lite and full pipelines.
```

## 📋 Requirements Files

### `requirements.txt` (Lite Pipeline)
- Core dependencies only
- No heavy ML libraries
- Fast installation

### `requirements_cpu_only.txt` (Full Pipeline)
- All ML dependencies
- Explicitly excludes PyTorch (must be installed separately)
- CPU-only constraints

### `requirements_full.txt` (Full Pipeline - Alternative)
- Includes all dependencies
- Contains CPU-only installation instructions
- More verbose documentation

## ⚠️ Critical Points

### 1. PyTorch Installation
**NEVER** install PyTorch from default PyPI:
```bash
# ❌ WRONG - may include CUDA
pip install torch

# ✅ CORRECT - CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Package Verification
Always verify the installation:
```python
import torch
print(torch.__version__)        # Should show "+cpu"
print(torch.cuda.is_available()) # Should be False
```

### 3. Dependencies
The following packages are **CPU-only**:
- `torch==2.8.0+cpu`
- `spacy` (no GPU acceleration)
- `sentence-transformers` (CPU inference)
- `transformers` (CPU inference)
- `scikit-learn` (CPU-only)

## 🚨 Troubleshooting

### If CUDA is detected:
1. Uninstall PyTorch: `pip uninstall torch`
2. Reinstall CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
3. Verify: `python verify_cpu_only.py`

### If installation fails:
1. Check virtual environment is activated
2. Update pip: `pip install --upgrade pip`
3. Clear pip cache: `pip cache purge`
4. Try installation script: `python install_cpu_only.py`

### If imports fail:
1. Verify all dependencies: `pip list`
2. Check spaCy model: `python -m spacy download en_core_web_sm`
3. Test individual components

## 📊 Performance Notes

### CPU-Only Performance:
- **Startup time**: 3-5 seconds (vs 1 second for lite)
- **Memory usage**: ~500MB (vs ~100MB for lite)
- **Processing speed**: Slower but still functional
- **Quality**: Full semantic understanding

### When to Use:
- **Lite Pipeline**: Quick prototyping, simple text
- **Full Pipeline**: Complex text, high-quality analysis needed

## 🔧 Configuration

Both pipelines work with CPU-only dependencies:

```python
# Lite pipeline
from text_to_image_pipeline_lite import TextToImagePipelineLite
pipeline = TextToImagePipelineLite(segmentation_method='rule_based')

# Full pipeline
from text_to_image_pipeline import TextToImagePipeline
pipeline = TextToImagePipeline(segmentation_method='semantic')
```

## ✅ Final Checklist

- [ ] Virtual environment activated
- [ ] PyTorch installed with CPU-only index
- [ ] All ML dependencies installed
- [ ] spaCy model downloaded
- [ ] Verification script passes
- [ ] Both pipelines import successfully
- [ ] No CUDA packages detected

## 📞 Support

If you encounter issues:
1. Run `python verify_cpu_only.py`
2. Check the troubleshooting section
3. Ensure you're using the CPU-only installation method
4. Verify your virtual environment is properly activated
