# Google Nano Banana Integration Guide

This guide explains how to set up and use Google Nano Banana with the text-to-image pipeline.

## 🚀 What is Google Nano Banana?

Google Nano Banana is Google's latest text-to-image generation model, designed to create high-quality images from text descriptions. It's integrated into our pipeline as the primary image generation service.

## 📋 Prerequisites

- Python 3.8+
- Google Nano Banana API access
- API key from Google AI Studio

## 🔑 Getting Your API Key

1. **Visit Google AI Studio**: Go to [Google AI Studio](https://aistudio.google.com/)
2. **Sign in**: Use your Google account
3. **Navigate to Nano Banana**: Find the Nano Banana model in the available models
4. **Get API Key**: Generate an API key for Nano Banana
5. **Copy the key**: Save it securely for use in your environment

## ⚙️ Environment Setup

1. **Copy the environment template**:
   ```bash
   cp env_example.txt .env
   ```

2. **Edit your `.env` file** and add your Nano Banana credentials:
   ```env
   # Google Nano Banana API Configuration (primary image generation service)
   NANO_BANANA_API_KEY=your_actual_api_key_here
   NANO_BANANA_API_URL=https://api.nanobanana.ai/v1/images/generations
   
   # OpenAI API Key (optional - for GPT prompt generation if needed)
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Save the file** and ensure it's in your project root directory

## 🧪 Testing the Integration

Run the Nano Banana test script to verify everything is working:

```bash
python test_nano_banana.py
```

This will test:
- ✅ Environment variable loading
- ✅ ImageGenerator initialization
- ✅ Nano Banana service configuration
- ✅ Image generation (or mock fallback)
- ✅ Service switching

## 🎯 Usage Examples

### Basic Usage

```python
from image_generator import ImageGenerator

# Initialize with Nano Banana
image_gen = ImageGenerator(service="nano_banana")

# Generate an image
result = image_gen.generate_image(
    prompt="A serene forest scene with sunlight filtering through trees",
    size="1024x1024",
    quality="standard"
)

print(f"Image generated: {result['success']}")
print(f"Image URL: {result['image_url']}")
```

### Pipeline Integration

```python
from text_to_image_pipeline import TextToImagePipeline

# Initialize pipeline with Nano Banana
pipeline = TextToImagePipeline(
    image_service="nano_banana",
    segmentation_method="semantic"
)

# Process text and generate images
results = pipeline.process_text(
    text="Your story text here...",
    style="realistic",
    save_images=True
)
```

## 🔧 Configuration Options

### Nano Banana Parameters

- **size**: Image dimensions (e.g., "1024x1024", "512x512")
- **quality**: Image quality ("standard", "hd")
- **style**: Visual style ("natural", "artistic", "cinematic")
- **n**: Number of images to generate (default: 1)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NANO_BANANA_API_KEY` | Your API key | Required |
| `NANO_BANANA_API_URL` | API endpoint | `https://api.nanobanana.ai/v1/images/generations` |

## 🚨 Troubleshooting

### Common Issues

1. **"API Key not found"**
   - Ensure your `.env` file exists in the project root
   - Check that `NANO_BANANA_API_KEY` is set correctly
   - Verify no extra spaces or quotes around the key

2. **"API URL not accessible"**
   - Check your internet connection
   - Verify the API endpoint is correct
   - Ensure your API key has the correct permissions

3. **"Mock image generated"**
   - This is normal when no API key is set
   - Set your API key to generate real images
   - Check the console for specific error messages

### Debug Mode

Enable debug logging by setting the environment variable:
```bash
export DEBUG=1
python your_script.py
```

## 📊 Performance Considerations

- **Response Time**: Nano Banana typically responds in 10-30 seconds
- **Rate Limits**: Check Google's current rate limits for your API tier
- **Image Quality**: Higher quality settings may take longer to generate
- **Batch Processing**: Consider processing multiple prompts sequentially

## 🔄 Fallback Services

If Nano Banana is unavailable, the pipeline can fall back to:

1. **Stable Diffusion**: Set `STABLE_DIFFUSION_API_KEY` in your `.env`
2. **Mock Images**: Generated automatically when no services are available

## 📚 API Reference

### ImageGenerator Methods

- `generate_image(prompt, **kwargs)`: Generate a single image
- `generate_image_variations(prompt, num_variations)`: Generate multiple variations
- `save_image_locally(image_url, filename)`: Download and save images

### Response Format

```python
{
    "success": True,
    "image_url": "https://...",
    "service": "Nano Banana",
    "prompt": "Original prompt",
    "revised_prompt": "API-revised prompt",
    "metadata": {
        "size": "1024x1024",
        "quality": "standard",
        "style": "natural",
        "model": "nano-banana-1.0"
    }
}
```

## 🆘 Support

If you encounter issues:

1. **Check the logs**: Look for error messages in the console
2. **Verify credentials**: Ensure your API key is correct and active
3. **Test connectivity**: Run `test_nano_banana.py` to diagnose issues
4. **Check quotas**: Verify your Google AI Studio account has available quota

## 🔮 Future Updates

The Nano Banana integration will be updated to support:
- Additional model variants
- More customization options
- Batch processing capabilities
- Advanced prompt engineering features

---

**Note**: This integration is based on the current Nano Banana API. Refer to [Google's official documentation](https://aistudio.google.com/) for the most up-to-date information.
