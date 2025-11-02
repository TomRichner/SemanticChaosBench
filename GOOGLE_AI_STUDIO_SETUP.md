# Google AI Studio Setup Guide

## Overview

This project uses **Google AI Studio API** (not Vertex AI) for accessing Gemini models. This provides a simpler setup with just an API key, similar to OpenAI and Anthropic.

## Why Google AI Studio vs Vertex AI?

### Google AI Studio ✅ (What we use)
- **Simple API key authentication**
- **No GCP project required**
- **Consistent with other APIs** in this project
- **Free tier available** for testing
- **Same models as Vertex AI**

### Vertex AI ❌ (Not used)
- Requires GCP project setup
- Complex service account authentication
- Designed for enterprise/production
- Unnecessary overhead for benchmarking

## Setup Instructions

### 1. Get Your API Key

1. Go to [Google AI Studio](https://ai.google.dev)
2. Click "Get API key"
3. Create a new API key (or use existing)
4. Copy the key

### 2. Add to .env File

```bash
# In your .env file
GOOGLE_API_KEY=your-actual-api-key-here
```

### 3. Available Models (as of Nov 2024)

#### Recommended for Benchmarking:
- `gemini-2.5-pro` - Latest flagship model (best quality)
- `gemini-2.5-flash` - Fast, cost-effective
- `gemini-2.0-flash` - Previous generation, very fast

#### Stable Aliases:
- `gemini-pro-latest` - Always points to latest Pro model
- `gemini-flash-latest` - Always points to latest Flash model

#### Experimental Models:
- `gemini-2.0-flash-thinking-exp` - With chain-of-thought
- Various preview and experimental versions

See all available models by running:
```bash
python -c "import google.generativeai as genai; import os; genai.configure(api_key=os.getenv('GOOGLE_API_KEY')); [print(m.name) for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]"
```

## Usage in Code

### Basic Usage

```python
from src.models.google_wrapper import GoogleModel
from dotenv import load_dotenv

load_dotenv()

# Initialize model
model = GoogleModel(model_name="gemini-2.5-flash")

# Generate text
response = model.generate(
    prompt="Explain quantum computing in one sentence.",
    temperature=0.7,
    max_tokens=100
)

print(response.text)
print(f"Latency: {response.latency:.2f}s")
print(f"Tokens: {response.token_count}")
```

### In Benchmark Config

The `config.yaml` includes recommended Gemini models:

```yaml
models:
  google:
    - gemini-2.5-pro
    - gemini-2.5-flash
    - gemini-2.0-flash
    - gemini-pro-latest
    - gemini-flash-latest
```

## Testing Your Setup

Run the test script to verify everything works:

```bash
source .venv/bin/activate
python scripts/test_google_api.py
```

Or run the full setup test:

```bash
python scripts/test_setup.py
```

## API Limits & Pricing

### Free Tier (as of 2024)
- **Gemini 2.5 Flash**: 15 RPM (requests per minute)
- **Gemini 2.5 Pro**: 2 RPM (requests per minute)
- Very generous for testing and small benchmarks

### Paid Tier
- Much higher rate limits
- Pay-as-you-go pricing
- See [Google AI Studio pricing](https://ai.google.dev/pricing)

## Troubleshooting

### "Model not found" Error
- **Problem**: Old model name like `gemini-pro` (without version)
- **Solution**: Use versioned names like `gemini-2.5-pro` or `gemini-pro-latest`

### "API key invalid" Error
- **Problem**: API key not set or incorrect
- **Solution**: 
  1. Check `.env` file has `GOOGLE_API_KEY=...`
  2. Verify key from [ai.google.dev](https://ai.google.dev)
  3. Run `python scripts/test_setup.py` to verify

### Rate Limit Errors
- **Problem**: Hitting free tier limits (2-15 RPM)
- **Solution**: 
  1. Add delays between requests
  2. Use `rate_limit_delay` in `config.yaml`
  3. Consider upgrading to paid tier

### Import Error: `google.generativeai`
- **Problem**: Package not installed
- **Solution**: Run `uv pip install google-generativeai`

## Model Selection for Benchmarking

### For Quality Analysis
- Use **gemini-2.5-pro** for best quality
- Compare against GPT-4, Claude 3.5 Sonnet

### For Cost/Speed
- Use **gemini-2.5-flash** for faster, cheaper experiments
- ~10x faster than Pro version

### For Experimental Features
- Try **gemini-2.0-flash-thinking-exp** for chain-of-thought reasoning
- Good for comparing reasoning capabilities

## Migration from Vertex AI

If you previously had Vertex AI configured:

1. **Remove** these from `.env`:
   - `GOOGLE_APPLICATION_CREDENTIALS`
   - `GOOGLE_CLOUD_PROJECT`

2. **Add** this to `.env`:
   - `GOOGLE_API_KEY=your-key`

3. **Update** dependency in `pyproject.toml`:
   - Remove: `google-cloud-aiplatform`
   - Add: `google-generativeai>=0.3.0`

4. **Reinstall**:
   ```bash
   uv pip install -e .
   ```

## Additional Resources

- [Google AI Studio](https://ai.google.dev) - Get API keys
- [Gemini API Docs](https://ai.google.dev/docs) - Official documentation
- [Model Catalog](https://ai.google.dev/models/gemini) - Model details
- [Pricing](https://ai.google.dev/pricing) - Cost information

---

**Ready to benchmark?** Your Google AI Studio setup is complete! ✓

