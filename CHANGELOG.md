# Changelog

All notable changes to the Semantic Chaos Bench project will be documented in this file.

---

## [2024-11-02] Migration to Google AI Studio API

### Summary
Updated the project to use **Google AI Studio API** instead of Vertex AI for accessing Gemini models. This simplifies setup and aligns with the architecture pattern used for other API providers.

### Changes Made

#### 1. Dependencies (`pyproject.toml`)
**Changed:**
```diff
- # "google-cloud-aiplatform>=1.38.0",  # Commented out
+ "google-generativeai>=0.3.0",  # Added
```

**Installed:**
- `google-generativeai==0.8.5`
- Related dependencies: `google-api-core`, `google-auth`, `grpcio`, etc.

#### 2. Google Wrapper Implementation (`src/models/google_wrapper.py`)
**Before:** Stub with TODO comments

**After:** Fully implemented wrapper featuring:
- Simple API key authentication via `GOOGLE_API_KEY` environment variable
- Support for all Gemini models (2.5 Pro, 2.5 Flash, 2.0 Flash, etc.)
- Proper error handling
- Token counting and metadata extraction
- Latency tracking
- Compatible with BaseModel interface

**Key Features:**
```python
# Simple initialization
model = GoogleModel(model_name="gemini-2.5-flash")

# Generation with response metadata
response = model.generate(prompt, temperature=0.7, max_tokens=500)
# Returns: text, latency, token_count, model_name, metadata
```

#### 3. Configuration (`config.yaml`)
**Changed:**
```diff
google:
-   - gemini-pro
-   - gemini-1.5-pro
-   - gemini-1.5-flash
+   - gemini-2.5-pro
+   - gemini-2.5-flash
+   - gemini-2.0-flash
+   - gemini-pro-latest  # Alias for latest pro
+   - gemini-flash-latest  # Alias for latest flash
```

#### 4. Documentation Updates

**`README.md`**
- Updated API provider list: "Google AI Studio" instead of "Google Vertex AI"
- Updated dependency in example `pyproject.toml`
- Added `GOOGLE_API_KEY` to .env setup instructions

**Environment Variables (`.env.example`)**
```bash
GOOGLE_API_KEY=  # From ai.google.dev
```

**Removed/Not needed:**
```bash
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
# GOOGLE_CLOUD_PROJECT=your-project-id
```

#### 5. Test Scripts
- `scripts/test_setup.py` - Added "Google AI Studio" to API key checks
- `scripts/test_google_api.py` - Created standalone test script for Google AI Studio

### Testing Results

**Setup Test:**
```bash
$ python scripts/test_setup.py
✓ OpenAI: Configured
✓ Anthropic: Configured
✓ Google AI Studio: Configured
✓ Replicate: Configured
✓ Together: Configured
✓ All critical tests passed!
```

**Google API Test:**
```bash
$ python scripts/test_google_api.py
✓ Model initialized successfully
✓ Generation successful!
Response: Hello, I am Gemini.
Latency: 1.193s
Tokens: 78
Model: gemini-2.5-flash
✓ Google AI Studio integration working correctly!
```

### Benefits

1. **Simpler Setup**: Just an API key, no GCP project needed
2. **Consistent Architecture**: Same pattern as OpenAI, Anthropic, etc.
3. **Faster Onboarding**: Get started in minutes vs hours
4. **Lower Complexity**: No service account management
5. **Free Tier**: Generous free tier for testing
6. **Latest Models**: Access to Gemini 2.5 Pro and Flash

### Files Modified

1. `pyproject.toml` - Updated dependencies
2. `src/models/google_wrapper.py` - Full implementation
3. `config.yaml` - Updated model names
4. `README.md` - Documentation updates
5. `scripts/test_setup.py` - Added Google API check
6. `scripts/test_google_api.py` - Updated model name

---

## Google AI Studio Setup Reference

### Overview

This project uses **Google AI Studio API** (not Vertex AI) for accessing Gemini models. This provides a simpler setup with just an API key, similar to OpenAI and Anthropic.

### Why Google AI Studio vs Vertex AI?

#### Google AI Studio ✅ (What we use)
- **Simple API key authentication**
- **No GCP project required**
- **Consistent with other APIs** in this project
- **Free tier available** for testing
- **Same models as Vertex AI**

#### Vertex AI ❌ (Not used)
- Requires GCP project setup
- Complex service account authentication
- Designed for enterprise/production
- Unnecessary overhead for benchmarking

### Setup Instructions

#### 1. Get Your API Key

1. Go to [Google AI Studio](https://ai.google.dev)
2. Click "Get API key"
3. Create a new API key (or use existing)
4. Copy the key

#### 2. Add to .env File

```bash
# In your .env file
GOOGLE_API_KEY=your-actual-api-key-here
```

#### 3. Available Models (as of Nov 2024)

**Recommended for Benchmarking:**
- `gemini-2.5-pro` - Latest flagship model (best quality)
- `gemini-2.5-flash` - Fast, cost-effective
- `gemini-2.0-flash` - Previous generation, very fast

**Stable Aliases:**
- `gemini-pro-latest` - Always points to latest Pro model
- `gemini-flash-latest` - Always points to latest Flash model

**Experimental Models:**
- `gemini-2.0-flash-thinking-exp` - With chain-of-thought
- Various preview and experimental versions

See all available models by running:
```bash
python -c "import google.generativeai as genai; import os; genai.configure(api_key=os.getenv('GOOGLE_API_KEY')); [print(m.name) for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]"
```

### Usage in Code

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

### API Limits & Pricing

#### Free Tier (as of 2024)
- **Gemini 2.5 Flash**: 15 RPM (requests per minute)
- **Gemini 2.5 Pro**: 2 RPM (requests per minute)
- Very generous for testing and small benchmarks

#### Paid Tier
- Much higher rate limits
- Pay-as-you-go pricing
- See [Google AI Studio pricing](https://ai.google.dev/pricing)

### Troubleshooting

**"Model not found" Error**
- **Problem**: Old model name like `gemini-pro` (without version)
- **Solution**: Use versioned names like `gemini-2.5-pro` or `gemini-pro-latest`

**"API key invalid" Error**
- **Problem**: API key not set or incorrect
- **Solution**: 
  1. Check `.env` file has `GOOGLE_API_KEY=...`
  2. Verify key from [ai.google.dev](https://ai.google.dev)
  3. Run `python scripts/test_setup.py` to verify

**Rate Limit Errors**
- **Problem**: Hitting free tier limits (2-15 RPM)
- **Solution**: 
  1. Add delays between requests
  2. Use `rate_limit_delay` in `config.yaml`
  3. Consider upgrading to paid tier

**Import Error: `google.generativeai`**
- **Problem**: Package not installed
- **Solution**: Run `uv pip install google-generativeai`

### Model Selection for Benchmarking

**For Quality Analysis:**
- Use **gemini-2.5-pro** for best quality
- Compare against GPT-4, Claude 3.5 Sonnet

**For Cost/Speed:**
- Use **gemini-2.5-flash** for faster, cheaper experiments
- ~10x faster than Pro version

**For Experimental Features:**
- Try **gemini-2.0-flash-thinking-exp** for chain-of-thought reasoning
- Good for comparing reasoning capabilities

### Additional Resources

- [Google AI Studio](https://ai.google.dev) - Get API keys
- [Gemini API Docs](https://ai.google.dev/docs) - Official documentation
- [Model Catalog](https://ai.google.dev/models/gemini) - Model details
- [Pricing](https://ai.google.dev/pricing) - Cost information

---

## Initial Project Setup

### Installation Complete ✓

**Phase 1 Progress:**
- [x] **1a. Set up project with uv and dependencies** - COMPLETED
- [x] **1f. Set up API key management (.env file)** - COMPLETED
- [ ] 1b. Configure Sentence-BERT with MPS acceleration (local Mac)
- [ ] 1c. Implement prompt perturbation generator
- [ ] 1d. Create unified model API interface
- [ ] 1e. Build basic divergence measurement

**What's Been Done:**
- ✓ Directory structure created  
- ✓ Virtual environment initialized with `uv`  
- ✓ All dependencies installed (170 packages)  
- ✓ PyTorch 2.9.0 with MPS (Metal) acceleration configured  
- ✓ Sentence-BERT tested and working on MPS  
- ✓ Project structure validated

**Installed Packages (Key Dependencies):**
- `torch==2.9.0` - PyTorch with MPS support
- `sentence-transformers==5.1.2` - Sentence-BERT embeddings
- `openai==2.6.1` - OpenAI API client
- `anthropic==0.72.0` - Anthropic API client
- `google-generativeai` - Google AI Studio client (Gemini)
- `replicate==1.0.7` - Replicate API client
- `together==1.5.29` - Together AI API client
- `numpy`, `pandas`, `scipy` - Data processing
- `matplotlib`, `seaborn` - Visualization
- `pytest`, `jupyter`, `black`, `ruff` - Development tools
- `wandb` - Experiment tracking (optional)

