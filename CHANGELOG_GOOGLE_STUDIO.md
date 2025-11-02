# Changelog: Migration to Google AI Studio API

## Summary
Updated the project to use **Google AI Studio API** instead of Vertex AI for accessing Gemini models. This simplifies setup and aligns with the architecture pattern used for other API providers.

## Date
November 2, 2024

## Changes Made

### 1. Dependencies (`pyproject.toml`)
**Changed:**
```diff
- # "google-cloud-aiplatform>=1.38.0",  # Commented out
+ "google-generativeai>=0.3.0",  # Added
```

**Installed:**
- `google-generativeai==0.8.5`
- Related dependencies: `google-api-core`, `google-auth`, `grpcio`, etc.

### 2. Google Wrapper Implementation (`src/models/google_wrapper.py`)
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

### 3. Configuration (`config.yaml`)
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

### 4. Documentation Updates

#### `README.md`
- Updated API provider list: "Google AI Studio" instead of "Google Vertex AI"
- Updated dependency in example `pyproject.toml`
- Added `GOOGLE_API_KEY` to .env setup instructions

#### `QUICKSTART.md`
- Added `GOOGLE_API_KEY` to API key setup section
- Updated package list to include `google-generativeai`
- Clarified "AI Studio, not Vertex AI" in comments

#### `scripts/test_setup.py`
- Added "Google AI Studio" to API key checks
- Tests for `GOOGLE_API_KEY` environment variable

### 5. New Files Created

#### `GOOGLE_AI_STUDIO_SETUP.md`
Comprehensive setup guide including:
- Why Google AI Studio vs Vertex AI
- Step-by-step setup instructions
- Available models and recommendations
- Usage examples
- API limits and pricing
- Troubleshooting guide
- Migration instructions from Vertex AI

#### `scripts/test_google_api.py`
Standalone test script to verify Google AI Studio integration:
- Tests model initialization
- Tests text generation
- Reports latency and token usage
- Provides troubleshooting tips

### 6. Environment Variables

#### `.env.example` (already updated by user)
```bash
GOOGLE_API_KEY=  # From ai.google.dev
```

**Removed/Not needed:**
```bash
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
# GOOGLE_CLOUD_PROJECT=your-project-id
```

## Testing Results

### Setup Test
```bash
$ python scripts/test_setup.py
✓ OpenAI: Configured
✓ Anthropic: Configured
✓ Google AI Studio: Configured  # NEW
✓ Replicate: Configured
✓ Together: Configured
✓ All critical tests passed!
```

### Google API Test
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

## Breaking Changes

### For Existing Users
If you previously configured Vertex AI (unlikely since this was just set up):

1. **Remove** from `.env`:
   - `GOOGLE_APPLICATION_CREDENTIALS`
   - `GOOGLE_CLOUD_PROJECT`

2. **Add** to `.env`:
   - `GOOGLE_API_KEY=your-key-from-ai.google.dev`

3. **Reinstall** dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```

### Model Name Changes
- Old: `gemini-pro`, `gemini-1.5-pro` (deprecated)
- New: `gemini-2.5-pro`, `gemini-2.5-flash`, etc.

## Benefits of This Change

1. **Simpler Setup**: Just an API key, no GCP project needed
2. **Consistent Architecture**: Same pattern as OpenAI, Anthropic, etc.
3. **Faster Onboarding**: Get started in minutes vs hours
4. **Lower Complexity**: No service account management
5. **Free Tier**: Generous free tier for testing
6. **Latest Models**: Access to Gemini 2.5 Pro and Flash

## Files Modified

1. `pyproject.toml` - Updated dependencies
2. `src/models/google_wrapper.py` - Full implementation
3. `config.yaml` - Updated model names
4. `README.md` - Documentation updates
5. `QUICKSTART.md` - Documentation updates
6. `scripts/test_setup.py` - Added Google API check
7. `scripts/test_google_api.py` - Updated model name

## Files Created

1. `GOOGLE_AI_STUDIO_SETUP.md` - Comprehensive setup guide
2. `CHANGELOG_GOOGLE_STUDIO.md` - This file

## Next Steps

✅ Google AI Studio is now fully integrated and tested
✅ All API keys configured (OpenAI, Anthropic, Google, Replicate, Together, WandB)
✅ Ready to continue with Phase 1 implementation

Continue with remaining Phase 1 tasks:
- [ ] 1b. Configure Sentence-BERT with MPS acceleration
- [ ] 1c. Implement prompt perturbation generator
- [ ] 1d. Create unified model API interface
- [ ] 1e. Build basic divergence measurement

---

**Status**: ✅ Complete and Tested
**Author**: Automated setup
**Date**: November 2, 2024

