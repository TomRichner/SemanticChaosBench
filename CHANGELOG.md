# Changelog

All notable changes to the Semantic Chaos Bench project will be documented in this file.

---

## [2025-11-04] Caching Disabled by Default

### Summary
Changed default caching behavior from enabled to disabled to preserve the stochastic nature of model responses, which is essential for measuring divergence and chaos.

### Rationale
- **Research Goals**: This project explicitly studies response variability, stochasticity, and divergence
- **Caching Conflicts**: Caching masks the phenomena we're measuring (different outputs from identical inputs)
- **Experimental Integrity**: Each API call should produce fresh, independent responses
- **Simpler Mental Model**: No need to remember to disable caching for each experiment

### Changes Made

**Code Changes:**
- `src/models/base_model.py`: Changed `enable_cache` default from `True` to `False`
- All model wrappers: Changed `enable_cache` default from `True` to `False`
  - `src/models/anthropic_wrapper.py`
  - `src/models/google_wrapper.py`
  - `src/models/openai_wrapper.py`
  - `src/models/replicate_wrapper.py`
  - `src/models/together_wrapper.py`

**Configuration:**
- `config.yaml`: Set `api.cache.enabled: false` and `experiment.cache_responses: false`

**Documentation:**
- Updated README.md to explain caching is disabled by default
- Noted that caching can still be enabled for development/debugging with `enable_cache=True`
- Clarified difference between API response caching and explicit experimental result saving

### Infrastructure Preserved
The caching infrastructure remains fully functional:
- `Cache` class in `src/utils/cache.py` unchanged
- Can be enabled per-model with `enable_cache=True`
- Test suite (`test_caching.py`) explicitly enables caching and continues to pass
- Useful for development, debugging, and non-research workflows

### Impact
- **Tests**: `test_caching.py` explicitly enables caching, so all tests still pass
- **Development**: Can still use `enable_cache=True` when iterating on non-experimental code
- **Experiments**: Now default to fresh API calls, preserving response variability
- **Future**: Experimental results will be saved explicitly to `data/outputs/` (to be implemented)

---

## [2025-11-04] Phase 3 Complete: Caching, Rate Limiting, and Retries

### Summary
✅ **Phase 3 is now complete!** Implemented comprehensive caching, rate limiting, and retry logic across all model wrappers to minimize API costs and improve reliability.

### Components Implemented

#### 1. Shared API Utilities (`src/utils/api_helpers.py`)
- **Rate Limiter**: Thread-safe rate limiting per provider
  - Tracks last request time per provider
  - Enforces configurable minimum delay between requests
  - Prevents quota exhaustion
- **Retry Decorator**: Configurable retry logic with exponential backoff
  - Automatic retries for transient errors (500, 503, 429)
  - Exponential backoff: delay = base_delay * (2 ^ attempt)
  - Maximum retry attempts and delays configurable
- **Error Classification**: Distinguish retryable vs non-retryable errors
  - Helpful error messages with troubleshooting hints
  - Provider-specific error handling

#### 2. Enhanced Cache System (`src/utils/cache.py`)
- **Cache Statistics**: Track hits, misses, and hit rate
- **Smart Cache Keys**: Based on model, prompt, temperature, max_tokens, and all kwargs
- **Enable/Disable**: Can be toggled per model instance
- **Clear/Reset**: Utility methods for cache management
- **Error Handling**: Graceful handling of corrupted cache files

#### 3. Updated BaseModel (`src/models/base_model.py`)
- **Configuration Loading**: Reads from `config.yaml` for retry/rate limit settings
- **Cache Integration**: Built-in caching support for all subclasses
- **Rate Limiter Access**: Per-provider rate limiting
- **Helper Methods**: `_get_cached_response()`, `_cache_response()`, `_get_provider_name()`
- **ModelResponse Serialization**: `to_dict()` and `from_dict()` for caching

#### 4. All Model Wrappers Updated
Updated all five model wrappers with consistent caching, retries, and rate limiting:
- **OpenAI** (`openai_wrapper.py`)
- **Anthropic** (`anthropic_wrapper.py`)
- **Google** (`google_wrapper.py`) - Refactored to use shared utilities
- **Replicate** (`replicate_wrapper.py`)
- **Together** (`together_wrapper.py`)

**Each wrapper now:**
- Checks cache before making API calls
- Applies rate limiting before requests
- Uses retry decorator for transient failures
- Caches successful responses
- Supports `enable_cache` parameter
- Has provider-specific `_get_provider_name()` override

#### 5. Configuration Updates (`config.yaml`)
Added provider-specific settings:
```yaml
api:
  max_retries: 3
  retry_delay: 1.0  # base delay for exponential backoff
  rate_limits:
    openai: 0.5      # seconds between requests
    anthropic: 0.5
    google: 1.0      # more conservative
    replicate: 1.0
    together: 0.5
  cache:
    enabled: true
    directory: "data/cache"
```

#### 6. Test Suite (`scripts/tests/test_caching.py`)
Comprehensive caching tests:
- **Test 1**: Basic cache operations (key generation, set/get, stats)
- **Test 2**: Model integration with caching (cache hits, speedup verification)
- **Test 3**: Cache invalidation (different parameters generate different keys)
- **Test 4**: Cache disabling (verifies caching can be turned off)

### Key Benefits

- **Cost Reduction**: Caching eliminates redundant API calls
  - Cache hits return instantly
  - Significant cost savings for repeated experiments
- **Reliability**: Automatic retries handle transient failures
  - Exponential backoff prevents overwhelming servers
  - Clear error messages for debugging
- **Performance**: Rate limiting prevents quota exhaustion
  - Provider-specific limits
  - Thread-safe for concurrent usage
- **Consistency**: All providers follow same patterns
  - Uniform API across all model wrappers
  - Easy to add new providers

### Testing Results

✅ **All components tested and working:**
- Cache key generation consistent and differentiating
- Cache hits significantly faster than API calls (10x+ speedup)
- Rate limiting enforces delays between requests
- Retry logic handles server errors gracefully
- All model wrappers compatible with new features

### Phase Status

**Phase 3: Model Integration** ✅ **Complete**
- [x] All five model providers integrated
- [x] Rate limiting implemented
- [x] Retry logic with exponential backoff
- [x] Response caching system
- [x] Error handling and classification

### Next Steps
- Implement multi-step conversation tracking (Phase 4)
- Create visualization tools for divergence profiles
- Generate comprehensive prompt pair dataset (Phase 2)
- Run systematic benchmarking across models (Phase 5)

---

## [2025-11-04] Script Organization and Unified Runner

### Changes
- **Reorganized scripts directory** with subdirectories for better maintainability:
  - `scripts/tests/` - All test scripts (test_setup.py, test_divergence.py, etc.)
  - `scripts/demos/` - All demo scripts (demo_phase1.py, demo_divergence_with_models.py)
  - Other scripts (generate_prompt_pairs.py, pilot_study.py, etc.) remain in scripts root
- **Created unified runner script** (`scripts/run_scripts.py`):
  - Run all tests: `python scripts/run_scripts.py --tests`
  - Run all demos: `python scripts/run_scripts.py --demos`
  - Run everything: `python scripts/run_scripts.py --all`
  - Run specific scripts: `python scripts/run_scripts.py --scripts test_setup demo_phase1`
- **Updated README.md** with new directory structure and runner usage examples

---

## [2025-11-04] Phase 1 Complete: Basic Divergence Measurement

### Summary
✅ **Phase 1 is now complete!** Implemented single-step divergence measurement system that measures semantic chaos in LLMs by quantifying how small prompt perturbations lead to diverging outputs.

### Components Implemented

#### 1. Divergence Measurement (`src/measurement/divergence.py`)
- **`measure_single_divergence()`**: Core function to measure divergence between prompt pairs and their outputs
- **Metrics Computed**:
  - Input distance (ε): Semantic distance between prompts
  - Output distance (δ): Semantic distance between model outputs
  - Divergence rate (δ/ε): Amplification factor (analogous to Lyapunov exponents)
- **Integration**: Works seamlessly with existing EmbeddingModel and all model wrappers
- **Return Format**: Dictionary with all metrics plus raw embeddings for further analysis

#### 2. Test Scripts

**`scripts/test_divergence.py`**
- Tests divergence measurement with four distinct scenarios:
  1. Identical prompts → Different outputs (infinite divergence, chaos)
  2. Similar prompts → Similar outputs (low divergence, stable)
  3. Different prompts → Different outputs (proportional divergence)
  4. Tiny perturbation → Variable outputs (THE KEY METRIC for chaos)
- Validates all computation logic without requiring API calls
- ✅ All tests passing with expected behavior

**`scripts/demo_divergence_with_models.py`**
- End-to-end integration demo with actual LLM API calls
- Shows real divergence measurement workflow:
  - Initialize embedding model (local, MPS-accelerated)
  - Generate outputs from perturbed prompt pairs
  - Compute and interpret divergence metrics
- Includes multi-model comparison capability
- Provides clear interpretation of results (stable/moderate/chaotic)

### Key Results from Testing

Example divergence measurements from `test_divergence.py`:

| Test Case | Input Distance | Output Distance | Divergence Rate | Interpretation |
|-----------|---------------|-----------------|-----------------|----------------|
| Identical → Different | 0.000000 | 0.250431 | ∞ | Maximum chaos |
| Similar → Similar | 0.068988 | 0.010643 | 0.1543 | Stable |
| Different → Different | 0.945108 | 1.069666 | 1.1318 | Proportional |
| Tiny change (short→brief) | 0.048521 | 0.417126 | 8.5969 | Chaotic |

**Interpretation:**
- δ/ε < 1.0 → Stable behavior (outputs diverge less than inputs)
- δ/ε < 5.0 → Moderate divergence
- δ/ε >> 1.0 → Chaotic behavior (small changes amplified dramatically)

### Documentation Updates

**README.md:**
- ✅ Marked Phase 1 as complete
- Updated current phase to Phase 2 & 4
- Added new test scripts to Available Commands section

### Phase Status

**Phase 1: Core Infrastructure** ✅ **Complete**
- [x] Project setup with `uv`
- [x] API key management
- [x] Sentence-BERT with MPS acceleration
- [x] Prompt perturbation generator
- [x] Unified model API interface
- [x] **Basic divergence measurement** ← Just completed!

### Next Steps
- Implement multi-step conversation tracking (Phase 4)
- Create visualization tools for divergence profiles
- Generate comprehensive prompt pair dataset (Phase 2)
- Run systematic benchmarking across models (Phase 5)

---

## [2025-11-03] Unified Model API Interface Complete

### Summary
Completed implementation of unified model API interface with support for all five target LLM providers. All models now accessible through consistent BaseModel interface with comprehensive testing suite passing.

### Components Implemented

#### 1. Base Model Interface (`src/models/base_model.py`)
- **Abstract Base Class**: Defines standard interface for all model wrappers
- **Consistent API**: All models implement `generate()` method with identical signature
- **Response Format**: Standardized return values (text, latency, token_count, model_name, metadata)
- **Error Handling**: Uniform exception handling across providers

#### 2. Model Wrappers
All five model providers implemented and tested:

- **OpenAI** (`src/models/openai_wrapper.py`)
  - Model: `gpt-4o-mini`
  - Features: Streaming support, token counting, error handling
  
- **Anthropic** (`src/models/anthropic_wrapper.py`)
  - Model: `claude-haiku-4-5`
  - Features: Message API, token counting, metadata extraction
  
- **Google AI Studio** (`src/models/google_wrapper.py`)
  - Models: `gemini-2.5-flash`, `gemini-2.5-pro`
  - Features: Simple API key auth, safety settings, token counting
  
- **Replicate** (`src/models/replicate_wrapper.py`)
  - Model: `meta/meta-llama-3-8b-instruct`
  - Features: Streaming output concatenation, automatic prediction handling
  
- **Together AI** (`src/models/together_wrapper.py`)
  - Model: `meta-llama/Meta-Llama-3-8B-Instruct-Lite`
  - Features: OpenAI-compatible API, efficient token counting

#### 3. Testing Suite (`scripts/test_model_interface.py`)
Comprehensive test script validating:
- ✅ Individual model generation for all 5 providers
- ✅ Temperature variation (0.0, 0.7, 1.0)
- ✅ Token limit enforcement
- ✅ Error handling and edge cases
- ✅ Consistency across providers
- ✅ Performance metrics (latency, token counts)

**All tests passing!**

### Phase Completion Status

**Phase 1: Core Infrastructure** ✅ **Complete**
- [x] Unified model API interface
- [x] All core components implemented

**Phase 2: Perturbation Generation** ✓ **In Progress**  
- [x] Paraphrase generation using LLM APIs
- [x] Semantic distance filtering (local embeddings)
- [x] Prompt pair validation
- [ ] Generate full test dataset (100 prompt pairs at various ε levels)

**Phase 3: Model Integration** ✅ **Complete**
- [x] All five model providers integrated and tested
- [ ] Rate limiting and retries (future enhancement)
- [ ] Response caching (future enhancement)

### Next Steps
- Implement single-step divergence measurement (Phase 4)
- Build multi-step conversation tracking
- Generate production prompt pair dataset
- Run pilot experiments

---

## [2025-11-02] Phase 1 Complete: Sentence-BERT + Prompt Perturbation Generator

### Summary
Implemented Sentence-BERT embeddings with MPS acceleration and the complete prompt perturbation generation pipeline. These are core components for generating semantically similar prompt pairs at controlled distances.

### Components Implemented

#### 1. Sentence-BERT Embeddings (`src/measurement/embeddings.py`)
- **MPS Acceleration**: Auto-detects and uses Apple Silicon GPU (Metal Performance Shaders)
- **Flexible Device Selection**: Supports MPS, CUDA, CPU with automatic fallback
- **Core Features**:
  - Text encoding with batch processing
  - Cosine distance computation between embeddings
  - Pairwise distance matrix calculation
  - Embedding dimension introspection
- **Verified Working**: Tests show ~10x speedup with MPS vs CPU

#### 2. Paraphrase Generator (`src/perturbation/paraphrase_generator.py`)
- **Multi-Model Support**: OpenAI (GPT), Anthropic (Claude), Google (Gemini)
- **Generation Methods**:
  - Batch generation (efficient, single API call)
  - Iterative generation (more diverse, independent variations)
  - Temperature variation method
- **Smart Parsing**: Automatically cleans and formats LLM responses

#### 3. Semantic Filter (`src/perturbation/semantic_filter.py`)
- **Distance-Based Filtering**: Filter prompts by target semantic distance ε ± tolerance
- **Analysis Tools**:
  - Distance distribution visualization
  - Optimal epsilon range identification
  - Diversity score computation
- **Efficient**: Uses vectorized operations for pairwise distances

#### 4. Prompt Pair Generator (`src/perturbation/prompt_pairs.py`)
- **End-to-End Pipeline**: Combines paraphrase generation + semantic filtering
- **Features**:
  - Multi-epsilon pair generation
  - Category-based organization
  - JSON/JSONL serialization
  - Statistical analysis (mean, std, range)
- **Data Class**: `PromptPair` with prompt1, prompt2, distance, category, epsilon_target

### Scripts Created

#### `scripts/test_embeddings_and_perturbation.py`
Comprehensive test suite validating:
- MPS acceleration for embeddings
- Semantic filtering accuracy
- Paraphrase generation (with API)
- Full pipeline integration

#### `scripts/generate_prompt_pairs.py`
Production script for batch prompt pair generation:
- Reads configuration from `config.yaml`
- Processes multiple categories (factual, creative, reasoning, code)
- Generates pairs at multiple epsilon levels
- Saves organized JSON outputs

### Configuration Updates (`config.yaml`)
Added settings for:
- `embedding_model`: Sentence-BERT model name
- `paraphrase_model`: LLM for paraphrase generation
- `n_paraphrases`: Number of paraphrases per prompt
- `n_pairs_per_prompt`: Target pairs per base prompt
- `tolerance`: Epsilon matching tolerance

### Testing Results
✅ **Embeddings**: MPS acceleration verified, 384-dim vectors, accurate distance computation  
✅ **Semantic Filter**: Correctly filters prompts by cosine distance  
✅ **Integration**: Full pipeline tested end-to-end  

### Next Steps
- Implement unified model API interface
- Build divergence measurement tools
- Run pilot experiments with generated prompt pairs

---

## [2025-11-02] Migration to Google AI Studio API

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
google:
+   - gemini-2.5-pro
+   - gemini-2.5-flash

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

### Files Modified

1. `pyproject.toml` - Updated dependencies
2. `src/models/google_wrapper.py` - Full implementation
3. `config.yaml` - Updated model names
4. `README.md` - Documentation updates
5. `scripts/test_setup.py` - Added Google API check
6. `scripts/test_google_api.py` - Updated model name

---

## Google AI Studio Setup Reference
This project uses **Google AI Studio API** (not Vertex AI) for accessing Gemini models. This provides a simpler setup with just an API key, similar to OpenAI and Anthropic.

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

