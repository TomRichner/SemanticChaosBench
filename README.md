# Semantic Chaos Bench
## Measuring Chaos and Stability in Large Language Models

### Project Overview
Semantic Chaos Bench measures how small perturbations in input prompts lead to diverging outputs in LLMs, analogous to Lyapunov exponents in dynamical systems. By creating semantically similar prompt pairs and tracking output divergence, we can characterize the stability/chaos regimes of different models.

**Architecture**: Hybrid local/cloud setup running on Mac:
- **Local**: Sentence-BERT embeddings (runs on Mac with MPS acceleration)
- **Cloud**: All LLM inference via APIs (OpenAI, Anthropic, Google, and open models via Replicate/Together AI)
- **Package Management**: Using `uv` for fast, reliable Python dependency management

### Core Concept
1. Generate pairs of prompts that differ by small semantic distance ε
2. Feed both prompts to various LLMs (open and closed source)
3. Measure semantic distance between outputs using Sentence-BERT
4. Track divergence rate across multiple generation steps
5. Compare divergence characteristics across models

---

## Technical Architecture

### 1. Prompt Perturbation Pipeline

#### Method A: Paraphrase + Filter
```python
def generate_perturbed_prompts(base_prompt, epsilon, n_pairs):
    """
    1. Generate 100+ paraphrases using LLM
    2. Embed all with Sentence-BERT
    3. Select pairs with distance ≈ epsilon
    4. Return (prompt1, prompt2, measured_distance)
    """
```

#### Method B: Temperature Variation
```python
def generate_temperature_variants(base_prompt, model, epsilon):
    """
    1. Use high temperature (1.5-2.0) to generate variations
    2. Embed with Sentence-BERT
    3. Filter for semantic distance ≈ epsilon
    4. Return filtered pairs
    """
```

### 2. Model Interface Layer

#### Unified API Wrapper
```python
class ModelInterface:
    """
    All models accessed via API endpoints:
    - OpenAI API (GPT-4, GPT-3.5-turbo)
    - Anthropic API (Claude 3.5 Sonnet, Claude 3 Opus)
    - Google Vertex AI (Gemini Pro)
    - Replicate API (Llama 3, Mistral, Mixtral)
    - Together AI (Open models at scale)
    
    No local LLM inference - keeps Mac resource usage low
    """
    
    def generate(prompt, temperature=0.7, max_tokens=500):
        # Returns: text, latency, token_count
        # Routes to appropriate API based on model name
```

### 3. Divergence Measurement

#### Single-Step Divergence
```python
def measure_single_divergence(prompt1, prompt2, model):
    """
    1. Generate output1 from prompt1
    2. Generate output2 from prompt2
    3. Embed both with Sentence-BERT
    4. Return cosine distance
    """
```

#### Multi-Step Divergence (Conversations)
```python
def measure_trajectory_divergence(prompt1, prompt2, model, n_steps):
    """
    1. Initialize conversation with perturbed prompts
    2. Generate n_steps of conversation
    3. Track divergence at each step
    4. Return divergence trajectory
    """
```

### 4. Analysis Framework

#### Metrics to Compute
- **Divergence Rate**: δ(t) = ||output1(t) - output2(t)|| / ||prompt1 - prompt2||
- **Lyapunov-like Exponent**: λ ≈ (1/t) * log(δ(t)/δ(0))
- **Saturation Distance**: Maximum divergence reached
- **Divergence Onset**: Steps until significant divergence
- **Stability Regions**: Map prompt types to divergence behavior

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
- [ ] a. Set up project with `uv` and dependencies
- [ ] b. Configure Sentence-BERT with MPS acceleration (local Mac)
- [ ] c. Implement prompt perturbation generator
- [ ] d. Create unified model API interface
- [ ] e. Build basic divergence measurement
- [ ] f. Set up API key management (.env file)

### Phase 2: Perturbation Generation (Week 2)
- [ ] a. Implement paraphrase generation using GPT-4/Claude API
- [ ] b. Build semantic distance filtering (local embeddings)
- [ ] c. Create prompt pair validation
- [ ] d. Generate test dataset of 100 prompt pairs at various ε levels

### Phase 3: Model Integration (Week 3)
- [ ] a. Integrate OpenAI API (GPT-5, with and without thinking)
- [ ] b. Integrate Anthropic API (Claude models)
- [ ] c. Integrate Gemini API
- [ ] d. Integrate Replicate API (Llama, Mistral, etc.)
- [ ] e. Integrate Together AI (alternative for open models)
- [ ] f. Implement rate limiting, retries, and error handling
- [ ] g. Add response caching to minimize repeated API calls

### Phase 4: Measurement Suite (Week 4)
- [ ] a. Build single-step divergence measurement
- [ ] b. Implement multi-step conversation tracking
- [ ] c. Create visualization tools
- [ ] d. Build statistical analysis pipeline

### Phase 5: Benchmarking (Week 5)
- [ ] a. Run systematic experiments across models
- [ ] b. Generate divergence profiles for each model
- [ ] c. Identify chaos/stability regimes
- [ ] d. Create comparative analysis

---

## Code Structure

```
semantic_chaos_bench/
├── src/
│   ├── __init__.py
│   ├── perturbation/
│   │   ├── __init__.py
│   │   ├── paraphrase_generator.py
│   │   ├── semantic_filter.py
│   │   └── prompt_pairs.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── openai_wrapper.py
│   │   ├── anthropic_wrapper.py
│   │   ├── google_wrapper.py
│   │   ├── replicate_wrapper.py
│   │   └── together_wrapper.py
│   ├── measurement/
│   │   ├── __init__.py
│   │   ├── embeddings.py          # Local Sentence-BERT (MPS accelerated)
│   │   ├── divergence.py
│   │   └── trajectories.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   └── statistics.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── cache.py
│       └── logging.py
├── scripts/
│   ├── pilot_study.py             # Initial validation experiment
│   ├── generate_prompt_pairs.py   # Batch prompt pair generation
│   ├── run_benchmark.py           # Full benchmark runner
│   └── analyze_results.py         # Post-processing and reporting
├── experiments/
│   ├── configs/
│   └── results/
├── data/
│   ├── prompt_pairs/
│   ├── outputs/
│   └── cache/                     # Local cache for API responses
├── notebooks/
│   ├── exploration.ipynb
│   └── analysis.ipynb
├── docs/
│   ├── architecture.md            # Detailed system design
│   ├── api_integration.md         # Guide for adding new model APIs
│   ├── metrics.md                 # Explanation of divergence metrics
│   └── usage.md                   # Detailed usage examples
├── tests/
│   └── __init__.py
├── config.yaml                    # Default settings (epsilon levels, models, etc.)
├── pyproject.toml                 # Using uv for dependency management
├── .env.example                   # API keys template
├── .gitignore
└── README.md
```

---

## Key Dependencies

**Package Manager**: Using `uv` for fast, reliable dependency management

```bash
# Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize project
uv init
uv venv
source .venv/bin/activate  # On Mac
```

### Core Dependencies (pyproject.toml)
```toml
[project]
name = "semantic-divergence-bench"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    # Local embeddings (runs on Mac with MPS)
    "sentence-transformers>=2.2.0",
    "torch>=2.0.0",
    
    # API clients (all models via cloud)
    "openai>=1.0.0",
    "anthropic>=0.18.0",
    "replicate>=0.20.0",
    "together>=0.2.0",
    
    # Optional: Only if using Gemini
    # "google-cloud-aiplatform>=1.38.0",
    
    # Data & analysis
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    
    # Visualization
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    
    # Utilities
    "tqdm>=4.65.0",
    "python-dotenv>=1.0.0",
    "hydra-core>=1.3.0",
    
    # Optional: Experiment tracking
    "wandb>=0.15.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
]
```

### Installation
```bash
# Install all dependencies with uv (much faster than pip)
uv pip install -e .

# Or install with dev dependencies
uv pip install -e ".[dev]"

# Verify MPS (Metal) is available on Mac
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Quick Start Setup (Mac)
```bash
# 1. Clone repository
git clone <repo-url>
cd semantic_divergence_bench

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 4. Set up API keys
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# REPLICATE_API_TOKEN=r8_...
# TOGETHER_API_KEY=...

# 5. Test setup
python -c "from sentence_transformers import SentenceTransformer; import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# 6. Run pilot experiment
python scripts/pilot_study.py
```

---

## Experimental Design

### Prompt Categories to Test
1. **Factual Questions**: "What is the capital of [country]?"
2. **Creative Writing**: "Write a story about [topic]"
3. **Reasoning Tasks**: "Explain why [phenomenon] occurs"
4. **Code Generation**: "Write a function to [task]"
5. **Conversational**: "How should I [situation]?"

### Perturbation Levels
- ε = 0.01 (minimal perturbation)
- ε = 0.05 (small perturbation)
- ε = 0.10 (moderate perturbation)
- ε = 0.20 (large perturbation)

### Models to Compare
- **Closed Source**: GPT-4, GPT-3.5, Claude 3.5, Claude 3, Gemini Pro
- **Open Source**: Llama 3 (various sizes), Mistral, Mixtral, Qwen

### Measurements per Configuration
- 100 prompt pairs per category
- 3 different temperatures (0.0, 0.7, 1.0)
- 5 generation steps for multi-turn analysis
- Total: ~15,000 API calls per model

---

## Expected Outputs

### 1. Divergence Profiles
- Heatmap: prompt_type × perturbation_level → divergence_rate
- Model comparison charts
- Chaos vs. stability region maps

### 2. Model Characteristics
- Which models are most sensitive to perturbations?
- Do larger models show more or less divergence?
- How does temperature affect stability?

### 3. Prompt-Specific Insights
- Which prompt types lead to chaotic behavior?
- Are factual queries more stable than creative ones?
- How does prompt complexity affect divergence?

### 4. Research Paper Components
- Quantitative divergence metrics
- Statistical significance tests
- Reproducible benchmark suite
- Public leaderboard potential

---

## Usage Example

```python
from semantic_divergence_bench import DivergenceBench

# Initialize benchmark
bench = DivergenceBench(
    models=['gpt-4', 'claude-3.5-sonnet', 'llama-3-70b'],
    epsilon_levels=[0.01, 0.05, 0.10],
    prompt_categories=['factual', 'creative', 'reasoning']
)

# Run experiments
results = bench.run_full_benchmark()

# Analyze
bench.plot_divergence_profiles()
bench.compute_stability_metrics()
bench.generate_report('results/divergence_analysis.html')
```

---

## Next Steps

1. **Environment Setup**: Install `uv`, create project structure, configure API keys
2. **Pilot Study**: Start with 10 prompt pairs on 2 models (e.g., GPT-4 + Llama 3) to validate approach
3. **Optimize Costs**: Implement aggressive caching and batch processing (critical for API costs!)
4. **Benchmark Closed Models**: Run full suite on OpenAI and Anthropic APIs
5. **Add Open Models**: Integrate Replicate/Together for Llama, Mistral comparisons
6. **Analysis & Visualization**: Build comprehensive divergence profiles
7. **Community Input**: Share methodology and early results for feedback
8. **Publication**: Prepare findings for academic paper or technical blog post

---

## Research Questions to Explore

1. Do models with more parameters exhibit different divergence patterns?
2. Is there a correlation between model performance benchmarks and stability?
3. Can we predict which prompts will cause chaotic behavior?
4. Do instruction-tuned models show different stability than base models?
5. How does divergence relate to model confidence/calibration?
6. Can we find adversarial prompts that maximize divergence?

---

## Notes and Considerations

### Architecture Decisions
- **Local Mac Setup**: Runs Sentence-BERT embeddings with MPS (Metal) acceleration
- **Cloud APIs**: All LLM inference via APIs - no local model hosting required
- **Package Management**: Using `uv` for faster dependency resolution and installs
- **Minimal Compute**: Mac handles orchestration, caching, analysis - not inference

### Operational
- **API Costs**: Budget ~$500-1000 for comprehensive benchmarking
  - Closed models (OpenAI, Anthropic): ~$300-800
  - Open models (Replicate/Together): ~$100-200
- **Rate Limits**: Implement proper throttling and retry logic per API
- **Reproducibility**: Set seeds where possible, log all parameters and model versions
- **Ethics**: Ensure prompts are appropriate and non-harmful
- **Caching**: Store all API responses locally to avoid repeated calls (critical for costs!)
- **Versioning**: Track model versions as they update over time
- **API Keys**: Use `.env` file for secure credential management (never commit!)

### Mac-Specific
- **MPS Acceleration**: PyTorch automatically uses Metal for Sentence-BERT (10x+ faster)
- **Memory**: Embedding models are small (~100-500MB), easily fits in RAM
- **Storage**: Cache embeddings and API responses locally (~5-10GB estimated)