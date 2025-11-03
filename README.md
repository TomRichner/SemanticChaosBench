# Semantic Chaos Bench
## Measuring Chaos and Stability in Large Language Models

Semantic Chaos Bench measures how small perturbations in input prompts lead to diverging outputs in LLMs, analogous to Lyapunov exponents in dynamical systems. By creating semantically similar prompt pairs and tracking output divergence, we characterize the stability/chaos regimes of different models.

**How it works:**
1. Generate prompt pairs differing by small semantic distance ε
2. Feed both prompts to various LLMs via APIs
3. Measure semantic distance between outputs using Sentence-BERT (local, MPS-accelerated)
4. Track divergence rate across multiple generation steps
5. Compare divergence characteristics across models

**Architecture:**
- **Local (Mac)**: Sentence-BERT embeddings with MPS acceleration, caching, orchestration
- **Cloud**: All LLM inference via APIs (OpenAI, Anthropic, Google, Replicate, Together AI)
- **Package Management**: `uv` for fast dependency management

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
    - OpenAI API (gpt-4o-mini)
    - Anthropic API (claude-haiku-4-5)
    - Google AI Studio (gemini-2.5-flash)
    - Replicate API (meta/meta-llama-3-8b-instruct)
    - Together AI (meta-llama/Meta-Llama-3-8B-Instruct-Lite)
    
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

**Current Phase: Phase 1 - Core Infrastructure**

### Phase 1: Core Infrastructure ✓ In Progress
- [x] Set up project with `uv` and dependencies
- [x] Set up API key management (.env file)
- [x] Google AI Studio integration
- [x] Configure Sentence-BERT with MPS acceleration (local Mac)
- [x] Implement prompt perturbation generator 
- [ ] Create unified model API interface
- [ ] Build basic divergence measurement

### Phase 2: Perturbation Generation
- [ ] Implement paraphrase generation using LLM APIs
- [ ] Build semantic distance filtering (local embeddings)
- [ ] Create prompt pair validation
- [ ] Generate test dataset of 100 prompt pairs at various ε levels

### Phase 3: Model Integration
- [ ] Integrate OpenAI API (gpt-4o-mini)
- [ ] Integrate Anthropic API (claude-haiku-4-5)
- [ ] Integrate Google AI Studio (gemini-2.5-flash)
- [ ] Integrate Replicate API (meta/meta-llama-3-8b-instruct)
- [ ] Integrate Together AI (meta-llama/Meta-Llama-3-8B-Instruct-Lite)
- [ ] Implement rate limiting, retries, and error handling
- [ ] Add response caching to minimize repeated API calls

### Phase 4: Measurement Suite
- [ ] Build single-step divergence measurement
- [ ] Implement multi-step conversation tracking
- [ ] Create visualization tools
- [ ] Build statistical analysis pipeline

### Phase 5: Benchmarking
- [ ] Run systematic experiments across models
- [ ] Generate divergence profiles for each model
- [ ] Identify chaos/stability regimes
- [ ] Create comparative analysis

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

## Installation & Setup

### Prerequisites
- macOS with Apple Silicon (for MPS acceleration)
- Python 3.10 or higher
- API keys for model providers

### Quick Start

```bash
# 1. Clone and navigate to repository
git clone <repo-url>
cd semantic_chaos_bench

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
# GOOGLE_API_KEY=...  # From ai.google.dev
# REPLICATE_API_TOKEN=r8_...
# TOGETHER_API_KEY=...

# 5. Test setup
python scripts/test_setup.py

# 6. Run pilot experiment (once Phase 1 is complete)
python scripts/pilot_study.py
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
    "google-generativeai>=0.3.0",  # Google AI Studio (not Vertex AI)
    
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

### Available Commands

```bash
# Test setup (verifies MPS, API keys, dependencies)
python scripts/test_setup.py

# Generate prompt pairs
python scripts/generate_prompt_pairs.py

# Run full benchmark
python scripts/run_benchmark.py

# Analyze results
python scripts/analyze_results.py
```

### Troubleshooting

```bash
# Verify MPS acceleration
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Ensure you're in the virtual environment
source .venv/bin/activate

# Reinstall dependencies if needed
uv pip install -e ".[dev]"
```

---

## Experimental Design

**Prompt Categories**: Factual questions, creative writing, reasoning tasks, code generation, conversational

**Perturbation Levels**: ε = 0.01, 0.05, 0.10, 0.20 (minimal to large)

**Models**: 
- gpt-4o-mini (OpenAI)
- claude-haiku-4-5 (Anthropic)
- gemini-2.5-flash (Google)
- meta/meta-llama-3-8b-instruct (Replicate)
- meta-llama/Meta-Llama-3-8B-Instruct-Lite (Together AI)

**Measurements**: 100 prompt pairs per category × 3 temperatures (0.0, 0.7, 1.0) × 5 generation steps = ~15,000 API calls per model

**Expected Outputs**:
- Divergence profiles and heatmaps (prompt type × perturbation level)
- Model sensitivity rankings and stability comparisons
- Chaos vs. stability region identification
- Statistical analysis and reproducible benchmark suite

---

## Usage Example

```python
from semantic_divergence_bench import DivergenceBench

# Initialize benchmark
bench = DivergenceBench(
    models=['gpt-4o-mini', 'claude-haiku-4-5', 'gemini-2.5-flash', 'meta/meta-llama-3-8b-instruct', 'meta-llama/Meta-Llama-3-8B-Instruct-Lite'],
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

## Research Questions to Explore

1. Do models with more parameters exhibit different divergence patterns?
2. Is there a correlation between model performance benchmarks and stability?
3. Can we predict which prompts will cause chaotic behavior?
4. Do instruction-tuned models show different stability than base models?
5. How does divergence relate to model confidence/calibration?
6. Can we find adversarial prompts that maximize divergence?

---

## Key Considerations

### Cost & Performance
- **Budget**: ~$500-1000 for comprehensive benchmarking (primarily API costs)
- **Caching**: All API responses cached locally to minimize costs and enable reproducibility
- **MPS Acceleration**: Sentence-BERT runs on Apple Silicon GPU (10x+ faster than CPU)
- **Storage**: ~5-10GB for cached embeddings and API responses

### Best Practices
- **Reproducibility**: Set seeds, log all parameters and model versions
- **Rate Limiting**: Implement throttling and retry logic for each API
- **Security**: Use `.env` file for API keys (never commit to git)
- **Ethics**: Ensure prompts are appropriate and non-harmful