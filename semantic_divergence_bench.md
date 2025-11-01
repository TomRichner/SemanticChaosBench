# Semantic Divergence Bench
## Measuring Chaos and Stability in Large Language Models

### Project Overview
Semantic Divergence Bench measures how small perturbations in input prompts lead to diverging outputs in LLMs, analogous to Lyapunov exponents in dynamical systems. By creating semantically similar prompt pairs and tracking output divergence, we can characterize the stability/chaos regimes of different models.

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
    Supports:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic (Claude 3.5, Claude 3)
    - Google (Gemini Pro, PaLM)
    - Open models (Llama, Mistral via HuggingFace)
    - Local models (via transformers library)
    """
    
    def generate(prompt, temperature=0.7, max_tokens=500):
        # Returns: text, latency, token_count
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
- [ ] Set up Sentence-BERT for embedding computation
- [ ] Implement prompt perturbation generator
- [ ] Create unified model API interface
- [ ] Build basic divergence measurement

### Phase 2: Perturbation Generation (Week 2)
- [ ] Implement paraphrase generation using GPT-4/Claude
- [ ] Build semantic distance filtering
- [ ] Create prompt pair validation
- [ ] Generate test dataset of 100 prompt pairs at various ε levels

### Phase 3: Model Integration (Week 3)
- [ ] Integrate OpenAI API
- [ ] Integrate Anthropic API
- [ ] Integrate Google/Vertex AI
- [ ] Set up HuggingFace pipelines for open models
- [ ] Implement rate limiting and error handling

### Phase 4: Measurement Suite (Week 4)
- [ ] Build single-step divergence measurement
- [ ] Implement multi-step conversation tracking
- [ ] Create visualization tools
- [ ] Build statistical analysis pipeline

### Phase 5: Benchmarking (Week 5)
- [ ] Run systematic experiments across models
- [ ] Generate divergence profiles for each model
- [ ] Identify chaos/stability regimes
- [ ] Create comparative analysis

---

## Code Structure

```
semantic_divergence_bench/
├── src/
│   ├── perturbation/
│   │   ├── paraphrase_generator.py
│   │   ├── semantic_filter.py
│   │   └── prompt_pairs.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── openai_wrapper.py
│   │   ├── anthropic_wrapper.py
│   │   ├── google_wrapper.py
│   │   └── huggingface_wrapper.py
│   ├── measurement/
│   │   ├── embeddings.py
│   │   ├── divergence.py
│   │   └── trajectories.py
│   ├── analysis/
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   └── statistics.py
│   └── utils/
│       ├── config.py
│       ├── cache.py
│       └── logging.py
├── experiments/
│   ├── configs/
│   └── results/
├── data/
│   ├── prompt_pairs/
│   └── outputs/
├── notebooks/
│   ├── exploration.ipynb
│   └── analysis.ipynb
├── tests/
├── requirements.txt
└── README.md
```

---

## Key Dependencies

```python
# requirements.txt
sentence-transformers>=2.2.0  # Sentence-BERT
openai>=1.0.0                # OpenAI API
anthropic>=0.18.0            # Anthropic API
google-cloud-aiplatform      # Vertex AI
transformers>=4.35.0         # HuggingFace models
torch>=2.0.0                 # PyTorch backend
numpy>=1.24.0                # Numerical operations
pandas>=2.0.0                # Data management
matplotlib>=3.7.0            # Visualization
seaborn>=0.12.0             # Statistical plots
tqdm>=4.65.0                # Progress bars
hydra-core>=1.3.0           # Configuration management
wandb>=0.15.0               # Experiment tracking
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

1. **Pilot Study**: Start with 10 prompt pairs on 2 models to validate approach
2. **Optimize Costs**: Implement caching and batch processing
3. **Community Input**: Share methodology for feedback
4. **Scaling**: Consider distributed execution for large-scale experiments
5. **Publication**: Prepare findings for academic paper or blog post

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

- **API Costs**: Budget ~$500-1000 for comprehensive benchmarking
- **Rate Limits**: Implement proper throttling and retry logic
- **Reproducibility**: Set seeds where possible, log all parameters
- **Ethics**: Ensure prompts are appropriate and non-harmful
- **Caching**: Store all API responses to avoid repeated calls
- **Versioning**: Track model versions as they update over time