# Quick Start Guide

## Installation Complete! ✓

Your Semantic Chaos Bench environment is now set up and ready to go.

### What's Been Done

✓ Directory structure created  
✓ Virtual environment initialized with `uv`  
✓ All dependencies installed (170 packages)  
✓ PyTorch 2.9.0 with MPS (Metal) acceleration configured  
✓ Sentence-BERT tested and working on MPS  
✓ Project structure validated  

### Current Status

**Phase 1 Progress:**
- [x] **1a. Set up project with uv and dependencies** - COMPLETED
- [ ] 1b. Configure Sentence-BERT with MPS acceleration (local Mac)
- [ ] 1c. Implement prompt perturbation generator
- [ ] 1d. Create unified model API interface
- [ ] 1e. Build basic divergence measurement
- [x] **1f. Set up API key management (.env file)** - COMPLETED

### Next Steps

#### 1. Configure API Keys (Required for LLM access)

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Then edit `.env` and add your actual API keys:

```bash
# Required for most experiments
OPENAI_API_KEY=sk-your-actual-key
ANTHROPIC_API_KEY=sk-ant-your-actual-key

# Optional - for open source models
REPLICATE_API_TOKEN=r8_your-actual-token
TOGETHER_API_KEY=your-actual-key

# Optional - for experiment tracking
WANDB_API_KEY=your-actual-key
```

**Important:** Never commit the `.env` file to git! It's already in `.gitignore`.

#### 2. Verify Setup

Run the test script to verify everything is working:

```bash
source .venv/bin/activate
python scripts/test_setup.py
```

#### 3. Continue Phase 1 Implementation

The following tasks remain in Phase 1:
- Configure Sentence-BERT with MPS acceleration
- Implement prompt perturbation generator  
- Create unified model API interface
- Build basic divergence measurement

### Project Structure

```
semantic_chaos_bench/
├── src/                    # Core source code
│   ├── perturbation/       # Prompt perturbation generation
│   ├── models/             # LLM API wrappers
│   ├── measurement/        # Divergence measurement
│   ├── analysis/           # Analysis and visualization
│   └── utils/              # Utilities (config, cache, logging)
├── scripts/                # Executable scripts
├── experiments/            # Experiment configs and results
├── data/                   # Data storage and cache
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
├── config.yaml             # Default configuration
├── pyproject.toml          # Project dependencies
└── .env                    # API keys (create from .env.example)
```

### Available Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Test setup
python scripts/test_setup.py

# Run pilot study (once Phase 1 is complete)
python scripts/pilot_study.py

# Generate prompt pairs
python scripts/generate_prompt_pairs.py

# Run full benchmark
python scripts/run_benchmark.py

# Analyze results
python scripts/analyze_results.py
```

### Key Features

- **MPS Acceleration**: Sentence-BERT runs on Apple Silicon GPU (10x+ faster than CPU)
- **API-Only LLMs**: All model inference via cloud APIs (no local hosting needed)
- **Fast Package Management**: Using `uv` instead of pip
- **Caching**: API responses cached locally to minimize costs
- **Modular Design**: Clean separation between perturbation, measurement, and analysis

### Installed Packages (Key Dependencies)

- `torch==2.9.0` - PyTorch with MPS support
- `sentence-transformers==5.1.2` - Sentence-BERT embeddings
- `openai==2.6.1` - OpenAI API client
- `anthropic==0.72.0` - Anthropic API client
- `replicate==1.0.7` - Replicate API client
- `together==1.5.29` - Together AI API client
- `numpy`, `pandas`, `scipy` - Data processing
- `matplotlib`, `seaborn` - Visualization
- `pytest`, `jupyter`, `black`, `ruff` - Development tools
- `wandb` - Experiment tracking (optional)

### Troubleshooting

**MPS not available?**
```bash
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

**Import errors?**
```bash
# Make sure you're in the virtual environment
source .venv/bin/activate
```

**Need to reinstall?**
```bash
uv pip install -e ".[dev]"
```

### Documentation

- `README.md` - Full project overview and architecture
- `config.yaml` - Configuration settings
- `.env.example` - API key template

### Getting Help

If you encounter issues:
1. Check that MPS is available: `python scripts/test_setup.py`
2. Verify all directories exist (test script will show)
3. Ensure API keys are set in `.env` file
4. Check that virtual environment is activated

---

**Ready to continue?** Proceed with implementing the remaining Phase 1 tasks!

