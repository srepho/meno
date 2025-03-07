# Meno - Topic Modeling Toolkit Development Guidelines

## Build & Installation
```bash
# Install with uv
uv pip install -e .

# Install with dev dependencies
uv pip install -e ".[dev]"

# Install with CPU-only embeddings (recommended)
uv pip install -e ".[embeddings]"

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate     # Windows
```

## Test Commands
```bash
# IMPORTANT: Always run tests from within a virtual environment (uv or conda)
# Python 3.10 is the primary target for testing

# Create and activate a test environment with uv
uv venv -p 3.10
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate     # Windows

# Create and activate a test environment with conda
conda create -n meno_test python=3.10
conda activate meno_test

# Run all tests
python -m pytest

# Run single test file
python -m pytest tests/test_module_name.py

# Run single test
python -m pytest tests/test_module_name.py::test_function_name

# Run tests with coverage
python -m pytest --cov=meno

# Run CPU-specific tests
python -m pytest tests/test_cpu_embedding.py
```

## Project Structure
```
meno/
├── meno/                      # Main package
│   ├── preprocessing/         # Text cleaning & normalization
│   ├── modeling/              # Topic modeling (LDA & LLM)
│   ├── visualization/         # Plots & UMAP visualizations
│   ├── reporting/             # HTML report generation
│   ├── active_learning/       # Cleanlab integration
│   └── utils/                 # Config handling, I/O, etc.
├── config/                    # Configuration templates
├── examples/                  # Example notebooks
└── tests/                     # Test suite
```

## Code Style Guidelines
- **Python Version**: 3.10+ (primary target: 3.10)
- **Formatting**: Use black with line length 88
- **Linting**: Ruff for fast linting
- **Type Hints**: Use complete type hints throughout
- **Imports**: Group as stdlib, third-party, local (alphabetical)
- **Documentation**: Numpy-style docstrings with examples
- **Data Handling**: Pandas primary, with Polars optimizations for large datasets
- **Models**: ModernBERT as base embedding model (CPU-first design)
- **Error Handling**: Use specific exceptions with helpful messages
- **Config**: YAML with pydantic validation for type safety
- **Device Handling**: Default to CPU for all models, with opt-in GPU support