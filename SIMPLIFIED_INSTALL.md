# Simplified Installation Options for Meno

We've simplified our installation options into three straightforward choices based on common use cases:

## 1. Lightweight (CPU)
For basic topic modeling with minimal dependencies and fast performance:

```bash
pip install "meno[lightweight]"
```

Includes:
- scikit-learn based models (TF-IDF, NMF, LSA)
- Basic visualizations with Plotly
- Word cloud generation
- All core dependencies

Ideal for:
- Quick exploratory analysis
- Low-resource environments
- When you need just the essentials

## 2. Standard (CPU)
For full-featured topic modeling optimized for CPU:

```bash
pip install "meno[cpu]" -f https://download.pytorch.org/whl/torch_stable.html
```

Includes:
- All lightweight components
- CPU-optimized embedding models
- BERTopic with UMAP and HDBSCAN clustering
- Advanced visualizations
- Text processing with spaCy
- LDA modeling with gensim

Ideal for:
- Production-quality topic modeling
- Systems without GPU acceleration
- When quality matters more than speed

## 3. GPU-Accelerated
For maximum performance with GPU acceleration:

```bash
pip install "meno[gpu]"
```

Includes:
- All standard components
- GPU-optimized PyTorch
- Accelerate and BitsAndBytes for optimization
- 8-bit quantization support
- Maximum performance for all models

Ideal for:
- Systems with NVIDIA GPUs
- Processing large document collections
- Research environments

## Which Option Is Right For You?

- **Lightweight**: Choose this if you're just getting started, have limited resources, or need basic topic modeling capabilities for smaller datasets.

- **Standard**: Best for most users who want high-quality topic modeling without a GPU. Works well for medium-sized document collections.

- **GPU**: Best for researchers and large-scale applications with access to NVIDIA GPUs. Significantly faster for large datasets.

## Running the Quality-First CPU Example

To run the `cpu_quality_first.ipynb` notebook, which uses UMAP and BERTopic for high-quality results:

```bash
pip install "meno[cpu]" -f https://download.pytorch.org/whl/torch_stable.html
```

Also download the required embedding model:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
# This caches the model to your local system
```
