"""Configuration loading and validation utilities."""

from typing import Dict, List, Optional, Union, Any
import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field, validator


class NormalizationConfig(BaseModel):
    """Configuration for text normalization."""
    
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    lemmatize: bool = True
    language: str = "en"


class StopwordsConfig(BaseModel):
    """Configuration for stopword handling."""
    
    use_default: bool = True
    custom: List[str] = Field(default_factory=list)
    keep: List[str] = Field(default_factory=list)


class SpellingConfig(BaseModel):
    """Configuration for spelling correction."""
    
    enabled: bool = True
    max_distance: int = 2
    min_word_length: int = 4
    custom_dictionary: Dict[str, str] = Field(default_factory=dict)


class AcronymConfig(BaseModel):
    """Configuration for acronym expansion."""
    
    enabled: bool = True
    custom_mappings: Dict[str, str] = Field(default_factory=dict)


class PreprocessingConfig(BaseModel):
    """Configuration for text preprocessing."""
    
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    stopwords: StopwordsConfig = Field(default_factory=StopwordsConfig)
    spelling: SpellingConfig = Field(default_factory=SpellingConfig)
    acronyms: AcronymConfig = Field(default_factory=AcronymConfig)


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    
    model_name: str = "answerdotai/ModernBERT-base"
    batch_size: int = 32
    use_gpu: bool = False
    local_model_path: Optional[str] = None


class LDAConfig(BaseModel):
    """Configuration for LDA topic modeling."""
    
    num_topics: int = 10
    passes: int = 20
    iterations: int = 400
    alpha: Union[str, float] = "auto"
    eta: Union[str, float] = "auto"
    
    @validator("alpha")
    def validate_alpha(cls, v):
        if isinstance(v, str) and v not in ["auto", "symmetric", "asymmetric"]:
            raise ValueError("alpha must be 'auto', 'symmetric', 'asymmetric', or a float")
        return v
    
    @validator("eta")
    def validate_eta(cls, v):
        if isinstance(v, str) and v not in ["auto", "symmetric"]:
            raise ValueError("eta must be 'auto', 'symmetric', or a float")
        return v


class ClusteringConfig(BaseModel):
    """Configuration for clustering algorithms."""
    
    algorithm: str = "hdbscan"
    min_cluster_size: int = 15
    min_samples: int = 5
    cluster_selection_method: str = "eom"
    n_clusters: int = 10
    
    @validator("algorithm")
    def validate_algorithm(cls, v):
        allowed = ["hdbscan", "kmeans", "agglomerative"]
        if v not in allowed:
            raise ValueError(f"algorithm must be one of {allowed}")
        return v
    
    @validator("cluster_selection_method")
    def validate_selection_method(cls, v):
        allowed = ["eom", "leaf"]
        if v not in allowed:
            raise ValueError(f"cluster_selection_method must be one of {allowed}")
        return v


class TopicMatchingConfig(BaseModel):
    """Configuration for supervised topic matching."""
    
    threshold: float = 0.6
    assign_multiple: bool = False
    max_topics_per_doc: int = 3


class ModelingConfig(BaseModel):
    """Configuration for topic modeling."""
    
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    lda: LDAConfig = Field(default_factory=LDAConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    topic_matching: TopicMatchingConfig = Field(default_factory=TopicMatchingConfig)


class UMAPConfig(BaseModel):
    """Configuration for UMAP dimensionality reduction."""
    
    n_neighbors: int = 15
    n_components: int = 2
    min_dist: float = 0.1
    metric: str = "cosine"


class PlotConfig(BaseModel):
    """Configuration for plots."""
    
    width: int = 900
    height: int = 600
    colorscale: str = "Viridis"
    marker_size: int = 5
    opacity: float = 0.7


class WordCloudConfig(BaseModel):
    """Configuration for word clouds."""
    
    max_words: int = 100
    background_color: str = "white"
    width: int = 800
    height: int = 400


class VisualizationConfig(BaseModel):
    """Configuration for visualizations."""
    
    umap: UMAPConfig = Field(default_factory=UMAPConfig)
    plots: PlotConfig = Field(default_factory=PlotConfig)
    wordcloud: WordCloudConfig = Field(default_factory=WordCloudConfig)


class HTMLReportConfig(BaseModel):
    """Configuration for HTML reports."""
    
    title: str = "Topic Modeling Results"
    include_interactive: bool = True
    max_examples_per_topic: int = 5
    include_raw_data: bool = False
    max_samples_per_topic: int = 5  # Maximum number of samples per topic in the raw data table


class ExportConfig(BaseModel):
    """Configuration for data export."""
    
    formats: List[str] = Field(default_factory=lambda: ["csv", "json"])
    include_embeddings: bool = False
    
    @validator("formats")
    def validate_formats(cls, v):
        allowed = ["csv", "json", "excel", "pickle"]
        for fmt in v:
            if fmt not in allowed:
                raise ValueError(f"Format {fmt} not supported. Must be one of {allowed}")
        return v


class ReportingConfig(BaseModel):
    """Configuration for reporting."""
    
    html: HTMLReportConfig = Field(default_factory=HTMLReportConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)


class MenoConfig(BaseModel):
    """Complete configuration for meno."""
    
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    modeling: ModelingConfig = Field(default_factory=ModelingConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)


def load_config(config_path: Optional[Union[str, Path]] = None) -> MenoConfig:
    """Load and validate configuration from a YAML file.
    
    Parameters
    ----------
    config_path : Optional[Union[str, Path]], optional
        Path to the configuration file, by default None
        If None, loads the default configuration
    
    Returns
    -------
    MenoConfig
        Validated configuration object
    """
    if config_path is None:
        # First try to find the default config in the installed package
        try:
            import importlib.resources as pkg_resources
            import meno
            config_text = pkg_resources.read_text(meno, "default_config.yaml")
            config_dict = yaml.safe_load(config_text)
            return MenoConfig(**config_dict)
        except (ImportError, FileNotFoundError):
            # Fall back to looking for the file in the repository
            default_config_path = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
            config_path = default_config_path
    
    # Load YAML config from file
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        # If all else fails, use hardcoded default values from the MenoConfig class
        print(f"Warning: Config file {config_path} not found. Using default configuration.")
        return MenoConfig()
    
    # Validate with pydantic
    return MenoConfig(**config_dict)


def merge_configs(base_config: MenoConfig, override_config: Dict[str, Any]) -> MenoConfig:
    """Merge a base configuration with override values.
    
    Parameters
    ----------
    base_config : MenoConfig
        Base configuration to start with
    override_config : Dict[str, Any]
        Dictionary of override values
    
    Returns
    -------
    MenoConfig
        Merged configuration
    """
    # Convert to dict, update, and convert back to MenoConfig
    config_dict = base_config.dict()
    
    # Recursively update the config
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    updated_dict = update_dict(config_dict, override_config)
    return MenoConfig(**updated_dict)