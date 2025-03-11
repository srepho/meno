"""Configuration loading and validation utilities."""

from typing import Dict, List, Optional, Union, Any
import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field, validator, field_validator


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
    precision: str = "float32"  # "float32" or "float16"
    use_mmap: bool = True
    cache_dir: Optional[str] = None
    cache_embeddings: bool = True
    
    @field_validator("precision")
    def validate_precision(cls, v):
        allowed = ["float32", "float16"]
        if v not in allowed:
            raise ValueError(f"precision must be one of {allowed}")
        return v


class LDAConfig(BaseModel):
    """Configuration for LDA topic modeling."""
    
    num_topics: int = 10
    passes: int = 20
    iterations: int = 400
    alpha: Union[str, float] = "auto"
    eta: Union[str, float] = "auto"
    
    @field_validator("alpha")
    def validate_alpha(cls, v):
        if isinstance(v, str) and v not in ["auto", "symmetric", "asymmetric"]:
            raise ValueError("alpha must be 'auto', 'symmetric', 'asymmetric', or a float")
        return v
    
    @field_validator("eta")
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
    
    @field_validator("algorithm")
    def validate_algorithm(cls, v):
        allowed = ["hdbscan", "kmeans", "agglomerative"]
        if v not in allowed:
            raise ValueError(f"algorithm must be one of {allowed}")
        return v
    
    @field_validator("cluster_selection_method")
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
    auto_detect_topics: bool = True


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
    
    @field_validator("formats")
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


class WorkflowFeaturesConfig(BaseModel):
    """Configuration for workflow feature toggles."""
    
    acronym_detection: bool = True
    spelling_correction: bool = True
    interactive_reports: bool = True
    auto_open_browser: bool = False


class WorkflowReportPathsConfig(BaseModel):
    """Configuration for default report paths."""
    
    acronym_report: str = "meno_acronym_report.html"
    spelling_report: str = "meno_spelling_report.html"
    comprehensive_report: str = "meno_topic_report.html"


class WorkflowInteractiveConfig(BaseModel):
    """Configuration for interactive report settings."""
    
    max_acronyms: int = 30
    min_acronym_length: int = 2
    min_acronym_count: int = 3
    max_misspellings: int = 30
    min_word_length: int = 4
    min_word_count: int = 3


class WorkflowConfig(BaseModel):
    """Configuration for workflow behavior."""
    
    features: WorkflowFeaturesConfig = Field(default_factory=WorkflowFeaturesConfig)
    report_paths: WorkflowReportPathsConfig = Field(default_factory=WorkflowReportPathsConfig)
    interactive: WorkflowInteractiveConfig = Field(default_factory=WorkflowInteractiveConfig)


class VisualizationDefaultsConfig(BaseModel):
    """Configuration for default visualization types."""
    
    plot_type: str = "embeddings"
    map_type: str = "point_map"
    trend_type: str = "line"
    
    @field_validator("plot_type")
    def validate_plot_type(cls, v):
        allowed = ["embeddings", "distribution", "trends", "map", "timespace"]
        if v not in allowed:
            raise ValueError(f"plot_type must be one of {allowed}")
        return v
    
    @field_validator("map_type")
    def validate_map_type(cls, v):
        allowed = ["point_map", "choropleth", "density_map", "postcode_map"]
        if v not in allowed:
            raise ValueError(f"map_type must be one of {allowed}")
        return v
    
    @field_validator("trend_type")
    def validate_trend_type(cls, v):
        allowed = ["line", "heatmap", "stacked_area", "ridge", "calendar"]
        if v not in allowed:
            raise ValueError(f"trend_type must be one of {allowed}")
        return v


class TimeVisualizationConfig(BaseModel):
    """Configuration for time-based visualizations."""
    
    date_format: str = "%Y-%m-%d"
    resample_freq: str = "W"  # Weekly
    
    @field_validator("resample_freq")
    def validate_resample_freq(cls, v):
        allowed = ["D", "W", "M", "Q", "Y"]
        if v not in allowed:
            raise ValueError(f"resample_freq must be one of {allowed}")
        return v


class GeoVisualizationConfig(BaseModel):
    """Configuration for geospatial visualizations."""
    
    map_style: str = "carto-positron"
    zoom: int = 4
    center: Dict[str, float] = Field(default_factory=lambda: {"lat": -25.2744, "lon": 133.7751})


class CategoryVisualizationConfig(BaseModel):
    """Configuration for category-based visualizations."""
    
    max_categories: int = 8
    color_palette: str = "rainbow"


class ExtendedVisualizationConfig(VisualizationConfig):
    """Extended visualization configuration with workflow-specific settings."""
    
    defaults: VisualizationDefaultsConfig = Field(default_factory=VisualizationDefaultsConfig)
    time: TimeVisualizationConfig = Field(default_factory=TimeVisualizationConfig)
    geo: GeoVisualizationConfig = Field(default_factory=GeoVisualizationConfig)
    category: CategoryVisualizationConfig = Field(default_factory=CategoryVisualizationConfig)


class PerformanceConfig(BaseModel):
    """Configuration for performance and memory management."""
    
    low_memory: bool = True
    use_mmap: bool = True
    cache_dir: Optional[str] = None
    persist_embeddings: bool = True
    clean_temp_files_on_exit: bool = True
    precision: str = "float32"  # "float32" or "float16"
    
    @field_validator("precision")
    def validate_precision(cls, v):
        allowed = ["float32", "float16"]
        if v not in allowed:
            raise ValueError(f"precision must be one of {allowed}")
        return v

class ExtendedModelingConfig(ModelingConfig):
    """Extended modeling configuration with workflow-specific settings."""
    
    default_method: str = "embedding_cluster"
    default_num_topics: int = 10
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    @field_validator("default_method")
    def validate_default_method(cls, v):
        allowed = ["embedding_cluster", "lda", "bertopic"]
        if v not in allowed:
            raise ValueError(f"default_method must be one of {allowed}")
        return v
    
    # Update the embeddings config to include additional fields
    @validator("embeddings", pre=True)  # Keep using validator with pre=True since field_validator doesn't support it
    def update_embeddings_config(cls, v):
        # Add performance parameters if they don't exist in embeddings
        if isinstance(v, EmbeddingConfig):
            embeddings_dict = v.model_dump()
            embeddings_dict.setdefault("use_gpu", False)
            embeddings_dict.setdefault("precision", "float32")
            embeddings_dict.setdefault("use_mmap", True)
            embeddings_dict.setdefault("cache_embeddings", True)
            return EmbeddingConfig(**embeddings_dict)
        return v


class MenoConfig(BaseModel):
    """Complete configuration for meno."""
    
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    modeling: ModelingConfig = Field(default_factory=ModelingConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)


class WorkflowMenoConfig(BaseModel):
    """Complete configuration for meno with workflow-specific settings."""
    
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    modeling: ExtendedModelingConfig = Field(default_factory=ExtendedModelingConfig)
    visualization: ExtendedVisualizationConfig = Field(default_factory=ExtendedVisualizationConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)


def create_default_config_file(config_type: str = "standard", output_path: Optional[Union[str, Path]] = None) -> str:
    """Create a default configuration file.
    
    Parameters
    ----------
    config_type : str, optional
        Type of configuration to create, by default "standard"
        Options: "standard", "workflow"
    output_path : Optional[Union[str, Path]], optional
        Path to save the configuration file, by default None
        If None, saves to the current directory with a default name
        
    Returns
    -------
    str
        Path to the created configuration file
    """
    # Create default config object
    if config_type == "workflow":
        config = WorkflowMenoConfig()
        default_filename = "meno_workflow_config.yaml"
    else:
        config = MenoConfig()
        default_filename = "meno_config.yaml"
    
    # Determine output path
    if output_path is None:
        output_path = Path.cwd() / default_filename
    else:
        output_path = Path(output_path)
    
    # Convert to dict and save to YAML
    config_dict = config.model_dump()
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created default {config_type} configuration file at {output_path}")
    return str(output_path)

def load_config(
    config_path: Optional[Union[str, Path]] = None,
    config_type: str = "standard",
    auto_create: bool = True
) -> Union[MenoConfig, WorkflowMenoConfig]:
    """Load and validate configuration from a YAML file.
    
    Parameters
    ----------
    config_path : Optional[Union[str, Path]], optional
        Path to the configuration file, by default None
        If None, loads the default configuration
    config_type : str, optional
        Type of configuration to load, by default "standard"
        Options: "standard", "workflow"
    auto_create : bool, optional
        Whether to automatically create a config file if none exists, by default True
    
    Returns
    -------
    Union[MenoConfig, WorkflowMenoConfig]
        Validated configuration object
    """
    if config_path is None:
        # First try to find the default config in the installed package
        try:
            import importlib.resources as pkg_resources
            import meno
            
            if config_type == "workflow":
                filename = "workflow_config.yaml"
            else:
                filename = "default_config.yaml"
                
            try:
                config_text = pkg_resources.read_text(meno, filename)
                config_dict = yaml.safe_load(config_text)
            except FileNotFoundError:
                # Try to load from config directory
                config_text = pkg_resources.read_text(meno.config, filename)
                config_dict = yaml.safe_load(config_text)
                
        except (ImportError, FileNotFoundError, AttributeError):
            # Fall back to looking for the file in the repository
            if config_type == "workflow":
                default_config_path = Path(__file__).parent.parent.parent / "config" / "workflow_config.yaml"
            else:
                default_config_path = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
                
            config_path = default_config_path
            
            # Try to load YAML config from file
            try:
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            except FileNotFoundError:
                # If auto_create is enabled, create a default config file
                if auto_create:
                    new_config_path = create_default_config_file(config_type)
                    with open(new_config_path, 'r') as f:
                        config_dict = yaml.safe_load(f)
                else:
                    # If all else fails, use hardcoded default values
                    print(f"Warning: Config file {config_path} not found. Using default configuration.")
                    if config_type == "workflow":
                        return WorkflowMenoConfig()
                    else:
                        return MenoConfig()
    else:
        # Try to load YAML config from specified file
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except FileNotFoundError:
            if auto_create:
                # Create config at the specified path
                create_default_config_file(config_type, config_path)
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            else:
                # If file not found, use hardcoded default values
                print(f"Warning: Config file {config_path} not found. Using default configuration.")
                if config_type == "workflow":
                    return WorkflowMenoConfig()
                else:
                    return MenoConfig()
    
    # Validate with pydantic
    if config_type == "workflow":
        # Check if workflow section exists, if not, add it
        if "workflow" not in config_dict:
            config_dict["workflow"] = WorkflowConfig().model_dump()
        return WorkflowMenoConfig(**config_dict)
    else:
        return MenoConfig(**config_dict)


def merge_configs(
    base_config: Union[MenoConfig, WorkflowMenoConfig],
    override_config: Dict[str, Any]
) -> Union[MenoConfig, WorkflowMenoConfig]:
    """Merge a base configuration with override values.
    
    Parameters
    ----------
    base_config : Union[MenoConfig, WorkflowMenoConfig]
        Base configuration to start with
    override_config : Dict[str, Any]
        Dictionary of override values
    
    Returns
    -------
    Union[MenoConfig, WorkflowMenoConfig]
        Merged configuration
    """
    # Convert to dict, update, and convert back
    config_dict = base_config.model_dump()
    
    # Recursively update the config
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    updated_dict = update_dict(config_dict, override_config)
    
    # Determine config type and return appropriate class
    if isinstance(base_config, WorkflowMenoConfig):
        return WorkflowMenoConfig(**updated_dict)
    else:
        return MenoConfig(**updated_dict)


def save_config(
    config: Union[MenoConfig, WorkflowMenoConfig],
    output_path: Union[str, Path]
) -> None:
    """Save a configuration to a YAML file.
    
    Parameters
    ----------
    config : Union[MenoConfig, WorkflowMenoConfig]
        Configuration to save
    output_path : Union[str, Path]
        Path to save the configuration to
    """
    # Convert to dict
    config_dict = config.dict()
    
    # Save to YAML
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)