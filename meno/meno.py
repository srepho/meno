"""Main interface for the meno topic modeling toolkit."""

from typing import List, Dict, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
from datetime import datetime

# Import from submodules
from .utils.config import MenoConfig, load_config, merge_configs
from .preprocessing import TextNormalizer
from .modeling import DocumentEmbedding, LDAModel, EmbeddingClusterModel, TopicMatcher
from .visualization import (
    plot_embeddings, plot_topic_distribution, create_umap_projection,
    # Time series visualizations
    create_topic_trend_plot, create_topic_heatmap, create_topic_stacked_area,
    create_topic_ridge_plot, create_topic_calendar_heatmap,
    # Geospatial visualizations
    create_topic_map, create_region_choropleth, create_topic_density_map,
    create_postcode_map,
    # Time-space visualizations
    create_animated_map, create_space_time_heatmap, create_category_time_plot
)

# Set up logging
logger = logging.getLogger(__name__)


class MenoTopicModeler:
    """Main interface for topic modeling with meno.
    
    This class provides a high-level interface to the meno topic modeling toolkit,
    integrating preprocessing, modeling, visualization, and reporting components.
    
    Parameters
    ----------
    config_path : Optional[Union[str, Path]], optional
        Path to configuration file, by default None
        If None, the default configuration is used
    config_overrides : Optional[Dict[str, Any]], optional
        Dictionary of configuration overrides, by default None
    
    Attributes
    ----------
    config : MenoConfig
        Configuration object
    text_normalizer : TextNormalizer
        Text normalization component
    embedding_model : DocumentEmbedding
        Document embedding model
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[Any] = None,
        offline_mode: bool = False,
        auto_create_config: bool = True,
    ):
        """Initialize the topic modeler with configuration."""
        # Load configuration
        self.config = load_config(config_path, auto_create=auto_create_config)
        
        # Add offline_mode to config overrides if specified
        if offline_mode:
            config_overrides = config_overrides or {}
            config_overrides['offline_mode'] = offline_mode
        
        # Apply overrides if provided
        if config_overrides:
            self.config = merge_configs(self.config, config_overrides)
            
        # Store embedding model and offline mode if provided
        self.custom_embedding_model = embedding_model
        self.offline_mode = offline_mode
            
        # Initialize components
        self._initialize_components()
        
        # Storage for results
        self.documents = None
        self.document_embeddings = None
        self.topics = None
        self.topic_embeddings = None
        self.topic_assignments = None
        self.umap_projection = None
        
        logger.info("MenoTopicModeler initialized")
    
    def _initialize_components(self):
        """Initialize all components based on configuration."""
        # Initialize text normalizer
        self.text_normalizer = TextNormalizer(
            lowercase=self.config.preprocessing.normalization.lowercase,
            remove_punctuation=self.config.preprocessing.normalization.remove_punctuation,
            remove_numbers=self.config.preprocessing.normalization.remove_numbers,
            lemmatize=self.config.preprocessing.normalization.lemmatize,
            language=self.config.preprocessing.normalization.language,
            stopwords_config=self.config.preprocessing.stopwords,
        )
        
        # Initialize embedding model if not provided
        try:
            if hasattr(self, 'custom_embedding_model') and self.custom_embedding_model is not None:
                self.embedding_model = self.custom_embedding_model
                logger.debug(f"Using custom embedding model: {type(self.embedding_model)}")
            else:
                self.embedding_model = DocumentEmbedding(
                    model_name=self.config.modeling.embeddings.model_name,
                    batch_size=self.config.modeling.embeddings.batch_size,
                    local_files_only=self.config.modeling.embeddings.local_files_only if hasattr(self.config.modeling.embeddings, 'local_files_only') else False,
                )
                logger.debug(f"Initialized embedding model: {self.config.modeling.embeddings.model_name}")
            
            # Verify the embedding model is properly initialized
            if not hasattr(self.embedding_model, 'embed_documents'):
                logger.warning(f"Embedding model doesn't have embed_documents method. Type: {type(self.embedding_model)}")
                
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise ValueError(f"Failed to initialize embedding model: {e}")
        
        logger.debug("Components initialized")
    
    def preprocess(
        self,
        documents: Union[List[str], pd.Series, pd.DataFrame],
        text_column: Optional[str] = None,
        id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Preprocess documents for topic modeling.
        
        Parameters
        ----------
        documents : Union[List[str], pd.Series, pd.DataFrame]
            Documents to preprocess
            If DataFrame, text_column must be specified
        text_column : Optional[str], optional
            Column containing text in DataFrame, by default None
        id_column : Optional[str], optional
            Column to use as document ID, by default None
        
        Returns
        -------
        pd.DataFrame
            DataFrame with preprocessed documents
        """
        # Convert input to DataFrame
        if isinstance(documents, list):
            self.documents = pd.DataFrame({"text": documents})
        elif isinstance(documents, pd.Series):
            self.documents = pd.DataFrame({"text": documents})
        elif isinstance(documents, pd.DataFrame):
            if text_column is None:
                raise ValueError("text_column must be specified for DataFrame input")
            self.documents = documents.copy()
            # Rename text column to 'text' for consistency
            self.documents.rename(columns={text_column: "text"}, inplace=True)
        else:
            raise TypeError(
                f"documents must be List[str], pd.Series, or pd.DataFrame, got {type(documents)}"
            )
        
        # Use id_column if provided, else create document IDs
        if id_column and id_column in self.documents.columns:
            self.documents.rename(columns={id_column: "doc_id"}, inplace=True)
        else:
            self.documents["doc_id"] = [f"doc_{i}" for i in range(len(self.documents))]
        
        # Apply preprocessing pipeline
        logger.info(f"Preprocessing {len(self.documents)} documents")
        
        # Apply text normalization
        self.documents["processed_text"] = self.text_normalizer.normalize_batch(
            self.documents["text"]
        )
        
        logger.info("Preprocessing complete")
        return self.documents
    
    def embed_documents(self) -> np.ndarray:
        """Generate embeddings for preprocessed documents.
        
        Returns
        -------
        np.ndarray
            Document embeddings
        """
        if self.documents is None or "processed_text" not in self.documents.columns:
            raise ValueError("Documents must be preprocessed before embedding")
        
        logger.info(f"Generating embeddings for {len(self.documents)} documents")
        
        # Verify that embedding_model is a proper object with embed_documents method
        if self.embedding_model is None:
            raise ValueError("Embedding model is not initialized")
            
        if isinstance(self.embedding_model, str):
            raise ValueError(f"Embedding model is a string '{self.embedding_model}' instead of an initialized model object. This usually happens when the model failed to load properly.")
            
        if not hasattr(self.embedding_model, 'embed_documents'):
            raise ValueError(f"Embedding model does not have an 'embed_documents' method. Type: {type(self.embedding_model)}")
        
        self.document_embeddings = self.embedding_model.embed_documents(
            self.documents["processed_text"]
        )
        
        logger.info(
            f"Generated embeddings with shape {self.document_embeddings.shape}"
        )
        return self.document_embeddings
    
    def discover_topics(
        self,
        method: str = "embedding_cluster",
        modeling_approach: str = "bertopic",
        num_topics: Optional[int] = None,
        min_topic_similarity_threshold: float = 0.75,
        auto_reduce_topics: bool = True,
    ) -> pd.DataFrame:
        """Discover topics in an unsupervised manner.
        
        Parameters
        ----------
        method : str, optional
            Method to use for topic discovery, by default "embedding_cluster"
            Options: "lda", "embedding_cluster"
        modeling_approach : str, optional
            Specific modeling approach to use, by default "bertopic"
            Options: "bertopic", "lightweight", "nmf", "lsa", "tfidf"
        num_topics : Optional[int], optional
            Number of topics to discover, by default None
            If None, uses the value from config
        min_topic_similarity_threshold : float, optional
            Threshold for detecting duplicate topics (0.0-1.0), by default 0.75
            Higher values mean topics need to be more similar to be considered duplicates
        auto_reduce_topics : bool, optional
            Whether to automatically reduce the number of topics by merging similar ones, by default True
        
        Returns
        -------
        pd.DataFrame
            DataFrame with topic assignments
        """
        if self.documents is None:
            raise ValueError("Documents must be preprocessed before topic discovery")
        
        if self.document_embeddings is None and method == "embedding_cluster":
            logger.info("Document embeddings not found, generating them now")
            self.embed_documents()
        
        # Set num_topics from config if not provided, respecting auto_detect_topics setting
        if num_topics is None:
            if self.config.modeling.auto_detect_topics:
                # Auto-detect topics if configured to do so
                num_topics = None
            elif method == "lda":
                num_topics = self.config.modeling.lda.num_topics
            else:
                if self.config.modeling.clustering.algorithm == "hdbscan":
                    # HDBSCAN determines number of clusters automatically
                    num_topics = None
                else:
                    num_topics = self.config.modeling.clustering.n_clusters
        
        # Discover topics based on method
        if method == "lda":
            logger.info(f"Discovering topics using LDA with {num_topics} topics")
            lda_model = LDAModel(
                num_topics=num_topics,
                passes=self.config.modeling.lda.passes,
                iterations=self.config.modeling.lda.iterations,
                alpha=self.config.modeling.lda.alpha,
                eta=self.config.modeling.lda.eta,
            )
            topic_assignments = lda_model.fit_transform(self.documents["processed_text"])
            
        elif method == "embedding_cluster":
            logger.info(
                f"Discovering topics using embedding clustering with algorithm={self.config.modeling.clustering.algorithm}, "
                f"n_clusters={num_topics if num_topics is not None else 'auto-detected'}, "
                f"min_cluster_size={self.config.modeling.clustering.min_cluster_size}, "
                f"min_samples={self.config.modeling.clustering.min_samples}"
            )
            # If lightweight approach is selected, use the appropriate model
            if modeling_approach.lower() == "lightweight":
                # Import dynamically to avoid circular imports
                from meno.modeling.simple_models import SimpleTopicModel, TFIDFTopicModel, NMFTopicModel, LSATopicModel
                
                logger.info(f"Using lightweight topic modeling with {num_topics} topics (auto_reduce={auto_reduce_topics})")
                model = SimpleTopicModel(
                    num_topics=num_topics if num_topics else 10,
                    embedding_model=self.embedding_model,
                    min_topic_similarity_threshold=min_topic_similarity_threshold,
                    auto_reduce_topics=auto_reduce_topics,
                    random_state=42
                )
                model.fit(self.documents["processed_text"], self.document_embeddings)
                clusters, topic_matrix = model.transform(self.documents["processed_text"], self.document_embeddings)
                
                # Create assignments dataframe with descriptive topic labels
                topic_assignments = pd.DataFrame(
                    topic_matrix, 
                    columns=[model.topics[i] for i in range(len(model.topics))],
                    index=self.documents.index
                )
                topic_assignments["topic"] = [model.topics[c] for c in clusters]
                
            elif modeling_approach.lower() == "tfidf":
                # Use TF-IDF based topic modeling which doesn't require embeddings
                from meno.modeling.simple_models import TFIDFTopicModel
                
                logger.info(f"Using TF-IDF topic modeling with {num_topics} topics (auto_reduce={auto_reduce_topics})")
                model = TFIDFTopicModel(
                    num_topics=num_topics if num_topics else 10,
                    random_state=42
                )
                model.fit(self.documents["processed_text"])
                _, topic_matrix = model.transform(self.documents["processed_text"])
                
                # Create assignments dataframe with descriptive topic labels
                topic_assignments = pd.DataFrame(
                    topic_matrix, 
                    columns=[model.topics[i] for i in range(len(model.topics))],
                    index=self.documents.index
                )
                topic_assignments["topic"] = model.get_document_info(self.documents["processed_text"])["Name"]
                
            elif modeling_approach.lower() == "nmf":
                # Use NMF based topic modeling
                from meno.modeling.simple_models import NMFTopicModel
                
                logger.info(f"Using NMF topic modeling with {num_topics} topics")
                model = NMFTopicModel(
                    num_topics=num_topics if num_topics else 10,
                    random_state=42
                )
                model.fit(self.documents["processed_text"])
                topic_matrix = model.transform(self.documents["processed_text"])
                
                # Create assignments dataframe with descriptive topic labels
                topic_assignments = pd.DataFrame(
                    topic_matrix, 
                    columns=[model.topics[i] for i in range(len(model.topics))],
                    index=self.documents.index
                )
                topic_assignments["topic"] = model.get_document_info(self.documents["processed_text"])["Name"]
                
            elif modeling_approach.lower() == "lsa":
                # Use LSA based topic modeling
                from meno.modeling.simple_models import LSATopicModel
                
                logger.info(f"Using LSA topic modeling with {num_topics} topics")
                model = LSATopicModel(
                    num_topics=num_topics if num_topics else 10,
                    random_state=42
                )
                model.fit(self.documents["processed_text"])
                topic_matrix = model.transform(self.documents["processed_text"])
                
                # Create assignments dataframe with descriptive topic labels
                topic_assignments = pd.DataFrame(
                    topic_matrix, 
                    columns=[model.topics[i] for i in range(len(model.topics))],
                    index=self.documents.index
                )
                topic_assignments["topic"] = model.get_document_info(self.documents["processed_text"])["Name"]
                
            else:
                # Default to standard embedding clustering (bertopic approach)
                cluster_model = EmbeddingClusterModel(
                    algorithm=self.config.modeling.clustering.algorithm,
                    n_clusters=num_topics,
                    min_cluster_size=self.config.modeling.clustering.min_cluster_size,
                    min_samples=self.config.modeling.clustering.min_samples,
                    cluster_selection_method=self.config.modeling.clustering.cluster_selection_method,
                )
                topic_assignments = cluster_model.fit_transform(self.document_embeddings)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Store results
        if isinstance(topic_assignments, pd.DataFrame):
            self.topic_assignments = topic_assignments
        else:
            # Convert numpy array to DataFrame if needed
            self.topic_assignments = pd.DataFrame(
                topic_assignments,
                index=self.documents.index,
            )
            
        # Store modeling method for later reference
        self._last_modeling_method = method
        
        # Add topic assignments to documents DataFrame
        if "topic" not in self.topic_assignments.columns:
            # For clustering methods that return a single topic per document
            self.documents["topic"] = self.topic_assignments.idxmax(axis=1)
        else:
            # For methods that already assigned a topic column
            self.documents["topic"] = self.topic_assignments["topic"]
        
        # Get more descriptive information about the model used
        model_description = f"Model: {method}"
        if method == "embedding_cluster":
            model_description += f", Clustering: {self.config.modeling.clustering.algorithm}"
            if self.config.modeling.clustering.algorithm != "hdbscan":
                model_description += f", n_clusters: {num_topics}"
        elif method == "lda":
            model_description += f", Topics: {num_topics}"
        
        logger.info(f"Discovered {self.topic_assignments.shape[1]} topics using {model_description}")
        return self.documents
    
    def match_topics(
        self,
        topics: List[str],
        descriptions: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Match documents to predefined topics.
        
        Parameters
        ----------
        topics : List[str]
            List of topic names
        descriptions : Optional[List[str]], optional
            List of topic descriptions, by default None
        threshold : Optional[float], optional
            Similarity threshold for topic assignment, by default None
            If None, uses the value from config
        
        Returns
        -------
        pd.DataFrame
            DataFrame with topic assignments
        """
        if self.documents is None:
            raise ValueError("Documents must be preprocessed before topic matching")
        
        if self.document_embeddings is None:
            logger.info("Document embeddings not found, generating them now")
            self.embed_documents()
        
        # Set threshold from config if not provided
        if threshold is None:
            threshold = self.config.modeling.topic_matching.threshold
        
        # Store topics
        self.topics = topics
        
        # Generate topic embeddings
        logger.info(f"Generating embeddings for {len(topics)} topics")
        self.topic_embeddings = self.embedding_model.embed_topics(topics, descriptions)
        
        # Match documents to topics
        logger.info("Matching documents to topics")
        topic_matcher = TopicMatcher(
            threshold=threshold,
            assign_multiple=self.config.modeling.topic_matching.assign_multiple,
            max_topics_per_doc=self.config.modeling.topic_matching.max_topics_per_doc,
        )
        self.topic_assignments = topic_matcher.match(
            self.document_embeddings, self.topic_embeddings, topics
        )
        
        # Add topic assignments to documents DataFrame
        self.documents["topic"] = self.topic_assignments["primary_topic"]
        if "topic_probability" in self.topic_assignments.columns:
            self.documents["topic_probability"] = self.topic_assignments["topic_probability"]
        
        logger.info("Topic matching complete")
        return self.documents
    
    def visualize_embeddings(
        self,
        n_neighbors: Optional[int] = None,
        min_dist: Optional[float] = None,
        return_figure: bool = True,
    ):
        """Visualize document embeddings using UMAP.
        
        Parameters
        ----------
        n_neighbors : Optional[int], optional
            Number of neighbors for UMAP, by default None
            If None, uses the value from config
        min_dist : Optional[float], optional
            Minimum distance for UMAP, by default None
            If None, uses the value from config
        return_figure : bool, optional
            Whether to return the figure object, by default True
        
        Returns
        -------
        plotly.graph_objects.Figure or None
            Plotly figure if return_figure is True, else None
        """
        if self.document_embeddings is None:
            raise ValueError("Documents must be embedded before visualization")
        
        if self.documents is None or "topic" not in self.documents.columns:
            raise ValueError(
                "Topic assignment must be performed before visualization"
            )
        
        # Set UMAP parameters from config if not provided
        if n_neighbors is None:
            n_neighbors = self.config.visualization.umap.n_neighbors
        if min_dist is None:
            min_dist = self.config.visualization.umap.min_dist
        
        # Create UMAP projection if not already done
        if self.umap_projection is None:
            logger.info("Creating UMAP projection of document embeddings")
            self.umap_projection = create_umap_projection(
                self.document_embeddings,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=self.config.visualization.umap.n_components,
                metric=self.config.visualization.umap.metric,
            )
        
        # Create visualization
        logger.info("Creating interactive embedding visualization")
        fig = plot_embeddings(
            self.umap_projection,
            self.documents["topic"],
            document_texts=self.documents["text"],
            width=self.config.visualization.plots.width,
            height=self.config.visualization.plots.height,
            marker_size=self.config.visualization.plots.marker_size,
            opacity=self.config.visualization.plots.opacity,
            colorscale=self.config.visualization.plots.colorscale,
        )
        
        if return_figure:
            return fig
        else:
            fig.show()
    
    def visualize_topic_distribution(self, return_figure: bool = True):
        """Visualize topic distribution.
        
        Parameters
        ----------
        return_figure : bool, optional
            Whether to return the figure object, by default True
        
        Returns
        -------
        plotly.graph_objects.Figure or None
            Plotly figure if return_figure is True, else None
        """
        if self.documents is None or "topic" not in self.documents.columns:
            raise ValueError(
                "Topic assignment must be performed before visualization"
            )
        
        # Create visualization
        logger.info("Creating topic distribution visualization")
        fig = plot_topic_distribution(
            self.documents["topic"],
            width=self.config.visualization.plots.width,
            height=self.config.visualization.plots.height,
            colorscale=self.config.visualization.plots.colorscale,
        )
        
        if return_figure:
            return fig
        else:
            fig.show()
    
    def visualize_topic_trends(
        self,
        time_column: str,
        value_column: Optional[str] = None,
        time_interval: str = "M",
        normalize: bool = False,
        cumulative: bool = False,
        top_n_topics: Optional[int] = None,
        chart_type: str = "line",
        return_figure: bool = True,
    ):
        """Visualize topic trends over time.
        
        Parameters
        ----------
        time_column : str
            Name of column containing datetime values
        value_column : str, optional
            Name of column containing values to aggregate (if None, counts occurrences)
        time_interval : str, default "M"
            Time interval for aggregation: 'D' (daily), 'W' (weekly), 'M' (monthly),
            'Q' (quarterly), 'Y' (yearly)
        normalize : bool, default False
            If True, normalize values to show percentage distribution
        cumulative : bool, default False
            If True, show cumulative values over time
        top_n_topics : int, optional
            If provided, only show top N topics by overall frequency
        chart_type : str, default "line"
            Type of chart to create: "line", "heatmap", "area", "ridge", "calendar"
        return_figure : bool, optional
            Whether to return the figure object, by default True
        
        Returns
        -------
        plotly.graph_objects.Figure or None
            Plotly figure if return_figure is True, else None
        """
        if self.documents is None or "topic" not in self.documents.columns:
            raise ValueError(
                "Topic assignment must be performed before visualization"
            )
        
        if time_column not in self.documents.columns:
            raise ValueError(f"Time column '{time_column}' not found in documents")
        
        if value_column is not None and value_column not in self.documents.columns:
            raise ValueError(f"Value column '{value_column}' not found in documents")
        
        logger.info(f"Creating time series visualization with chart type: {chart_type}")
        
        # Choose the appropriate visualization based on chart_type
        if chart_type.lower() == "line":
            fig = create_topic_trend_plot(
                df=self.documents,
                topic_column="topic",
                time_column=time_column,
                value_column=value_column,
                time_interval=time_interval,
                normalize=normalize,
                cumulative=cumulative,
                top_n_topics=top_n_topics,
                height=self.config.visualization.plots.height,
                width=self.config.visualization.plots.width,
            )
        elif chart_type.lower() == "heatmap":
            fig = create_topic_heatmap(
                df=self.documents,
                topic_column="topic",
                time_column=time_column,
                value_column=value_column,
                time_interval=time_interval,
                normalize=normalize,
                top_n_topics=top_n_topics,
                height=self.config.visualization.plots.height,
                width=self.config.visualization.plots.width,
            )
        elif chart_type.lower() == "area":
            fig = create_topic_stacked_area(
                df=self.documents,
                topic_column="topic",
                time_column=time_column,
                value_column=value_column,
                time_interval=time_interval,
                normalize=normalize,
                top_n_topics=top_n_topics,
                height=self.config.visualization.plots.height,
                width=self.config.visualization.plots.width,
            )
        elif chart_type.lower() == "ridge":
            fig = create_topic_ridge_plot(
                df=self.documents,
                topic_column="topic",
                time_column=time_column,
                value_column=value_column,
                time_interval=time_interval,
                normalize=normalize,
                top_n_topics=top_n_topics,
                height=self.config.visualization.plots.height,
                width=self.config.visualization.plots.width,
            )
        elif chart_type.lower() == "calendar":
            if "topic" not in self.documents.columns:
                raise ValueError("Need a topic column to create a calendar visualization")
            
            # For calendar, we need a specific topic
            topics = self.documents["topic"].unique()
            if len(topics) > 0:
                topic_to_plot = topics[0]  # Default to first topic
                fig = create_topic_calendar_heatmap(
                    df=self.documents,
                    topic_column="topic",
                    time_column=time_column,
                    topic_to_plot=topic_to_plot,
                    value_column=value_column,
                    height=250,  # Calendar has a specific height ratio
                    width=self.config.visualization.plots.width,
                )
            else:
                raise ValueError("No topics found to visualize")
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        
        if return_figure:
            return fig
        else:
            fig.show()
    
    def visualize_geospatial_topics(
        self,
        lat_column: Optional[str] = "latitude",
        lon_column: Optional[str] = "longitude",
        region_column: Optional[str] = None,
        postcode_column: Optional[str] = None,
        value_column: Optional[str] = None,
        map_type: str = "point",
        color_by_topic: bool = True,
        geojson: Optional[Any] = None,
        feature_id_property: Optional[str] = None,
        return_figure: bool = True,
    ):
        """Visualize topics across geographic locations.
        
        Parameters
        ----------
        lat_column : str, optional
            Name of column containing latitude values, by default "latitude"
        lon_column : str, optional
            Name of column containing longitude values, by default "longitude"
        region_column : str, optional
            Name of column containing region identifiers, by default None
        postcode_column : str, optional
            Name of column containing postal/zip codes, by default None
        value_column : str, optional
            Name of column containing values to aggregate, by default None
        map_type : str, default "point"
            Type of map to create: "point", "choropleth", "density", "postcode"
        color_by_topic : bool, default True
            If True, color markers by topic; if False, color by value_column
        geojson : Any, optional
            GeoJSON object for choropleth maps, by default None
        feature_id_property : str, optional
            Property in GeoJSON features that matches region identifiers, by default None
        return_figure : bool, optional
            Whether to return the figure object, by default True
        
        Returns
        -------
        plotly.graph_objects.Figure or None
            Plotly figure if return_figure is True, else None
        """
        if self.documents is None or "topic" not in self.documents.columns:
            raise ValueError(
                "Topic assignment must be performed before visualization"
            )
        
        logger.info(f"Creating geospatial visualization with map type: {map_type}")
        
        # Choose the appropriate visualization based on map_type
        if map_type.lower() == "point":
            # Check if required columns exist
            for col in [lat_column, lon_column]:
                if col not in self.documents.columns:
                    raise ValueError(f"Required column '{col}' not found in documents")
            
            fig = create_topic_map(
                df=self.documents,
                topic_column="topic",
                lat_column=lat_column,
                lon_column=lon_column,
                value_column=value_column,
                color_by_topic=color_by_topic,
                height=self.config.visualization.plots.height,
                width=self.config.visualization.plots.width,
            )
        elif map_type.lower() == "choropleth":
            # Check if required columns and data exist
            if region_column is None:
                raise ValueError("Region column must be provided for choropleth maps")
            if region_column not in self.documents.columns:
                raise ValueError(f"Region column '{region_column}' not found in documents")
            if geojson is None:
                raise ValueError("GeoJSON must be provided for choropleth maps")
            if feature_id_property is None:
                raise ValueError("Feature ID property must be provided for choropleth maps")
            
            fig = create_region_choropleth(
                df=self.documents,
                topic_column="topic",
                region_column=region_column,
                geojson=geojson,
                feature_id_property=feature_id_property,
                value_column=value_column,
                height=self.config.visualization.plots.height,
                width=self.config.visualization.plots.width,
            )
        elif map_type.lower() == "density":
            # Check if required columns exist
            for col in [lat_column, lon_column]:
                if col not in self.documents.columns:
                    raise ValueError(f"Required column '{col}' not found in documents")
            
            fig = create_topic_density_map(
                df=self.documents,
                topic_column="topic",
                lat_column=lat_column,
                lon_column=lon_column,
                height=self.config.visualization.plots.height,
                width=self.config.visualization.plots.width,
            )
        elif map_type.lower() == "postcode":
            # Check if required columns exist
            if postcode_column is None:
                raise ValueError("Postcode column must be provided for postcode maps")
            if postcode_column not in self.documents.columns:
                raise ValueError(f"Postcode column '{postcode_column}' not found in documents")
            
            fig = create_postcode_map(
                df=self.documents,
                topic_column="topic",
                postcode_column=postcode_column,
                value_column=value_column,
                color_by_topic=color_by_topic,
                height=self.config.visualization.plots.height,
                width=self.config.visualization.plots.width,
            )
        else:
            raise ValueError(f"Unsupported map type: {map_type}")
        
        if return_figure:
            return fig
        else:
            fig.show()
    
    def visualize_timespace_topics(
        self,
        time_column: str,
        visualization_type: str = "animated_map",
        lat_column: Optional[str] = "latitude",
        lon_column: Optional[str] = "longitude",
        region_column: Optional[str] = None,
        category_column: Optional[str] = None,
        value_column: Optional[str] = None,
        time_interval: str = "M",
        normalize: bool = False,
        plot_type: str = "line",
        return_figure: bool = True,
    ):
        """Visualize topics across both time and space dimensions.
        
        Parameters
        ----------
        time_column : str
            Name of column containing datetime values
        visualization_type : str, default "animated_map"
            Type of visualization: "animated_map", "space_time_heatmap", "category_time"
        lat_column : str, optional
            Name of column containing latitude values, by default "latitude"
        lon_column : str, optional
            Name of column containing longitude values, by default "longitude"
        region_column : str, optional
            Name of column containing region identifiers, by default None
        category_column : str, optional
            Name of column containing category labels, by default None
        value_column : str, optional
            Name of column containing values to aggregate, by default None
        time_interval : str, default "M"
            Time interval for aggregation: 'D', 'W', 'M', 'Q', 'Y'
        normalize : bool, default False
            If True, normalize values to show percentage distribution
        plot_type : str, default "line"
            Type of plot for category_time visualization: "line", "area", "bar"
        return_figure : bool, optional
            Whether to return the figure object, by default True
        
        Returns
        -------
        plotly.graph_objects.Figure or None
            Plotly figure if return_figure is True, else None
        """
        if self.documents is None or "topic" not in self.documents.columns:
            raise ValueError(
                "Topic assignment must be performed before visualization"
            )
        
        if time_column not in self.documents.columns:
            raise ValueError(f"Time column '{time_column}' not found in documents")
        
        logger.info(f"Creating time-space visualization: {visualization_type}")
        
        # Choose the appropriate visualization based on visualization_type
        if visualization_type.lower() == "animated_map":
            # Check if required columns exist
            for col in [lat_column, lon_column]:
                if col not in self.documents.columns:
                    raise ValueError(f"Required column '{col}' not found in documents")
            
            fig = create_animated_map(
                df=self.documents,
                topic_column="topic",
                time_column=time_column,
                lat_column=lat_column,
                lon_column=lon_column,
                value_column=value_column,
                time_interval=time_interval,
                height=self.config.visualization.plots.height,
                width=self.config.visualization.plots.width,
            )
        elif visualization_type.lower() == "space_time_heatmap":
            # Check if required columns exist
            if region_column is None:
                raise ValueError("Region column must be provided for space-time heatmaps")
            if region_column not in self.documents.columns:
                raise ValueError(f"Region column '{region_column}' not found in documents")
            
            fig = create_space_time_heatmap(
                df=self.documents,
                topic_column="topic",
                time_column=time_column,
                region_column=region_column,
                value_column=value_column,
                time_interval=time_interval,
                normalize=normalize,
                height=self.config.visualization.plots.height,
                width=self.config.visualization.plots.width,
            )
        elif visualization_type.lower() == "category_time":
            # Check if required columns exist
            if category_column is None:
                raise ValueError("Category column must be provided for category-time plots")
            if category_column not in self.documents.columns:
                raise ValueError(f"Category column '{category_column}' not found in documents")
            
            fig = create_category_time_plot(
                df=self.documents,
                topic_column="topic",
                time_column=time_column,
                category_column=category_column,
                value_column=value_column,
                time_interval=time_interval,
                plot_type=plot_type,
                normalize=normalize,
                height=self.config.visualization.plots.height,
                width=self.config.visualization.plots.width,
            )
        else:
            raise ValueError(f"Unsupported visualization type: {visualization_type}")
        
        if return_figure:
            return fig
        else:
            fig.show()
    
    def generate_report(
        self,
        output_path: Optional[Union[str, Path]] = None,
        include_interactive: Optional[bool] = None,
        title: Optional[str] = None,
        include_raw_data: Optional[bool] = None,
        max_examples_per_topic: Optional[int] = None,
        max_samples_per_topic: Optional[int] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        topic_words: Optional[Dict[str, Dict[str, float]]] = None,
        open_browser: Optional[bool] = None,
    ) -> str:
        """Generate an HTML report of the topic modeling results.
        
        Parameters
        ----------
        output_path : Optional[Union[str, Path]], optional
            Path to save the report, by default None
            If None, creates a file in the current directory
        include_interactive : Optional[bool], optional
            Whether to include interactive visualizations, by default None
            If None, uses the value from config
        title : Optional[str], optional
            Title for the report, by default None
            If None, uses the value from config
        include_raw_data : Optional[bool], optional
            Whether to include raw data table in the report, by default None
            If None, uses the value from config
        max_examples_per_topic : Optional[int], optional
            Maximum number of example documents per topic, by default None
            If None, uses the value from config
        max_samples_per_topic : Optional[int], optional
            Maximum number of samples per topic in the raw data table, by default None
            If None, uses the value from config (default is 5)
        similarity_matrix : Optional[np.ndarray], optional
            Matrix of topic similarities, shape (n_topics, n_topics), by default None
        topic_words : Optional[Dict[str, Dict[str, float]]], optional
            Dictionary mapping topic names to word frequency dictionaries, by default None
        open_browser : Optional[bool], optional
            Whether to automatically open the report in a web browser, by default None
            If None, uses the default value (False)
        
        Returns
        -------
        str
            Path to the generated report
        """
        from .reporting import generate_html_report
        
        if self.documents is None or "topic" not in self.documents.columns:
            raise ValueError(
                "Topic assignment must be performed before report generation"
            )
        
        # Build report config from parameters and default config
        report_config = self.config.reporting.html.dict()
        
        # Override with provided parameters
        if include_interactive is not None:
            report_config["include_interactive"] = include_interactive
        if title is not None:
            report_config["title"] = title
        if include_raw_data is not None:
            report_config["include_raw_data"] = include_raw_data
        if max_examples_per_topic is not None:
            report_config["max_examples_per_topic"] = max_examples_per_topic
        if max_samples_per_topic is not None:
            report_config["max_samples_per_topic"] = max_samples_per_topic
        
        # Create output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"meno_report_{timestamp}.html"
        
        # Generate report
        logger.info(f"Generating HTML report at {output_path}")
        
        # Create visualizations for the report
        if self.document_embeddings is not None and self.umap_projection is None:
            logger.info("Creating UMAP projection for the report")
            self.umap_projection = create_umap_projection(
                self.document_embeddings,
                n_neighbors=self.config.visualization.umap.n_neighbors,
                min_dist=self.config.visualization.umap.min_dist,
                n_components=self.config.visualization.umap.n_components,
                metric=self.config.visualization.umap.metric,
            )
        
        # Generate topic words if not provided but we have processed texts
        if topic_words is None and "processed_text" in self.documents.columns:
            logger.info("Generating topic word frequencies for the report")
            topic_words = {}
            
            # First pass: calculate TF-IDF style word importance
            # Get all documents
            all_docs = self.documents["processed_text"]
            
            # Get total word counts across all documents
            all_words = " ".join(all_docs).lower().split()
            total_word_counts = {}
            for word in set(all_words):
                if len(word) > 3:  # Only include words with more than 3 characters
                    total_word_counts[word] = all_words.count(word)
            
            # Now calculate topic-specific word frequencies with TF-IDF influence
            for topic in self.documents["topic"].unique():
                # Get documents for this topic
                topic_docs = self.documents[self.documents["topic"] == topic]["processed_text"]
                
                # Create a word frequency dictionary 
                topic_words_text = " ".join(topic_docs).lower().split()
                word_freq = {}
                
                # Filter to unique words with minimum length
                unique_words = set([w for w in topic_words_text if len(w) > 3])
                
                for word in unique_words:
                    # Count in this topic
                    topic_count = topic_words_text.count(word)
                    
                    # Calculate a TF-IDF style score to emphasize distinctive words
                    # Higher score for words that appear more in this topic than overall
                    if word in total_word_counts:
                        # Calculate the word's importance for this topic vs all topics
                        # Higher values = more distinctive to this topic
                        distinctiveness = (topic_count / len(topic_words_text)) / (total_word_counts[word] / len(all_words))
                        word_freq[word] = topic_count * distinctiveness
                
                topic_words[topic] = word_freq
        
        # Add metadata about modeling method for the report
        metadata = {}
        if hasattr(self, '_last_modeling_method'):
            metadata['modeling_method'] = self._last_modeling_method
            if self._last_modeling_method == 'embedding_cluster':
                metadata['clustering_algorithm'] = self.config.modeling.clustering.algorithm
                metadata['n_clusters'] = num_topics if 'num_topics' in locals() else self.config.modeling.clustering.n_clusters
        
        # Call the report generator with all the data
        report_path = generate_html_report(
            documents=self.documents,
            topic_assignments=self.topic_assignments,
            umap_projection=self.umap_projection if report_config["include_interactive"] else None,
            output_path=output_path,
            config=report_config,
            similarity_matrix=similarity_matrix,
            topic_words=topic_words,
            metadata=metadata,
            open_browser=open_browser if open_browser is not None else False,
        )
        
        logger.info(f"Report generated at {report_path}")
        return report_path
    
    def export_results(
        self,
        output_path: Optional[Union[str, Path]] = None,
        formats: Optional[List[str]] = None,
        include_embeddings: Optional[bool] = None,
    ) -> Dict[str, str]:
        """Export topic modeling results to files.
        
        Parameters
        ----------
        output_path : Optional[Union[str, Path]], optional
            Path to save the exports, by default None
            If None, creates files in the current directory
        formats : Optional[List[str]], optional
            Formats to export, by default None
            If None, uses the value from config
        include_embeddings : Optional[bool], optional
            Whether to include embeddings in the export, by default None
            If None, uses the value from config
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping format to file path
        """
        if self.documents is None or "topic" not in self.documents.columns:
            raise ValueError(
                "Topic assignment must be performed before export"
            )
        
        # Set parameters from config if not provided
        if formats is None:
            formats = self.config.reporting.export.formats
        if include_embeddings is None:
            include_embeddings = self.config.reporting.export.include_embeddings
        
        # Create output directory if not exists
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"meno_results_{timestamp}"
        
        output_path = Path(output_path)
        os.makedirs(output_path, exist_ok=True)
        
        # Prepare export data
        export_data = self.documents.copy()
        
        # Add embeddings if requested
        if include_embeddings and self.document_embeddings is not None:
            for i in range(self.document_embeddings.shape[1]):
                export_data[f"embedding_{i}"] = self.document_embeddings[:, i]
        
        # Export in each format
        logger.info(f"Exporting results in formats: {formats}")
        export_paths = {}
        
        for fmt in formats:
            file_path = output_path / f"topic_results.{fmt}"
            
            if fmt == "csv":
                export_data.to_csv(file_path, index=False)
            elif fmt == "json":
                export_data.to_json(file_path, orient="records", indent=2)
            elif fmt == "excel":
                export_data.to_excel(file_path, index=False)
            elif fmt == "pickle":
                export_data.to_pickle(file_path)
                
            export_paths[fmt] = str(file_path)
            logger.info(f"Exported to {file_path}")
        
        return export_paths