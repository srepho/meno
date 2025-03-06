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
from .visualization import plot_embeddings, plot_topic_distribution, create_umap_projection

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
    ):
        """Initialize the topic modeler with configuration."""
        # Load configuration
        self.config = load_config(config_path)
        
        # Apply overrides if provided
        if config_overrides:
            self.config = merge_configs(self.config, config_overrides)
            
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
        
        # Initialize embedding model
        self.embedding_model = DocumentEmbedding(
            model_name=self.config.modeling.embeddings.model_name,
            batch_size=self.config.modeling.embeddings.batch_size,
        )
        
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
        num_topics: Optional[int] = None,
    ) -> pd.DataFrame:
        """Discover topics in an unsupervised manner.
        
        Parameters
        ----------
        method : str, optional
            Method to use for topic discovery, by default "embedding_cluster"
            Options: "lda", "embedding_cluster"
        num_topics : Optional[int], optional
            Number of topics to discover, by default None
            If None, uses the value from config
        
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
        
        # Set num_topics from config if not provided
        if num_topics is None:
            if method == "lda":
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
                f"Discovering topics using embedding clustering with algorithm={self.config.modeling.clustering.algorithm}"
            )
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
        
        # Add topic assignments to documents DataFrame
        if "topic" not in self.topic_assignments.columns:
            # For clustering methods that return a single topic per document
            self.documents["topic"] = self.topic_assignments.idxmax(axis=1)
        else:
            # For methods that already assigned a topic column
            self.documents["topic"] = self.topic_assignments["topic"]
        
        logger.info(f"Discovered {self.topic_assignments.shape[1]} topics")
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
    
    def generate_report(
        self,
        output_path: Optional[Union[str, Path]] = None,
        include_interactive: Optional[bool] = None,
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
        
        # Set include_interactive from config if not provided
        if include_interactive is None:
            include_interactive = self.config.reporting.html.include_interactive
        
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
        
        # Call the report generator with all the data
        report_path = generate_html_report(
            documents=self.documents,
            topic_assignments=self.topic_assignments,
            umap_projection=self.umap_projection if include_interactive else None,
            output_path=output_path,
            config=self.config.reporting.html,
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