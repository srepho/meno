"""Workflow module for Meno topic modeling toolkit.

This module provides a guided workflow for topic modeling, including:
1. Acronym detection and expansion
2. Spelling correction
3. Topic modeling (LDA or embedding-based)
4. Visualization and reporting
"""

from typing import Dict, List, Optional, Union, Tuple, Callable, Any
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import logging
import tempfile
import webbrowser
from datetime import datetime

from .preprocessing.acronyms import AcronymExpander
from .preprocessing.spelling import SpellingCorrector
from .preprocessing.normalization import TextNormalizer
from .modeling import DocumentEmbedding, LDAModel, EmbeddingClusterModel
from .visualization import (
    plot_embeddings, plot_topic_distribution, create_umap_projection,
    # Time series visualizations
    create_topic_trend_plot, create_topic_heatmap, create_topic_stacked_area,
    # Geospatial visualizations
    create_topic_map, create_region_choropleth, create_topic_density_map,
    # Time-space visualizations
    create_animated_map, create_space_time_heatmap, create_category_time_plot
)
from .reporting.html_generator import generate_html_report
from .meno import MenoTopicModeler
from .utils.config import (
    load_config, save_config, merge_configs, 
    WorkflowMenoConfig, MenoConfig
)

# Set up logging
logger = logging.getLogger(__name__)


class MenoWorkflow:
    """Interactive workflow for topic modeling with user feedback steps.
    
    This class extends the MenoTopicModeler with a guided, interactive workflow
    that includes preprocessing suggestions, acronym expansion, spelling correction,
    and visualization.
    
    Parameters
    ----------
    modeler : Optional[MenoTopicModeler], optional
        Existing MenoTopicModeler instance, by default None.
        If None, a new instance will be created.
    config_path : Optional[Union[str, Path]], optional
        Path to configuration file, by default None.
    config_overrides : Optional[Dict[str, Any]], optional
        Dictionary of configuration overrides, by default None.
    
    Attributes
    ----------
    modeler : MenoTopicModeler
        The underlying topic modeler instance
    acronym_expander : AcronymExpander
        Acronym detection and expansion component
    spelling_corrector : SpellingCorrector
        Spelling correction component
    """
    
    def __init__(
        self,
        modeler: Optional[MenoTopicModeler] = None,
        config_path: Optional[Union[str, Path]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        local_model_path: Optional[str] = None,
        local_files_only: bool = False,
        offline_mode: bool = False,
        auto_create_config: bool = True,
    ):
        """Initialize the workflow with an optional existing modeler instance."""
        # Load workflow configuration
        self.config = load_config(
            config_path=config_path,
            config_type="workflow",
            auto_create=auto_create_config
        )
        
        # Apply overrides if provided
        if config_overrides:
            self.config = merge_configs(self.config, config_overrides)
        
        # Create embedding model with local settings if requested
        embedding_model = None
        if local_model_path or local_files_only:
            from meno.modeling.embeddings import DocumentEmbedding
            embedding_model = DocumentEmbedding(
                local_model_path=local_model_path,
                local_files_only=local_files_only
            )
            
        # Add offline_mode to config overrides if specified
        if offline_mode:
            config_overrides = config_overrides or {}
            config_overrides['offline_mode'] = offline_mode
        
        # Set up modeler
        self.modeler = modeler or MenoTopicModeler(
            config_path=config_path,
            config_overrides=config_overrides,
            embedding_model=embedding_model,
            offline_mode=offline_mode
        )
        
        # Initialize components based on configuration
        acronym_config = self.config.preprocessing.acronyms
        self.acronym_expander = AcronymExpander(
            custom_mappings=acronym_config.custom_mappings
        )
        
        spelling_config = self.config.preprocessing.spelling
        self.spelling_corrector = SpellingCorrector(
            dictionary=spelling_config.custom_dictionary,
            min_word_length=spelling_config.min_word_length,
            max_distance=spelling_config.max_distance
        )
        
        # Data storage
        self.documents = None
        self.text_column = None
        self.time_column = None
        self.geo_column = None
        self.category_column = None
        
        # Processing state
        self.acronyms_expanded = False
        self.spelling_corrected = False
        self.preprocessing_complete = False
        self.modeling_complete = False
        
    def load_data(
        self,
        data: Union[pd.DataFrame, str, Path],
        text_column: str,
        time_column: Optional[str] = None,
        geo_column: Optional[str] = None,
        category_column: Optional[str] = None,
        deduplicate: bool = False,
    ) -> pd.DataFrame:
        """Load data from a DataFrame or file path.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, str, Path]
            Data to load, either a DataFrame or a path to a CSV file.
        text_column : str
            Name of the column containing text data.
        time_column : Optional[str], optional
            Name of the column containing time/date data, by default None.
        geo_column : Optional[str], optional
            Name of the column containing geographic data, by default None.
        category_column : Optional[str], optional
            Name of the column containing category data, by default None.
        deduplicate : bool, optional
            Whether to deduplicate documents before processing, by default False.
            When True, only unique documents are passed to the topic modeling algorithm,
            and the resulting topics are then joined back to all documents.
            
        Returns
        -------
        pd.DataFrame
            The loaded data.
        """
        # Load data from file if needed
        if isinstance(data, (str, Path)):
            self.documents = pd.read_csv(data)
        else:
            self.documents = data.copy()
            
        # Deduplicate if requested
        self.original_documents = None
        self.doc_hash_column = None
        
        if deduplicate:
            logger.info("Deduplicating documents before processing")
            # Create a hash of each document text for deduplication and later joining
            self.documents['doc_hash'] = self.documents[text_column].apply(lambda x: hash(str(x)))
            self.doc_hash_column = 'doc_hash'
            
            # Store original dataset
            self.original_documents = self.documents.copy()
            
            # Deduplicate based on text content
            self.documents = self.documents.drop_duplicates(subset=['doc_hash'])
            
            logger.info(f"Reduced dataset from {len(self.original_documents)} to {len(self.documents)} unique documents")
        
        # Store column names
        self.text_column = text_column
        self.time_column = time_column
        self.geo_column = geo_column
        self.category_column = category_column
        self.deduplicated = deduplicate
        
        # Validate data
        if self.text_column not in self.documents.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in data.")
            
        if self.time_column and self.time_column not in self.documents.columns:
            raise ValueError(f"Time column '{self.time_column}' not found in data.")
            
        if self.geo_column and self.geo_column not in self.documents.columns:
            raise ValueError(f"Geographic column '{self.geo_column}' not found in data.")
            
        if self.category_column and self.category_column not in self.documents.columns:
            raise ValueError(f"Category column '{self.category_column}' not found in data.")
            
        logger.info(f"Loaded data with {len(self.documents)} documents.")
        return self.documents
    
    def detect_acronyms(
        self,
        min_length: int = 2,
        min_count: int = 5,
        limit: int = 20,
    ) -> Dict[str, int]:
        """Detect potential acronyms in the text data.
        
        Parameters
        ----------
        min_length : int, optional
            Minimum length for an acronym, by default 2.
        min_count : int, optional
            Minimum count for an acronym to be included, by default 5.
        limit : int, optional
            Maximum number of acronyms to return, by default 20.
            
        Returns
        -------
        Dict[str, int]
            Dictionary of acronyms and their counts.
        """
        if self.documents is None or self.text_column is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Initialize acronym expander with appropriate settings
        self.acronym_expander = AcronymExpander(min_length=min_length)
        
        # Detect acronyms
        acronym_counts = self.acronym_expander.extract_acronyms_batch(
            self.documents[self.text_column]
        )
        
        # Filter by minimum count and sort by frequency
        filtered_counts = {
            k: v for k, v in acronym_counts.items() 
            if v >= min_count and k not in self.acronym_expander.acronym_dict
        }
        
        # Sort by count (descending)
        sorted_counts = dict(
            sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        )
        
        logger.info(f"Detected {len(sorted_counts)} potential acronyms.")
        return sorted_counts
    
    def generate_acronym_report(
        self,
        min_length: Optional[int] = None,
        min_count: Optional[int] = None,
        limit: Optional[int] = None,
        output_path: Optional[str] = None,
        open_browser: Optional[bool] = None,
    ) -> str:
        """Generate an HTML report of potential acronyms for review.
        
        Parameters
        ----------
        min_length : Optional[int], optional
            Minimum length for an acronym, by default None.
            If None, uses the value from config.
        min_count : Optional[int], optional
            Minimum count for an acronym to be included, by default None.
            If None, uses the value from config.
        limit : Optional[int], optional
            Maximum number of acronyms to return, by default None.
            If None, uses the value from config.
        output_path : Optional[str], optional
            Path to save the report, by default None.
            If None, uses the path from config.
        open_browser : Optional[bool], optional
            Whether to open the report in a browser, by default None.
            If None, uses the value from config.
            
        Returns
        -------
        str
            Path to the generated report.
        """
        # Get values from config if not provided
        if min_length is None:
            min_length = self.config.workflow.interactive.min_acronym_length
        if min_count is None:
            min_count = self.config.workflow.interactive.min_acronym_count
        if limit is None:
            limit = self.config.workflow.interactive.max_acronyms
        if output_path is None:
            output_path = self.config.workflow.report_paths.acronym_report
        if open_browser is None:
            open_browser = self.config.workflow.features.auto_open_browser
            
        # Detect acronyms
        acronym_counts = self.detect_acronyms(
            min_length=min_length,
            min_count=min_count,
            limit=limit
        )
        
        # Generate suggestions for each acronym
        acronym_suggestions = {}
        for acronym in acronym_counts:
            suggestions = self.acronym_expander.suggest_expansions(acronym, min_score=60)
            acronym_suggestions[acronym] = suggestions if suggestions else []
        
        # Get sample contexts
        acronym_contexts = {}
        for acronym in acronym_counts:
            # Get up to 3 sample texts containing the acronym
            samples = []
            pattern = r'\b' + acronym + r'\b'
            for text in self.documents[self.text_column]:
                if pd.isna(text):
                    continue
                if len(samples) >= 3:
                    break
                if re.search(pattern, text):
                    # Truncate long texts
                    if len(text) > 200:
                        text = text[:200] + "..."
                    samples.append(text)
            acronym_contexts[acronym] = samples
        
        # Generate a simple HTML report
        html_content = self._generate_acronym_html_report(
            acronym_counts, acronym_suggestions, acronym_contexts
        )
        
        # Save to file
        if output_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
                output_path = f.name
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Open in browser if requested
        if open_browser:
            webbrowser.open(f'file://{os.path.abspath(output_path)}')
        
        logger.info(f"Acronym report generated at {output_path}")
        return output_path
    
    def _generate_acronym_html_report(
        self,
        acronym_counts: Dict[str, int],
        acronym_suggestions: Dict[str, List[Tuple[str, int]]],
        acronym_contexts: Dict[str, List[str]],
    ) -> str:
        """Generate HTML report for acronym review.
        
        Parameters
        ----------
        acronym_counts : Dict[str, int]
            Dictionary of acronyms and their counts.
        acronym_suggestions : Dict[str, List[Tuple[str, int]]]
            Dictionary of acronyms and their suggested expansions.
        acronym_contexts : Dict[str, List[str]]
            Dictionary of acronyms and sample contexts.
            
        Returns
        -------
        str
            HTML content for the report.
        """
        import html
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Meno Acronym Detection Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }
                h1 {
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }
                h2 {
                    color: #3498db;
                    margin-top: 30px;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 30px;
                }
                th, td {
                    text-align: left;
                    padding: 12px;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                    color: #2c3e50;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .context {
                    font-style: italic;
                    color: #7f8c8d;
                    margin-bottom: 5px;
                    font-size: 0.9em;
                }
                .count {
                    font-weight: bold;
                    color: #e74c3c;
                }
                .suggestion {
                    color: #27ae60;
                }
                .score {
                    color: #7f8c8d;
                    font-size: 0.8em;
                }
                .instructions {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
            </style>
        </head>
        <body>
            <h1>Meno Acronym Detection Report</h1>
            
            <div class="instructions">
                <p>This report shows potential acronyms detected in your text data. Review the suggestions and contexts to determine which acronyms to expand in your text processing.</p>
                <p>To use these findings:</p>
                <ol>
                    <li>Review the acronyms and their potential meanings</li>
                    <li>Use the sample contexts to verify if they are true acronyms</li>
                    <li>Create a custom mapping of acronyms to expansions</li>
                    <li>Pass this mapping to the AcronymExpander in your workflow</li>
                </ol>
            </div>
            
            <h2>Detected Acronyms</h2>
            <p>Found ${len(acronym_counts)} potential acronyms in the dataset.</p>
            
            <table>
                <tr>
                    <th>Acronym</th>
                    <th>Count</th>
                    <th>Suggestions</th>
                    <th>Sample Contexts</th>
                </tr>
        """
        
        # Add rows for each acronym
        for acronym, count in acronym_counts.items():
            # Format suggestions
            suggestions_html = ""
            if acronym_suggestions[acronym]:
                for suggestion, score in acronym_suggestions[acronym]:
                    suggestions_html += f'<div class="suggestion">{html.escape(suggestion)} <span class="score">({score}%)</span></div>'
            else:
                suggestions_html = "<em>No suggestions available</em>"
            
            # Format contexts
            contexts_html = ""
            if acronym_contexts[acronym]:
                for context in acronym_contexts[acronym]:
                    contexts_html += f'<div class="context">{html.escape(context)}</div>'
            else:
                contexts_html = "<em>No context examples available</em>"
            
            # Add row
            html_content += f"""
                <tr>
                    <td>{html.escape(acronym)}</td>
                    <td class="count">{count}</td>
                    <td>{suggestions_html}</td>
                    <td>{contexts_html}</td>
                </tr>
            """
        
        # Close HTML
        html_content += """
            </table>
            
            <h2>Next Steps</h2>
            <p>After reviewing this report, you can:</p>
            <ol>
                <li>Add custom acronym mappings to your workflow</li>
                <li>Proceed to spelling correction</li>
                <li>Run the topic modeling process</li>
            </ol>
            
            <p><small>Generated by Meno Topic Modeling Toolkit</small></p>
        </body>
        </html>
        """
        
        return html_content
    
    def expand_acronyms(
        self,
        custom_mappings: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Expand acronyms in the text data.
        
        Parameters
        ----------
        custom_mappings : Optional[Dict[str, str]], optional
            Custom acronym mappings, by default None.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with expanded acronyms in the text column.
        """
        if self.documents is None or self.text_column is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Add custom mappings to acronym expander
        if custom_mappings:
            self.acronym_expander.add_acronyms(custom_mappings)
            
        # Expand acronyms
        self.documents[self.text_column] = self.acronym_expander.expand_acronyms_batch(
            self.documents[self.text_column]
        )
        
        self.acronyms_expanded = True
        logger.info("Acronyms expanded in text data.")
        return self.documents
    
    def detect_potential_misspellings(
        self,
        min_length: int = 4,
        min_count: int = 3,
        limit: int = 20,
    ) -> Dict[str, List[Tuple[str, int]]]:
        """Detect potential misspellings in the text data.
        
        Parameters
        ----------
        min_length : int, optional
            Minimum length for words to check, by default 4.
        min_count : int, optional
            Minimum count for words to check, by default 3.
        limit : int, optional
            Maximum number of potential misspellings to return, by default 20.
            
        Returns
        -------
        Dict[str, List[Tuple[str, int]]]
            Dictionary of potential misspellings and suggested corrections.
        """
        if self.documents is None or self.text_column is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Initialize spelling corrector
        self.spelling_corrector = SpellingCorrector(min_word_length=min_length)
        
        # Extract all words from the text
        import re
        all_words = []
        for text in self.documents[self.text_column]:
            if pd.isna(text):
                continue
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        # Count word frequencies
        from collections import Counter
        word_counts = Counter(all_words)
        
        # Filter rare and short words
        filtered_words = {
            word: count for word, count in word_counts.items() 
            if count >= min_count and len(word) >= min_length
        }
        
        # Get corrections for each word
        potential_misspellings = {}
        for word, count in filtered_words.items():
            correction = self.spelling_corrector.correct_word(word)
            if correction != word:
                # Get similarity score
                from thefuzz import fuzz
                score = fuzz.ratio(word, correction)
                
                potential_misspellings[word] = [(correction, score)]
        
        # Sort by count and limit
        sorted_misspellings = {
            k: v for k, v in sorted(
                potential_misspellings.items(),
                key=lambda x: word_counts[x[0]],
                reverse=True
            )[:limit]
        }
        
        logger.info(f"Detected {len(sorted_misspellings)} potential misspellings.")
        return sorted_misspellings
    
    def generate_misspelling_report(
        self,
        min_length: Optional[int] = None,
        min_count: Optional[int] = None,
        limit: Optional[int] = None,
        output_path: Optional[str] = None,
        open_browser: Optional[bool] = None,
    ) -> str:
        """Generate an HTML report of potential misspellings for review.
        
        Parameters
        ----------
        min_length : Optional[int], optional
            Minimum length for words to check, by default None.
            If None, uses the value from config.
        min_count : Optional[int], optional
            Minimum count for words to check, by default None.
            If None, uses the value from config.
        limit : Optional[int], optional
            Maximum number of potential misspellings to return, by default None.
            If None, uses the value from config.
        output_path : Optional[str], optional
            Path to save the report, by default None.
            If None, uses the path from config.
        open_browser : Optional[bool], optional
            Whether to open the report in a browser, by default None.
            If None, uses the value from config.
            
        Returns
        -------
        str
            Path to the generated report.
        """
        # Get values from config if not provided
        if min_length is None:
            min_length = self.config.workflow.interactive.min_word_length
        if min_count is None:
            min_count = self.config.workflow.interactive.min_word_count
        if limit is None:
            limit = self.config.workflow.interactive.max_misspellings
        if output_path is None:
            output_path = self.config.workflow.report_paths.spelling_report
        if open_browser is None:
            open_browser = self.config.workflow.features.auto_open_browser
            
        # Detect misspellings
        misspellings = self.detect_potential_misspellings(
            min_length=min_length,
            min_count=min_count,
            limit=limit
        )
        
        # Get sample contexts
        import re
        misspelling_contexts = {}
        for word in misspellings:
            # Get up to 3 sample texts containing the word
            samples = []
            pattern = r'\b' + word + r'\b'
            for text in self.documents[self.text_column]:
                if pd.isna(text):
                    continue
                if len(samples) >= 3:
                    break
                if re.search(pattern, text, re.IGNORECASE):
                    # Truncate long texts
                    if len(text) > 200:
                        text = text[:200] + "..."
                    samples.append(text)
            misspelling_contexts[word] = samples
        
        # Generate the HTML report
        html_content = self._generate_misspelling_html_report(
            misspellings, misspelling_contexts
        )
        
        # Save to file
        if output_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
                output_path = f.name
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Open in browser if requested
        if open_browser:
            webbrowser.open(f'file://{os.path.abspath(output_path)}')
        
        logger.info(f"Misspelling report generated at {output_path}")
        return output_path
    
    def _generate_misspelling_html_report(
        self,
        misspellings: Dict[str, List[Tuple[str, int]]],
        misspelling_contexts: Dict[str, List[str]],
    ) -> str:
        """Generate HTML report for misspelling review.
        
        Parameters
        ----------
        misspellings : Dict[str, List[Tuple[str, int]]]
            Dictionary of potential misspellings and suggested corrections.
        misspelling_contexts : Dict[str, List[str]]
            Dictionary of potential misspellings and sample contexts.
            
        Returns
        -------
        str
            HTML content for the report.
        """
        import html
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Meno Spelling Correction Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }
                h1 {
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }
                h2 {
                    color: #3498db;
                    margin-top: 30px;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 30px;
                }
                th, td {
                    text-align: left;
                    padding: 12px;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                    color: #2c3e50;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .context {
                    font-style: italic;
                    color: #7f8c8d;
                    margin-bottom: 5px;
                    font-size: 0.9em;
                }
                .misspelled {
                    font-weight: bold;
                    color: #e74c3c;
                }
                .correction {
                    color: #27ae60;
                    font-weight: bold;
                }
                .score {
                    color: #7f8c8d;
                    font-size: 0.8em;
                }
                .instructions {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
            </style>
        </head>
        <body>
            <h1>Meno Spelling Correction Report</h1>
            
            <div class="instructions">
                <p>This report shows potential spelling errors detected in your text data. Review the suggestions and contexts to determine which corrections to apply.</p>
                <p>To use these findings:</p>
                <ol>
                    <li>Review the potential misspellings and their suggested corrections</li>
                    <li>Use the sample contexts to verify if they are true misspellings</li>
                    <li>Create a custom dictionary of corrections</li>
                    <li>Pass this dictionary to the SpellingCorrector in your workflow</li>
                </ol>
            </div>
            
            <h2>Detected Potential Misspellings</h2>
            <p>Found ${len(misspellings)} potential misspellings in the dataset.</p>
            
            <table>
                <tr>
                    <th>Potential Misspelling</th>
                    <th>Suggested Correction</th>
                    <th>Sample Contexts</th>
                </tr>
        """
        
        # Add rows for each misspelling
        for word, corrections in misspellings.items():
            # Format corrections
            correction_html = ""
            if corrections:
                for correction, score in corrections:
                    correction_html += f'<div><span class="correction">{html.escape(correction)}</span> <span class="score">({score}% match)</span></div>'
            else:
                correction_html = "<em>No corrections available</em>"
            
            # Format contexts
            contexts_html = ""
            if misspelling_contexts[word]:
                for context in misspelling_contexts[word]:
                    # Highlight the misspelled word
                    highlighted_context = re.sub(
                        r'\b' + re.escape(word) + r'\b',
                        f'<span class="misspelled">{word}</span>',
                        html.escape(context),
                        flags=re.IGNORECASE
                    )
                    contexts_html += f'<div class="context">{highlighted_context}</div>'
            else:
                contexts_html = "<em>No context examples available</em>"
            
            # Add row
            html_content += f"""
                <tr>
                    <td>{html.escape(word)}</td>
                    <td>{correction_html}</td>
                    <td>{contexts_html}</td>
                </tr>
            """
        
        # Close HTML
        html_content += """
            </table>
            
            <h2>Next Steps</h2>
            <p>After reviewing this report, you can:</p>
            <ol>
                <li>Add custom spelling corrections to your workflow</li>
                <li>Proceed to topic modeling</li>
                <li>Generate visualizations and reports</li>
            </ol>
            
            <p><small>Generated by Meno Topic Modeling Toolkit</small></p>
        </body>
        </html>
        """
        
        return html_content
    
    def correct_spelling(
        self,
        custom_corrections: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Correct spelling errors in the text data.
        
        Parameters
        ----------
        custom_corrections : Optional[Dict[str, str]], optional
            Custom spelling corrections, by default None.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with corrected spelling in the text column.
        """
        if self.documents is None or self.text_column is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Add custom corrections
        if custom_corrections:
            self.spelling_corrector.add_corrections(custom_corrections)
            
        # Correct spelling
        self.documents[self.text_column] = self.spelling_corrector.correct_texts(
            self.documents[self.text_column]
        )
        
        self.spelling_corrected = True
        logger.info("Spelling corrected in text data.")
        return self.documents
    
    def preprocess_documents(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        custom_stopwords: Optional[List[str]] = None,
        additional_stopwords: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Preprocess text data for topic modeling.
        
        Parameters
        ----------
        lowercase : bool, optional
            Whether to convert text to lowercase, by default True.
        remove_punctuation : bool, optional
            Whether to remove punctuation, by default True.
        remove_numbers : bool, optional
            Whether to remove numbers, by default False.
        remove_stopwords : bool, optional
            Whether to remove stopwords, by default True.
        lemmatize : bool, optional
            Whether to lemmatize words, by default True.
        custom_stopwords : Optional[List[str]], optional
            Custom stopwords to remove, by default None.
        additional_stopwords : Optional[List[str]], optional
            Additional stopwords to remove, alias for custom_stopwords, by default None.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with preprocessed text.
        """
        if self.documents is None or self.text_column is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Update the TextNormalizer settings in the modeler first
        self.modeler.text_normalizer.lowercase = lowercase
        self.modeler.text_normalizer.remove_punctuation = remove_punctuation
        self.modeler.text_normalizer.remove_numbers = remove_numbers
        self.modeler.text_normalizer.lemmatize = lemmatize
        
        # Handle custom stopwords (combine both parameters for backward compatibility)
        stopwords_to_add = []
        if custom_stopwords:
            stopwords_to_add.extend(custom_stopwords)
        if additional_stopwords:
            stopwords_to_add.extend(additional_stopwords)
            
        if stopwords_to_add:
            self.modeler.text_normalizer.add_stopwords(stopwords_to_add)
        
        # Then call preprocess with only the parameters it accepts
        processed_docs = self.modeler.preprocess(
            self.documents,
            text_column=self.text_column,
        )
        
        self.preprocessing_complete = True
        logger.info("Text preprocessing completed.")
        return processed_docs
        
    def get_preprocessed_data(self) -> pd.DataFrame:
        """Get the preprocessed data.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with preprocessed text.
        """
        if not self.preprocessing_complete:
            raise ValueError("No preprocessing has been performed. Call preprocess_documents() first.")
            
        return self.modeler.documents
        
    def set_topic_assignments(self, topic_assignments: pd.DataFrame) -> None:
        """Set topic assignments from an external model (e.g., BERTopic).
        
        Parameters
        ----------
        topic_assignments : pd.DataFrame
            DataFrame with topic assignments. 
            Must contain columns 'topic' and optionally 'topic_probability'.
            
        Returns
        -------
        None
        """
        if self.modeler.documents is None:
            raise ValueError("No documents loaded. Call load_data() and preprocess_documents() first.")
            
        if "topic" not in topic_assignments.columns:
            raise ValueError("Topic assignments must contain a 'topic' column.")
            
        # Ensure the index matches
        if not topic_assignments.index.equals(self.modeler.documents.index):
            raise ValueError("Topic assignments index does not match documents index.")
            
        # Add topic assignments to the modeler's documents
        self.modeler.documents["topic"] = topic_assignments["topic"]
        
        # Add topic probability if available
        if "topic_probability" in topic_assignments.columns:
            self.modeler.documents["topic_probability"] = topic_assignments["topic_probability"]
            
        # Set topic assignments in the modeler
        self.modeler.topic_assignments = topic_assignments
        
        # Set modeling complete flag
        self.modeling_complete = True
        logger.info("Topic assignments set from external model.")
    
    def discover_topics(
        self,
        method: Optional[str] = None,
        num_topics: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Discover topics in the preprocessed text data.
        
        Parameters
        ----------
        method : Optional[str], optional
            Topic modeling method, by default None.
            Options: "embedding_cluster", "lda".
            If None, uses the value from config.
        num_topics : Optional[int], optional
            Number of topics to discover, by default None.
            If None, uses the value from config.
        **kwargs : dict
            Additional method-specific parameters.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with topic assignments.
        """
        # Get values from config if not provided
        if method is None:
            method = self.config.modeling.default_method
        if num_topics is None:
            num_topics = self.config.modeling.default_num_topics
            
        # Ensure preprocessing has been done
        if not self.preprocessing_complete:
            logger.warning("Documents have not been preprocessed. Running preprocessing with default settings.")
            self.preprocess_documents()
            
        # Discover topics using the underlying modeler
        topics_df = self.modeler.discover_topics(
            method=method,
            num_topics=num_topics,
            **kwargs
        )
        
        self.modeling_complete = True
        
        # If we used deduplication, map topics back to original dataset
        if self.deduplicated and self.original_documents is not None:
            logger.info("Joining topic assignments back to full dataset with duplicates")
            
            # Create a mapping from hash to topic
            topic_map = topics_df[['doc_hash', 'topic']].set_index('doc_hash').to_dict()['topic']
            
            # Apply topics to original dataset by using the hash values
            self.original_topics_df = self.original_documents.copy()
            self.original_topics_df['topic'] = self.original_documents[self.doc_hash_column].map(topic_map)
            
            # If topic probabilities are available, map those too
            if 'topic_probability' in topics_df.columns:
                prob_map = topics_df[['doc_hash', 'topic_probability']].set_index('doc_hash').to_dict()['topic_probability']
                self.original_topics_df['topic_probability'] = self.original_documents[self.doc_hash_column].map(prob_map)
            
            logger.info(f"Successfully mapped topics to all {len(self.original_topics_df)} documents (including duplicates)")
        else:
            self.original_topics_df = None
        # Get clustering algorithm info if using embedding clustering
        if method == "embedding_cluster":
            clustering_algo = self.modeler.config.modeling.clustering.algorithm
            if clustering_algo == "hdbscan":
                logger.info(f"Topic modeling completed using {method} method with HDBSCAN clustering (auto-detected {self.modeler.topic_assignments.shape[1]} topics)")
            else:
                logger.info(f"Topic modeling completed using {method} method with {clustering_algo} clustering and {num_topics} topics")
        else:
            logger.info(f"Topic modeling completed using {method} method with {num_topics} topics")
        # Return the appropriate topic assignments
        if self.deduplicated and self.original_topics_df is not None:
            return self.original_topics_df
        else:
            return topics_df
    
    def visualize_topics(
        self,
        plot_type: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Visualize the discovered topics.
        
        Parameters
        ----------
        plot_type : Optional[str], optional
            Type of visualization, by default None.
            Options: "embeddings", "distribution", "trends", "map", "timespace".
            If None, uses the value from config.
        **kwargs : dict
            Additional visualization-specific parameters.
            
        Returns
        -------
        Any
            Visualization object (typically a plotly Figure).
        """
        if not self.modeling_complete:
            raise ValueError("Topic modeling has not been completed. Call discover_topics() first.")
            
        # Get value from config if not provided
        if plot_type is None:
            plot_type = self.config.visualization.defaults.plot_type
            
        # Validate plot type
        plot_types = ["embeddings", "distribution", "trends", "map", "timespace"]
        if plot_type not in plot_types:
            raise ValueError(f"Invalid plot_type: {plot_type}. Must be one of {plot_types}.")
            
        # Generate visualization based on plot type
        if plot_type == "embeddings":
            return self.modeler.visualize_embeddings(**kwargs)
        elif plot_type == "distribution":
            return self.modeler.visualize_topic_distribution(**kwargs)
        elif plot_type == "trends":
            if self.time_column is None:
                raise ValueError("Time column must be specified for trend visualization.")
            return self.modeler.visualize_topic_trends(
                time_column=self.time_column,
                **kwargs
            )
        elif plot_type == "map":
            if self.geo_column is None:
                raise ValueError("Geographic column must be specified for map visualization.")
            return self.modeler.visualize_geospatial_topics(
                geo_column=self.geo_column,
                **kwargs
            )
        elif plot_type == "timespace":
            if self.time_column is None or self.geo_column is None:
                raise ValueError("Time and geographic columns must be specified for time-space visualization.")
            return self.modeler.visualize_timespace_topics(
                time_column=self.time_column,
                geo_column=self.geo_column,
                **kwargs
            )
    
    def generate_comprehensive_report(
        self,
        output_path: Optional[str] = None,
        open_browser: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """Generate a comprehensive HTML report of the topic modeling results.
        
        Parameters
        ----------
        output_path : Optional[str], optional
            Path to save the report, by default None.
            If None, uses the path from config.
        open_browser : Optional[bool], optional
            Whether to open the report in a browser, by default None.
            If None, uses the value from config.
        **kwargs : dict
            Additional report-specific parameters.
            
        Returns
        -------
        str
            Path to the generated report.
        """
        if not self.modeling_complete:
            raise ValueError("Topic modeling has not been completed. Call discover_topics() first.")
            
        # Get values from config if not provided
        if output_path is None:
            output_path = self.config.workflow.report_paths.comprehensive_report
        if open_browser is None:
            open_browser = self.config.workflow.features.auto_open_browser
            
        # Generate report using the underlying modeler
        report_path = self.modeler.generate_report(
            output_path=output_path,
            **kwargs
        )
        
        # Open in browser if requested
        if open_browser:
            webbrowser.open(f'file://{os.path.abspath(report_path)}')
            
        logger.info(f"Comprehensive report generated at {report_path}")
        return report_path
    
    def run_complete_workflow(
        self,
        data: Union[pd.DataFrame, str, Path],
        text_column: str,
        time_column: Optional[str] = None,
        geo_column: Optional[str] = None,
        category_column: Optional[str] = None,
        acronym_mappings: Optional[Dict[str, str]] = None,
        spelling_corrections: Optional[Dict[str, str]] = None,
        modeling_method: str = "embedding_cluster",
        num_topics: int = 10,
        output_path: Optional[str] = None,
        open_browser: bool = True,
    ) -> str:
        """Run the complete workflow from data loading to report generation.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, str, Path]
            Data to load, either a DataFrame or a path to a CSV file.
        text_column : str
            Name of the column containing text data.
        time_column : Optional[str], optional
            Name of the column containing time/date data, by default None.
        geo_column : Optional[str], optional
            Name of the column containing geographic data, by default None.
        category_column : Optional[str], optional
            Name of the column containing category data, by default None.
        acronym_mappings : Optional[Dict[str, str]], optional
            Custom acronym mappings, by default None.
        spelling_corrections : Optional[Dict[str, str]], optional
            Custom spelling corrections, by default None.
        modeling_method : str, optional
            Topic modeling method, by default "embedding_cluster".
        num_topics : int, optional
            Number of topics to discover, by default 10.
        output_path : Optional[str], optional
            Path to save the report, by default None.
        open_browser : bool, optional
            Whether to open the report in a browser, by default True.
            
        Returns
        -------
        str
            Path to the generated report.
        """
        # 1. Load data
        self.load_data(
            data=data,
            text_column=text_column,
            time_column=time_column,
            geo_column=geo_column,
            category_column=category_column,
        )
        
        # 2. Expand acronyms if mappings provided
        if acronym_mappings:
            self.expand_acronyms(custom_mappings=acronym_mappings)
        
        # 3. Correct spelling if corrections provided
        if spelling_corrections:
            self.correct_spelling(custom_corrections=spelling_corrections)
        
        # 4. Preprocess documents
        self.preprocess_documents()
        
        # 5. Discover topics
        self.discover_topics(method=modeling_method, num_topics=num_topics)
        
        # 6. Generate report
        report_path = self.generate_comprehensive_report(
            output_path=output_path,
            open_browser=open_browser,
        )
        
        return report_path


# Function for easy access to the workflow
def create_workflow(
    config_path: Optional[Union[str, Path]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    auto_create_config: bool = True,
) -> MenoWorkflow:
    """Create a new MenoWorkflow instance.
    
    Parameters
    ----------
    config_path : Optional[Union[str, Path]], optional
        Path to configuration file, by default None.
    config_overrides : Optional[Dict[str, Any]], optional
        Dictionary of configuration overrides, by default None.
    auto_create_config : bool, optional
        Whether to automatically create a config file if none exists, by default True.
        
    Returns
    -------
    MenoWorkflow
        Initialized workflow instance.
    """
    return MenoWorkflow(
        config_path=config_path,
        config_overrides=config_overrides,
        auto_create_config=auto_create_config,
    )


def load_workflow_config(
    config_path: Optional[Union[str, Path]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    auto_create: bool = True,
) -> WorkflowMenoConfig:
    """Load a workflow configuration.
    
    Parameters
    ----------
    config_path : Optional[Union[str, Path]], optional
        Path to configuration file, by default None.
    config_overrides : Optional[Dict[str, Any]], optional
        Dictionary of configuration overrides, by default None.
    auto_create : bool, optional
        Whether to automatically create a config file if none exists, by default True.
        
    Returns
    -------
    WorkflowMenoConfig
        Loaded and validated configuration.
    """
    config = load_config(config_path=config_path, config_type="workflow", auto_create=auto_create)
    if config_overrides:
        config = merge_configs(config, config_overrides)
    return config


def save_workflow_config(
    config: WorkflowMenoConfig,
    output_path: Union[str, Path],
) -> None:
    """Save a workflow configuration to a file.
    
    Parameters
    ----------
    config : WorkflowMenoConfig
        Configuration to save.
    output_path : Union[str, Path]
        Path to save the configuration to.
    """
    save_config(config, output_path)