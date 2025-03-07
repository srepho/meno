"""Tests for the web interface module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, call
import dash
from dash.testing.composite import DashComposite
from dash.testing.application_runners import ThreadedRunner

from meno.web_interface import MenoWebApp, launch_web_interface


class TestMenoWebApp:
    """Tests for the MenoWebApp class."""
    
    def test_init(self):
        """Test initialization of the web app."""
        app = MenoWebApp(port=8051, debug=True)
        assert app.port == 8051
        assert app.debug is True
        assert app.app is not None
        assert hasattr(app, 'temp_dir')
        app.cleanup()
    
    @patch('meno.web_interface.tempfile.mkdtemp')
    def test_temp_dir_creation(self, mock_mkdtemp):
        """Test that a temporary directory is created."""
        mock_mkdtemp.return_value = '/tmp/meno_test_dir'
        app = MenoWebApp()
        assert app.temp_dir == '/tmp/meno_test_dir'
        app.cleanup()
    
    def test_layout_structure(self):
        """Test that the layout is structured correctly."""
        app = MenoWebApp()
        
        # Check main layout components
        layout = app.app.layout
        assert isinstance(layout, dash.html.Div)
        
        # Check tabs
        tabs = [c for c in layout.children if isinstance(c, dash.dcc.Tabs)]
        assert len(tabs) == 1
        main_tabs = tabs[0]
        
        # Check number of tabs
        assert len(main_tabs.children) == 4
        tab_ids = [tab.tab_id for tab in main_tabs.children]
        assert set(tab_ids) == {'tab-data', 'tab-model', 'tab-results', 'tab-search'}
        
        # Check initial state
        assert main_tabs.active_tab == 'tab-data'
        for tab in main_tabs.children:
            if tab.tab_id != 'tab-data':
                assert tab.disabled is True
        
        app.cleanup()
    
    @patch('meno.web_interface.shutil.rmtree')
    def test_cleanup(self, mock_rmtree):
        """Test cleanup method removes temp directory."""
        with patch('meno.web_interface.os.path.exists', return_value=True):
            app = MenoWebApp()
            app.temp_dir = '/tmp/meno_test_dir'
            app.cleanup()
            mock_rmtree.assert_called_once_with('/tmp/meno_test_dir')
    
    @patch('meno.web_interface.MenoWebApp.run')
    def test_launch_web_interface(self, mock_run):
        """Test launch_web_interface function."""
        with patch('meno.web_interface.MenoWebApp') as mock_app_class:
            mock_app = MagicMock()
            mock_app_class.return_value = mock_app
            
            launch_web_interface(port=9000, debug=True)
            
            mock_app_class.assert_called_once_with(port=9000, debug=True)
            mock_app.run.assert_called_once()
            mock_app.cleanup.assert_called_once()


@pytest.fixture
def mock_models():
    """Create mock topic models for testing."""
    simple_model = MagicMock()
    simple_model.transform.return_value = (
        np.array([0, 1, 0, 2]), 
        np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    )
    
    tfidf_model = MagicMock()
    tfidf_model.transform.return_value = (
        np.array([1, 0, 1, 2]), 
        np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    )
    
    nmf_model = MagicMock()
    nmf_model.transform.return_value = np.array([
        [0.1, 0.8, 0.1], 
        [0.7, 0.2, 0.1], 
        [0.2, 0.7, 0.1], 
        [0.1, 0.1, 0.8]
    ])
    
    lsa_model = MagicMock()
    lsa_model.transform.return_value = np.array([
        [0.1, 0.8, 0.1], 
        [0.7, 0.2, 0.1], 
        [0.2, 0.7, 0.1], 
        [0.1, 0.1, 0.8]
    ])
    
    # Mock topic info for all models
    topic_info = pd.DataFrame({
        'Topic': [0, 1, 2],
        'Name': ['Service', 'Product', 'Software'],
        'Size': [10, 15, 8],
        'Count': [10, 15, 8],
        'Words': [
            ['service', 'customer', 'support'],
            ['product', 'quality', 'features'],
            ['software', 'interface', 'user']
        ]
    })
    
    for model in [simple_model, tfidf_model, nmf_model, lsa_model]:
        model.get_topic_info.return_value = topic_info
        model.is_fitted = True
    
    return {
        'simple_kmeans': simple_model,
        'tfidf': tfidf_model,
        'nmf': nmf_model,
        'lsa': lsa_model
    }


class TestCallbacks:
    """Test the callbacks of the web interface."""
    
    @patch('meno.web_interface.pd.read_csv')
    def test_process_data_csv(self, mock_read_csv):
        """Test processing CSV data."""
        app = MenoWebApp()
        
        # Mock CSV data
        mock_df = pd.DataFrame({
            'text': ['Document 1', 'Document 2', 'Document 3']
        })
        mock_read_csv.return_value = mock_df
        
        # Mock Base64 content
        base64_content = 'data:text/csv;base64,RG9jdW1lbnQgMSxEb2N1bWVudCAyLERvY3VtZW50IDM='
        
        # Test callback context
        with patch('dash.callback_context') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': 'upload-data.contents'}]
            
            output = app.app.callback_map['upload-status.children']['callback'](
                base64_content, None, 'test.csv', None
            )
            
            # Verify result
            assert output[0] is not None  # Upload status
            assert output[3] == ['Document 1', 'Document 2', 'Document 3']  # Documents
            assert output[4] is False  # Model tab enabled
        
        app.cleanup()
    
    @patch('meno.modeling.unified_topic_modeling.UnifiedTopicModeler')
    def test_train_model(self, mock_unified_modeler):
        """Test training a model."""
        app = MenoWebApp()
        
        # Setup mock model
        mock_model = MagicMock()
        mock_model.get_topic_info.return_value = pd.DataFrame({
            'Topic': [0, 1, 2],
            'Name': ['Topic 0', 'Topic 1', 'Topic 2'],
            'Count': [5, 10, 15]
        })
        mock_unified_modeler.return_value = mock_model
        
        # Test parameters
        documents = ['Doc 1', 'Doc 2', 'Doc 3']
        model_type = 'nmf'
        num_topics = 5
        random_seed = 42
        max_features = 1000
        preprocessing = ['stop', 'lower']
        
        # Call train_model
        output = app.app.callback_map['model-status.children']['callback'](
            1, documents, model_type, num_topics, random_seed, max_features, preprocessing
        )
        
        # Verify model was trained
        mock_unified_modeler.assert_called_once()
        mock_model.fit.assert_called_once_with(documents)
        
        # Verify output enables results tab
        assert output[2] is False  # Results tab enabled
        assert output[3] is False  # Search tab enabled
        assert output[4] == 'tab-results'  # Active tab switched
        
        app.cleanup()
    
    def test_model_description(self):
        """Test model description callback."""
        app = MenoWebApp()
        
        for model_type in ['simple_kmeans', 'tfidf', 'nmf', 'lsa']:
            # Call callback
            description = app.app.callback_map['model-description.children']['callback'](model_type)
            
            # Verify non-empty description
            assert description is not None
            assert len(description) > 0
        
        app.cleanup()
    
    def test_update_topics_overview(self):
        """Test topic overview update."""
        app = MenoWebApp()
        
        # Mock model data
        model_data = {'is_trained': True, 'model_id': 'test_model'}
        
        # Call topics overview callback
        topics_overview = app.app.callback_map['topics-overview.children']['callback'](model_data)
        
        # Verify output contains a visualization
        assert topics_overview is not None
        
        # Test with untrained model
        untrained_model = {'is_trained': False}
        empty_overview = app.app.callback_map['topics-overview.children']['callback'](untrained_model)
        assert empty_overview is not None
        
        app.cleanup()
    
    def test_topic_selector_update(self):
        """Test topic selector update callback."""
        app = MenoWebApp()
        
        # Mock model data
        model_data = {'is_trained': True, 'model_id': 'test_model'}
        results_data = None
        
        # Call callback
        options, value, filter_options = app.app.callback_map['topic-selector.options']['callback'](
            model_data, results_data
        )
        
        # Verify options are created
        assert len(options) > 0
        assert value is not None
        assert len(filter_options) > 0
        assert filter_options[0]['value'] == 'all'
        
        app.cleanup()
    
    def test_search_results(self):
        """Test search results callback."""
        app = MenoWebApp()
        
        # Mock parameters
        n_clicks = 1
        topic_filter = 'all'
        sort_by = 'relevance'
        query = 'test query'
        model_data = {'is_trained': True}
        documents = ['Doc 1', 'Doc 2', 'Doc 3', 'Doc 4', 'Doc 5']
        
        # Call callback
        results = app.app.callback_map['search-results.children']['callback'](
            n_clicks, topic_filter, sort_by, query, model_data, documents
        )
        
        # Verify results are displayed
        assert results is not None
        
        # Test with no query
        no_query_results = app.app.callback_map['search-results.children']['callback'](
            n_clicks, topic_filter, sort_by, None, model_data, documents
        )
        assert "Enter a search query" in str(no_query_results)
        
        app.cleanup()


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the web interface.
    
    These tests simulate user interactions with the interface.
    Note: These tests are marked for separate execution as they start server processes.
    """
    
    @patch('meno.web_interface.launch_web_interface')  
    def test_launch_from_cli(self, mock_launch):
        """Test launching from CLI."""
        from meno.cli.web_interface_cli import main
        
        # Mock sys.argv
        with patch('sys.argv', ['meno-web', '--port=9000', '--debug']):
            with patch('argparse.ArgumentParser.parse_args') as mock_args:
                mock_args.return_value = MagicMock(port=9000, debug=True)
                
                # Run the CLI function
                main()
                
                # Verify launch function was called
                mock_launch.assert_called_once_with(port=9000, debug=True)
    
    @pytest.mark.skip(reason="Requires browser webdriver for testing")
    def test_end_to_end_workflow(self, mock_models):
        """Test end-to-end workflow with the interface.
        
        Note: This test requires a browser driver and is skipped by default.
        """
        with patch('meno.modeling.unified_topic_modeling.UnifiedTopicModeler') as mock_modeler:
            # Configure the mock to return our test models
            mock_modeler.side_effect = lambda **kwargs: mock_models.get(kwargs.get('method', 'simple_kmeans'))
            
            # Create app for testing
            app = MenoWebApp(debug=True)
            
            # Setup DashComposite for testing
            dash_duo = DashComposite(
                app.app,
                browser='chrome',
                headless=True,
                options={'no-sandbox': True}
            )
            
            try:
                # Wait for the app to load
                dash_duo.wait_for_element('#upload-data', timeout=4)
                
                # Enter sample text
                dash_duo.find_element('#sample-text').send_keys(
                    "Document 1\nDocument 2\nDocument 3\nDocument 4"
                )
                dash_duo.find_element('#use-sample-text').click()
                
                # Wait for data preview
                dash_duo.wait_for_text_to_equal('#data-stats', 'Total documents: 4', timeout=4)
                
                # Switch to model tab
                dash_duo.wait_for_element('#tab-model', timeout=4)
                dash_duo.find_element('#tab-model').click()
                
                # Configure model
                dash_duo.find_element('#model-type').select_by_value('nmf')
                dash_duo.find_element('#num-topics').send_keys('3')
                
                # Train model
                dash_duo.find_element('#train-model').click()
                
                # Verify results tab is enabled and switched to
                dash_duo.wait_for_element('#topic-landscape-viz', timeout=4)
                
                # Test topic selector
                dash_duo.find_element('#topic-selector').select_by_value('1')
                dash_duo.wait_for_text_to_equal('#topic-details h5', 'Product', timeout=4)
                
                # Test search feature
                dash_duo.find_element('#tab-search').click()
                dash_duo.find_element('#search-query').send_keys('product')
                dash_duo.find_element('#search-button').click()
                
                # Verify search results are displayed
                dash_duo.wait_for_element('.card-header', timeout=4)
                
            finally:
                dash_duo.driver.quit()
                app.cleanup()