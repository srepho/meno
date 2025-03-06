# Meno Tests

This directory contains tests for the Meno package. The tests are written using pytest and hypothesis.

## Running Tests

To run the tests, use the following command from the project root:

```bash
python -m pytest
```

To run with coverage:

```bash
python -m pytest --cov=meno
```

## Test Structure

- `conftest.py`: Contains fixtures used across tests
- `test_config.py`: Tests for configuration loading and validation
- `test_preprocessing.py`: Tests for text normalization and preprocessing
- `test_modeling.py`: Tests for embedding, topic modeling, and clustering
- `test_visualization.py`: Tests for visualization functions
- `test_integration.py`: Integration tests for the MenoTopicModeler class

## Test Data

The `data` directory contains test files used by the tests.

## Hypothesis Testing

Some tests use hypothesis for property-based testing. These tests generate random inputs to test a wider range of scenarios than traditional example-based tests.

## Coverage

The tests aim to achieve high coverage of the codebase. Coverage reports can be generated with the `--cov` option to pytest.