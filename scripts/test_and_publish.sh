#!/bin/bash
# Script to run tests and publish to PyPI if they pass

set -e  # Exit on error

# Ensure we're in the correct directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.."

echo "Running tests with pytest..."
python -m pytest

# Check test exit code
if [ $? -eq 0 ]; then
    echo "Tests passed! Proceeding with package build and publication..."
    
    # Clean previous builds
    rm -rf dist/ build/ *.egg-info
    
    # Build the package
    echo "Building package..."
    python -m build
    
    # Publish to PyPI
    echo "Publishing to PyPI..."
    python -m twine upload dist/*
    
    echo "Publication complete!"
else
    echo "Tests failed. Fix the issues before publishing."
    exit 1
fi