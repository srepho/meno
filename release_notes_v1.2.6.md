# Meno v1.2.6 Release Notes

## Bug Fixes

- **Python 3.10+ Compatibility**: Fixed additional f-string syntax issues in the preprocessing module
  - Fixed incompatible `rf"..."` syntax in `lm_preprocessor.py` that caused errors with Python 3.10+
  - Updated pattern matching for acronym extraction to use string concatenation instead of raw f-strings
  - Resolved import issues related to the `SimpleFeedback` and `TopicFeedbackManager` classes

## Minor Improvements

- **Examples Organization**: Restructured the examples directory for better navigation
  - Organized examples by functionality (basic, advanced, models, visualization, specialized)
  - Improved example documentation and headers
  - Added missing examples for key functionality

- **Documentation Updates**:
  - Updated README to reflect the latest version and features
  - Improved installation and usage instructions
  - Added cross-references to related documentation

## Installation

Install the latest version with pip:
```bash
pip install meno==1.2.6
```

Or upgrade your existing installation:
```bash
pip install --upgrade meno
```

For different installation options (lightweight, CPU-optimized, GPU-accelerated):
```bash
# Lightweight installation
pip install "meno[lightweight]"

# CPU-optimized
pip install "meno[cpu]"

# Full installation with GPU support
pip install "meno[full-gpu]"
```