# Meno Release Notes v1.2.8

## Compatibility Improvements

- Modified package imports to avoid f-string compatibility issues in Python 3.10
  - Made active learning modules load on-demand rather than at import time
  - Removed automatic import of feedback visualization components
  - This change allows core modules to be imported without triggering f-string errors

## Usage Changes

- Users now need to explicitly import feedback components:
  ```python
  # Previous import (no longer works automatically)
  from meno import SimpleFeedback, TopicFeedbackManager

  # New import method
  from meno.active_learning.simple_feedback import SimpleFeedback, TopicFeedbackManager
  ```

- Feedback visualization components also need explicit imports:
  ```python
  # Previous import (no longer works automatically)
  from meno import plot_feedback_impact

  # New import method
  from meno.visualization.enhanced_viz.feedback_viz import plot_feedback_impact
  ```

## Other Notes

- No functional changes to the code implementation
- This is strictly a compatibility release to improve Python 3.10 support