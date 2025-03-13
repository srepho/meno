# Meno Release Notes v1.2.7

## Bug Fixes

- Fixed f-string compatibility issue in `simple_feedback.py` for Python 3.10
  - Resolved SyntaxError: "f-string expression part cannot include a backslash"
  - Changed conditional logic in f-string to use a separate variable

## Compatibility

- Python 3.10 and above compatibility improvements
- Fixed import issues with some Python 3.10 environments

## Other Notes

- No new features or API changes in this release
- This is a compatibility release to address Python 3.10 import errors