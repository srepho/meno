# Meno v1.2.5 Release Notes

## Fixed
- Fixed syntax error in f-strings containing backslashes in LM preprocessor module
- Fixed Python 3.12+ compatibility issue with combined raw/f-strings (rf"...")
- Updated error handling for f-string expressions with backslash characters
- Fixed issue causing import errors with simple_feedback module

## Technical Improvements
- Improved code for constructing regex patterns by separating raw strings and f-strings
- Simplified pattern creation for word boundary regex patterns
- Ensured compatibility with Python 3.12+ string literal handling

## Developer Notes
- The issue was related to Python 3.12's stricter handling of backslashes in f-string expressions
- Fixed by replacing problematic `rf"..."` patterns with separate construction using concatenation
- For example, replaced `rf"\b{re.escape(word)}\b"` with `r"\b" + re.escape(word) + r"\b"`