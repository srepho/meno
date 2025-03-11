# Meno v1.2.2: Automatic Configuration Creation

This release enhances usability with automatic configuration file creation, making it easier for users to get started with Meno without needing to manually create configuration files.

## New Features
- **Auto Config Creation**: Added automatic configuration file creation functionality
  - New `create_default_config_file` function to explicitly create config files
  - All initialization methods now have `auto_create_config` parameter (default: True)
  - Config files are automatically created when not found during initialization
  - Makes it easier for new users to get started with sensible defaults

## API Enhancements
- Added new top-level export: `create_default_config_file`
- Added `auto_create` parameter to `load_config` function
- Added `auto_create_config` parameter to:
  - `MenoTopicModeler.__init__`
  - `MenoWorkflow.__init__`
  - `create_workflow`
  - `load_workflow_config`

## New Examples
- Added `examples/auto_config_example.py` demonstrating automatic config creation

## Development
- Updated Pydantic usage to use `model_dump()` in place of deprecated `dict()` method
- Bumped version from 1.2.1 to 1.2.2

## Bug Fixes
- Fixed several validators to properly use the current pydantic API

## Documentation
- Updated parameter documentation to explain auto creation behavior

A special thank you to all contributors who helped make this release possible!