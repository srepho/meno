"""Command-line interface for team configuration management.

This module provides a CLI for creating, updating, merging, and analyzing team configurations.
These tools make it easier to manage domain-specific knowledge across teams and organizations.
"""

import argparse
import sys
import os
import json
import pprint
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from ..utils.team_config import (
    create_team_config,
    update_team_config,
    merge_team_configs,
    get_team_config_stats,
    compare_team_configs,
    export_team_acronyms,
    export_team_spelling_corrections,
    import_acronyms_from_file,
    import_spelling_corrections_from_file
)

# Custom exception classes for better error handling
class ConfigError(Exception):
    """Base class for configuration-related errors."""
    pass

class ConfigNotFoundError(ConfigError):
    """Raised when a configuration file is not found."""
    pass

class ConfigFormatError(ConfigError):
    """Raised when a configuration file has invalid format."""
    pass

# Centralized error handling function
def handle_error(error, exit_code=1):
    """Handle errors in a consistent way."""
    if isinstance(error, ConfigNotFoundError):
        print(f"Configuration error: {str(error)}")
    elif isinstance(error, json.JSONDecodeError):
        print(f"JSON parsing error: {str(error)}")
    elif isinstance(error, FileNotFoundError):
        print(f"File not found: {str(error)}")
    else:
        print(f"Error: {str(error)}")
    
    if exit_code is not None:
        sys.exit(exit_code)

# Helper functions for file operations
def load_json_file(file_path):
    """Safely load a JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ConfigNotFoundError(f"File {file_path} not found")
    except json.JSONDecodeError as e:
        raise ConfigFormatError(f"Invalid JSON in {file_path}: {str(e)}")
    except Exception as e:
        raise ConfigError(f"Failed to load JSON from {file_path}: {str(e)}")








def create_config_cmd(args):
    """Create a new team configuration."""
    acronyms = None
    spelling_corrections = None
    
    # Load acronyms from file if provided
    if args.acronyms_file:
        acronyms = import_acronyms_from_file(args.acronyms_file)
        print(f"Loaded {len(acronyms)} acronyms from {args.acronyms_file}")
    
    # Load spelling corrections from file if provided
    if args.corrections_file:
        spelling_corrections = import_spelling_corrections_from_file(args.corrections_file)
        print(f"Loaded {len(spelling_corrections)} spelling corrections from {args.corrections_file}")
    
    # Load model settings from file if provided
    model_settings = None
    if args.model_settings_file:
        with open(args.model_settings_file, 'r') as f:
            model_settings = json.load(f)
        print(f"Loaded model settings from {args.model_settings_file}")
    
    # Load visualization settings from file if provided
    visualization_settings = None
    if args.visualization_settings_file:
        with open(args.visualization_settings_file, 'r') as f:
            visualization_settings = json.load(f)
        print(f"Loaded visualization settings from {args.visualization_settings_file}")
    
    # Create the configuration
    config = create_team_config(
        team_name=args.team_name,
        acronyms=acronyms,
        spelling_corrections=spelling_corrections,
        model_settings=model_settings,
        visualization_settings=visualization_settings,
        base_config_path=args.base_config,
        output_path=args.output_path
    )
    
    # Print configuration statistics
    if args.output_path:
        stats = get_team_config_stats(args.output_path)
        print(f"\nTeam configuration created for '{args.team_name}':")
        print(f"- Acronyms: {stats['acronym_count']}")
        print(f"- Spelling corrections: {stats['spelling_correction_count']}")
        print(f"- Default model: {stats['default_model']}")
        print(f"- Output path: {args.output_path}")


def update_config_cmd(args):
    """Update an existing team configuration."""
    acronyms = None
    spelling_corrections = None
    
    # Load acronyms from file if provided
    if args.acronyms_file:
        acronyms = import_acronyms_from_file(args.acronyms_file)
        print(f"Loaded {len(acronyms)} acronyms from {args.acronyms_file}")
    
    # Load spelling corrections from file if provided
    if args.corrections_file:
        spelling_corrections = import_spelling_corrections_from_file(args.corrections_file)
        print(f"Loaded {len(spelling_corrections)} spelling corrections from {args.corrections_file}")
    
    # Load model settings from file if provided
    model_settings = None
    if args.model_settings_file:
        with open(args.model_settings_file, 'r') as f:
            model_settings = json.load(f)
        print(f"Loaded model settings from {args.model_settings_file}")
    
    # Load visualization settings from file if provided
    visualization_settings = None
    if args.visualization_settings_file:
        with open(args.visualization_settings_file, 'r') as f:
            visualization_settings = json.load(f)
        print(f"Loaded visualization settings from {args.visualization_settings_file}")
    
    # Update the configuration
    config = update_team_config(
        config_path=args.config_path,
        acronyms=acronyms,
        spelling_corrections=spelling_corrections,
        model_settings=model_settings,
        visualization_settings=visualization_settings,
        merge_mode=args.merge_mode
    )
    
    # Print configuration statistics
    stats = get_team_config_stats(args.config_path)
    print(f"\nTeam configuration updated:")
    print(f"- Team: {stats['team_name']}")
    print(f"- Acronyms: {stats['acronym_count']}")
    print(f"- Spelling corrections: {stats['spelling_correction_count']}")
    print(f"- Last modified: {stats['last_modified']}")


def merge_configs_cmd(args):
    """Merge multiple team configurations."""
    # Check if configs exist
    for config_path in args.config_paths:
        if not os.path.exists(config_path):
            print(f"Error: Configuration file {config_path} does not exist")
            sys.exit(1)
    
    # Merge configurations
    merged_config = merge_team_configs(
        configs=args.config_paths,
        output_path=args.output_path,
        team_name=args.team_name
    )
    
    # Print configuration statistics
    stats = get_team_config_stats(args.output_path)
    print(f"\nMerged configuration created:")
    print(f"- Team: {stats['team_name']}")
    print(f"- Acronyms: {stats['acronym_count']}")
    print(f"- Spelling corrections: {stats['spelling_correction_count']}")
    print(f"- Output path: {args.output_path}")


def stats_cmd(args):
    """Display statistics about a team configuration."""
    stats = get_team_config_stats(args.config_path)
    
    print(f"Statistics for {args.config_path}:")
    print(f"- Team: {stats['team_name']}")
    print(f"- Created: {stats['created']}")
    print(f"- Last modified: {stats['last_modified']}")
    print(f"- Acronyms: {stats['acronym_count']}")
    print(f"- Spelling corrections: {stats['spelling_correction_count']}")
    print(f"- Default model: {stats['default_model']}")
    print(f"- Default method: {stats['default_method']}")
    print(f"- Config hash: {stats['config_hash']}")


def compare_configs_cmd(args):
    """Compare two team configurations."""
    comparison = compare_team_configs(
        config1_path=args.config1_path,
        config2_path=args.config2_path
    )
    
    # Get team names
    team1 = comparison["team_names"]["config1"]
    team2 = comparison["team_names"]["config2"]
    
    print(f"Comparison between {team1} and {team2}:")
    
    # Acronyms comparison
    acronyms = comparison["acronyms"]
    print(f"\nAcronyms:")
    print(f"- Common: {acronyms['common_count']}")
    print(f"- Unique to {team1}: {len(acronyms['unique_to_config1'])}")
    print(f"- Unique to {team2}: {len(acronyms['unique_to_config2'])}")
    print(f"- Different expansions: {len(acronyms['differing_expansions'])}")
    
    # Show some examples of differing acronyms
    if acronyms['differing_expansions'] and not args.quiet:
        print("\nExamples of differing acronym expansions:")
        for i, (acronym, expansions) in enumerate(acronyms['differing_expansions'].items()):
            if i >= 5:  # Show at most 5 examples
                print(f"... and {len(acronyms['differing_expansions']) - 5} more")
                break
            print(f"  - {acronym}:")
            print(f"    - {team1}: {expansions[team1]}")
            print(f"    - {team2}: {expansions[team2]}")
    
    # Spelling corrections comparison
    spellings = comparison["spelling_corrections"]
    print(f"\nSpelling corrections:")
    print(f"- Common: {spellings['common_count']}")
    print(f"- Unique to {team1}: {len(spellings['unique_to_config1'])}")
    print(f"- Unique to {team2}: {len(spellings['unique_to_config2'])}")
    print(f"- Different corrections: {len(spellings['differing_corrections'])}")
    
    # Show some examples of differing corrections
    if spellings['differing_corrections'] and not args.quiet:
        print("\nExamples of differing spelling corrections:")
        for i, (word, corrections) in enumerate(spellings['differing_corrections'].items()):
            if i >= 5:  # Show at most 5 examples
                print(f"... and {len(spellings['differing_corrections']) - 5} more")
                break
            print(f"  - {word}:")
            print(f"    - {team1}: {corrections[team1]}")
            print(f"    - {team2}: {corrections[team2]}")
    
    # Model comparison
    models = comparison["models"]
    print(f"\nModels:")
    print(f"- Same model: {models['same_model']}")
    print(f"- {team1} model: {models['config1_model']}")
    print(f"- {team2} model: {models['config2_model']}")
    
    # Export full comparison if requested
    if args.output_path:
        with open(args.output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nFull comparison exported to {args.output_path}")


def export_acronyms_cmd(args):
    """Export acronyms from a team configuration."""
    acronyms = export_team_acronyms(
        config_path=args.config_path,
        output_format=args.format,
        output_path=args.output_path
    )
    
    print(f"Exported {len(acronyms)} acronyms from {args.config_path}")
    if args.output_path:
        print(f"Saved to {args.output_path}")
    else:
        # Print first 10 acronyms
        print("\nSample acronyms:")
        for i, (acronym, expansion) in enumerate(list(acronyms.items())[:10]):
            print(f"- {acronym}: {expansion}")
        if len(acronyms) > 10:
            print(f"... and {len(acronyms) - 10} more")


def export_corrections_cmd(args):
    """Export spelling corrections from a team configuration."""
    corrections = export_team_spelling_corrections(
        config_path=args.config_path,
        output_format=args.format,
        output_path=args.output_path
    )
    
    print(f"Exported {len(corrections)} spelling corrections from {args.config_path}")
    if args.output_path:
        print(f"Saved to {args.output_path}")
    else:
        # Print first 10 corrections
        print("\nSample corrections:")
        for i, (word, correction) in enumerate(list(corrections.items())[:10]):
            print(f"- {word} → {correction}")
        if len(corrections) > 10:
            print(f"... and {len(corrections) - 10} more")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Meno Team Configuration Manager",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create command
    create_parser = subparsers.add_parser(
        "create", help="Create a new team configuration"
    )
    create_parser.add_argument("team_name", help="Name of the team or domain")
    create_parser.add_argument(
        "--output-path", "-o",
        help="Path to save the configuration (defaults to team_name_config.yaml)"
    )
    create_parser.add_argument(
        "--base-config", "-b",
        help="Path to base configuration file to extend"
    )
    create_parser.add_argument(
        "--acronyms-file", "-a",
        help="Path to JSON or YAML file with acronyms"
    )
    create_parser.add_argument(
        "--corrections-file", "-c",
        help="Path to JSON or YAML file with spelling corrections"
    )
    create_parser.add_argument(
        "--model-settings-file", "-m",
        help="Path to JSON file with model settings"
    )
    create_parser.add_argument(
        "--visualization-settings-file", "-v",
        help="Path to JSON file with visualization settings"
    )
    
    # Update command
    update_parser = subparsers.add_parser(
        "update", help="Update an existing team configuration"
    )
    update_parser.add_argument(
        "config_path", help="Path to the team configuration file"
    )
    update_parser.add_argument(
        "--acronyms-file", "-a",
        help="Path to JSON or YAML file with acronyms to add/update"
    )
    update_parser.add_argument(
        "--corrections-file", "-c",
        help="Path to JSON or YAML file with spelling corrections to add/update"
    )
    update_parser.add_argument(
        "--model-settings-file", "-m",
        help="Path to JSON file with model settings to update"
    )
    update_parser.add_argument(
        "--visualization-settings-file", "-v",
        help="Path to JSON file with visualization settings to update"
    )
    update_parser.add_argument(
        "--merge-mode", choices=["update", "replace"], default="update",
        help="How to merge dictionaries (update or replace)"
    )
    
    # Merge command
    merge_parser = subparsers.add_parser(
        "merge", help="Merge multiple team configurations"
    )
    merge_parser.add_argument(
        "config_paths", nargs="+",
        help="Paths to the team configuration files to merge"
    )
    merge_parser.add_argument(
        "--output-path", "-o", required=True,
        help="Path to save the merged configuration"
    )
    merge_parser.add_argument(
        "--team-name", "-n",
        help="Name for the merged team configuration"
    )
    
    # Stats command
    stats_parser = subparsers.add_parser(
        "stats", help="Display statistics about a team configuration"
    )
    stats_parser.add_argument(
        "config_path", help="Path to the team configuration file"
    )
    
    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare two team configurations"
    )
    compare_parser.add_argument(
        "config1_path", help="Path to the first team configuration file"
    )
    compare_parser.add_argument(
        "config2_path", help="Path to the second team configuration file"
    )
    compare_parser.add_argument(
        "--output-path", "-o",
        help="Path to save the comparison result as JSON"
    )
    compare_parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Don't show detailed examples of differences"
    )
    
    # Export acronyms command
    export_acronyms_parser = subparsers.add_parser(
        "export-acronyms", help="Export acronyms from a team configuration"
    )
    export_acronyms_parser.add_argument(
        "config_path", help="Path to the team configuration file"
    )
    export_acronyms_parser.add_argument(
        "--output-path", "-o",
        help="Path to save the exported acronyms"
    )
    export_acronyms_parser.add_argument(
        "--format", "-f", choices=["json", "yaml"], default="json",
        help="Format to export (json or yaml)"
    )
    
    # Export corrections command
    export_corrections_parser = subparsers.add_parser(
        "export-corrections", help="Export spelling corrections from a team configuration"
    )
    export_corrections_parser.add_argument(
        "config_path", help="Path to the team configuration file"
    )
    export_corrections_parser.add_argument(
        "--output-path", "-o",
        help="Path to save the exported corrections"
    )
    export_corrections_parser.add_argument(
        "--format", "-f", choices=["json", "yaml"], default="json",
        help="Format to export (json or yaml)"
    )
    
    # Parse arguments and dispatch to appropriate function
    args = parser.parse_args()
    
    if args.command == "create":
        create_config_cmd(args)
    elif args.command == "update":
        update_config_cmd(args)
    elif args.command == "merge":
        merge_configs_cmd(args)
    elif args.command == "stats":
        stats_cmd(args)
    elif args.command == "compare":
        compare_configs_cmd(args)
    elif args.command == "export-acronyms":
        export_acronyms_cmd(args)
    elif args.command == "export-corrections":
        export_corrections_cmd(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()