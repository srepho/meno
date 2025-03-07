"""Team configuration utilities for sharing Meno configurations across teams."""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import json
import os
from pathlib import Path
import logging

from .config import MenoConfig, get_config

logger = logging.getLogger(__name__)


class TeamConfig:
    """
    Team configuration manager for sharing configurations across teams.
    
    This class provides utilities for managing and sharing configuration settings,
    dictionaries, and assets across team members.
    
    Attributes:
        config: Base configuration
        team_name: Name of the team
        acronyms: Team-specific acronyms dictionary
        spelling: Team-specific spelling dictionary
        custom_stopwords: Team-specific stopwords list
        domain_terms: Domain-specific terminology
    """
    
    def __init__(
        self,
        team_name: str = "default",
        config: Optional[MenoConfig] = None,
        base_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the team configuration manager.
        
        Args:
            team_name: Name of the team
            config: Base configuration to use
            base_path: Base path for storing team configuration files
        """
        self.team_name = team_name
        self.config = config if config is not None else get_config()
        
        # Set up base path
        if base_path is None:
            self.base_path = Path.home() / ".meno" / "teams"
        else:
            self.base_path = Path(base_path)
            
        # Create directory if it doesn't exist
        team_dir = self.base_path / team_name
        team_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dictionaries
        self.acronyms = {}
        self.spelling = {}
        self.custom_stopwords = []
        self.domain_terms = {}
        
        # Load team configuration if it exists
        self._load_team_config()
    
    def add_acronyms(self, acronyms: Dict[str, str]) -> None:
        """
        Add acronyms to the team dictionary.
        
        Args:
            acronyms: Dictionary of acronyms and their expansions
        """
        self.acronyms.update(acronyms)
        self._save_team_config()
    
    def add_spelling_corrections(self, corrections: Dict[str, str]) -> None:
        """
        Add spelling corrections to the team dictionary.
        
        Args:
            corrections: Dictionary of misspellings and their corrections
        """
        self.spelling.update(corrections)
        self._save_team_config()
    
    def add_stopwords(self, stopwords: List[str]) -> None:
        """
        Add custom stopwords to the team list.
        
        Args:
            stopwords: List of stopwords to add
        """
        self.custom_stopwords.extend([w for w in stopwords if w not in self.custom_stopwords])
        self._save_team_config()
    
    def add_domain_terms(self, terms: Dict[str, Any]) -> None:
        """
        Add domain-specific terminology to the team dictionary.
        
        Args:
            terms: Dictionary of domain terms and their definitions/metadata
        """
        self.domain_terms.update(terms)
        self._save_team_config()
    
    def export_config(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Export the team configuration to a file.
        
        Args:
            path: Path to save the configuration to
            
        Returns:
            Path to the exported configuration file
        """
        if path is None:
            path = self.base_path / self.team_name / "config_export.json"
        else:
            path = Path(path)
            
        config_data = {
            "team_name": self.team_name,
            "acronyms": self.acronyms,
            "spelling": self.spelling,
            "custom_stopwords": self.custom_stopwords,
            "domain_terms": self.domain_terms,
        }
        
        with open(path, "w") as f:
            json.dump(config_data, f, indent=2)
            
        logger.info(f"Team configuration exported to {path}")
        return str(path)
    
    def import_config(self, path: Union[str, Path]) -> None:
        """
        Import a team configuration from a file.
        
        Args:
            path: Path to the configuration file
        """
        path = Path(path)
        
        with open(path, "r") as f:
            config_data = json.load(f)
            
        # Update configuration
        if "team_name" in config_data:
            self.team_name = config_data["team_name"]
        if "acronyms" in config_data:
            self.acronyms.update(config_data["acronyms"])
        if "spelling" in config_data:
            self.spelling.update(config_data["spelling"])
        if "custom_stopwords" in config_data:
            self.custom_stopwords.extend([w for w in config_data["custom_stopwords"] 
                                        if w not in self.custom_stopwords])
        if "domain_terms" in config_data:
            self.domain_terms.update(config_data["domain_terms"])
            
        # Save imported configuration
        self._save_team_config()
        logger.info(f"Team configuration imported from {path}")
    
    def merge_configs(self, other_config: "TeamConfig") -> "TeamConfig":
        """
        Merge another team configuration with this one.
        
        Args:
            other_config: Another TeamConfig instance to merge
            
        Returns:
            New TeamConfig instance with merged configuration
        """
        merged = TeamConfig(
            team_name=f"{self.team_name}_merged",
            config=self.config,
            base_path=self.base_path
        )
        
        # Merge dictionaries
        merged.acronyms = {**self.acronyms, **other_config.acronyms}
        merged.spelling = {**self.spelling, **other_config.spelling}
        merged.custom_stopwords = list(set(self.custom_stopwords) | set(other_config.custom_stopwords))
        merged.domain_terms = {**self.domain_terms, **other_config.domain_terms}
        
        # Save merged configuration
        merged._save_team_config()
        return merged
    
    def _save_team_config(self) -> None:
        """Save the team configuration to disk."""
        team_dir = self.base_path / self.team_name
        
        # Save acronyms
        with open(team_dir / "acronyms.json", "w") as f:
            json.dump(self.acronyms, f, indent=2)
            
        # Save spelling corrections
        with open(team_dir / "spelling.json", "w") as f:
            json.dump(self.spelling, f, indent=2)
            
        # Save custom stopwords
        with open(team_dir / "stopwords.json", "w") as f:
            json.dump(self.custom_stopwords, f, indent=2)
            
        # Save domain terms
        with open(team_dir / "domain_terms.json", "w") as f:
            json.dump(self.domain_terms, f, indent=2)
    
    def _load_team_config(self) -> None:
        """Load the team configuration from disk."""
        team_dir = self.base_path / self.team_name
        
        # Load acronyms
        acronyms_path = team_dir / "acronyms.json"
        if acronyms_path.exists():
            with open(acronyms_path, "r") as f:
                self.acronyms = json.load(f)
                
        # Load spelling corrections
        spelling_path = team_dir / "spelling.json"
        if spelling_path.exists():
            with open(spelling_path, "r") as f:
                self.spelling = json.load(f)
                
        # Load custom stopwords
        stopwords_path = team_dir / "stopwords.json"
        if stopwords_path.exists():
            with open(stopwords_path, "r") as f:
                self.custom_stopwords = json.load(f)
                
        # Load domain terms
        domain_terms_path = team_dir / "domain_terms.json"
        if domain_terms_path.exists():
            with open(domain_terms_path, "r") as f:
                self.domain_terms = json.load(f)
    
    def compare_configs(self, other_config: "TeamConfig") -> Dict[str, Any]:
        """
        Compare this configuration with another one.
        
        Args:
            other_config: Another TeamConfig instance to compare
            
        Returns:
            Dictionary with differences between configurations
        """
        differences = {
            "acronyms": {
                "only_in_this": {k: v for k, v in self.acronyms.items() 
                               if k not in other_config.acronyms},
                "only_in_other": {k: v for k, v in other_config.acronyms.items() 
                                if k not in self.acronyms},
                "different_values": {k: (self.acronyms[k], other_config.acronyms[k]) 
                                   for k in set(self.acronyms) & set(other_config.acronyms) 
                                   if self.acronyms[k] != other_config.acronyms[k]}
            },
            "spelling": {
                "only_in_this": {k: v for k, v in self.spelling.items() 
                               if k not in other_config.spelling},
                "only_in_other": {k: v for k, v in other_config.spelling.items() 
                                if k not in self.spelling},
                "different_values": {k: (self.spelling[k], other_config.spelling[k]) 
                                   for k in set(self.spelling) & set(other_config.spelling) 
                                   if self.spelling[k] != other_config.spelling[k]}
            },
            "stopwords": {
                "only_in_this": [w for w in self.custom_stopwords 
                               if w not in other_config.custom_stopwords],
                "only_in_other": [w for w in other_config.custom_stopwords 
                                if w not in self.custom_stopwords]
            },
            "domain_terms": {
                "only_in_this": {k: v for k, v in self.domain_terms.items() 
                               if k not in other_config.domain_terms},
                "only_in_other": {k: v for k, v in other_config.domain_terms.items() 
                                if k not in self.domain_terms},
                "different_values": {k: (self.domain_terms[k], other_config.domain_terms[k]) 
                                   for k in set(self.domain_terms) & set(other_config.domain_terms) 
                                   if self.domain_terms[k] != other_config.domain_terms[k]}
            }
        }
        
        return differences
    
    def to_dataframe(self, data_type: str = "acronyms") -> pd.DataFrame:
        """
        Convert configuration data to a pandas DataFrame.
        
        Args:
            data_type: Type of data to convert ('acronyms', 'spelling', 'domain_terms')
            
        Returns:
            DataFrame with configuration data
        """
        if data_type == "acronyms":
            df = pd.DataFrame([
                {"acronym": k, "expansion": v} 
                for k, v in self.acronyms.items()
            ])
        elif data_type == "spelling":
            df = pd.DataFrame([
                {"misspelling": k, "correction": v} 
                for k, v in self.spelling.items()
            ])
        elif data_type == "stopwords":
            df = pd.DataFrame({"stopword": self.custom_stopwords})
        elif data_type == "domain_terms":
            # Convert domain terms to DataFrame - structure depends on term format
            rows = []
            for term, data in self.domain_terms.items():
                if isinstance(data, dict):
                    row = {"term": term, **data}
                else:
                    row = {"term": term, "definition": str(data)}
                rows.append(row)
            df = pd.DataFrame(rows)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
            
        return df


def get_team_config(
    team_name: str = "default",
    config: Optional[MenoConfig] = None,
    base_path: Optional[Union[str, Path]] = None,
) -> TeamConfig:
    """
    Get a team configuration instance.
    
    Args:
        team_name: Name of the team
        config: Base configuration to use
        base_path: Base path for storing team configuration files
        
    Returns:
        TeamConfig instance
    """
    return TeamConfig(team_name=team_name, config=config, base_path=base_path)