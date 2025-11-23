# src/utils/config.py
import yaml
import os
from typing import Dict, Any
from pathlib import Path

class Config:
    """Centralized configuration management"""
    
    def __init__(self, config_path: str = "config.yaml", overrides: Dict[str, Any] = None):
        self.config_path = config_path
        self._config = self._load_config()
        if overrides:
            self.merge(overrides)
        self._validate_config()

    def _recursive_merge(self, base_dict, override_dict):
        """Helper function to recursively merge dictionaries."""
        for k, v in override_dict.items():
            if k in base_dict and isinstance(base_dict[k], dict) and isinstance(v, dict):
                base_dict[k] = self._recursive_merge(base_dict[k], v)
            else:
                base_dict[k] = v
        return base_dict

    def merge(self, override_dict: Dict[str, Any]):
        """Merge an override dictionary into the current config."""
        self._config = self._recursive_merge(self._config, override_dict)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
    
    def _validate_config(self):
        """Validate required configuration sections"""
        required_sections = ['data', 'features', 'training', 'models', 'paths']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required config section: {section}")
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation (e.g., 'data.tickers')"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value if value is not None else default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Return the entire configuration dictionary."""
        return self._config
    
    def __getitem__(self, key: str):
        """Allow dictionary-style access to config."""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return self.get(key) is not None

# Global config instance for backward compatibility
_global_config = None

def get_config(config_path: str = "config.yaml", overrides: Dict[str, Any] = None) -> Config:
    """
    Get configuration instance. Returns global instance if no overrides provided.
    """
    global _global_config
    
    if overrides is None and _global_config is not None:
        return _global_config
    
    config = Config(config_path, overrides=overrides)
    
    if overrides is None and _global_config is None:
        _global_config = config
    
    return config

def reload_config(config_path: str = "config.yaml", overrides: Dict[str, Any] = None) -> Config:
    """Reloads and returns a new configuration object."""
    return get_config(config_path, overrides=overrides)