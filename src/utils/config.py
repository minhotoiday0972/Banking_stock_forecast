# src/utils/config.py
import yaml
import os
from typing import Dict, Any
from pathlib import Path

class Config:
    """Centralized configuration management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        self._validate_config()
    
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
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self):
        """Save current configuration to file"""
        with open(self.config_path, 'w', encoding='utf-8') as file:
            yaml.dump(self._config, file, default_flow_style=False, allow_unicode=True)
    
    @property
    def tickers(self):
        return self.get('data.tickers', [])
    
    @property
    def database_dir(self):
        return self.get('data.database_dir', 'data/database')
    
    @property
    def processed_dir(self):
        return self.get('data.processed_dir', 'data/processed')
    
    @property
    def raw_dir(self):
        return self.get('data.raw_dir', 'data/raw')
    
    @property
    def models_dir(self):
        return self.get('paths.models_dir', 'models')
    
    @property
    def outputs_dir(self):
        return self.get('paths.outputs_dir', 'outputs')

# Global config instance
_config_instance = None

def get_config(config_path: str = "config.yaml") -> Config:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance

def reload_config(config_path: str = "config.yaml") -> Config:
    """Reload configuration"""
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance