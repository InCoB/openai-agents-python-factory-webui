import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

class ConfigManager:
    """
    Hierarchical configuration manager that combines multiple sources:
    - Default values
    - Configuration files
    - Environment variables
    - Runtime overrides
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
            config_path: Optional path to configuration file
        """
        self.config_data = {}
        self.config_sources = {}
        self.config_path = config_path
        
        self._load_defaults()
        
        if config_path:
            self.load_from_file(config_path)
        
        self.load_from_env()
    
    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self.config_data = {
            "api_keys": {
                "openai": os.environ.get("OPENAI_API_KEY", ""),
                "tavily": os.environ.get("TAVILY_API_KEY", "")
            },
            "model_settings": {
                "default_model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 4096
            },
            "logging": {
                "level": "INFO",
                "console_output": True,
                "file_output": False,
                "log_file": "aigen.log"
            },
            "workflow": {
                "max_turns": 80,
                "max_retries": 3,
                "retry_delay": 5
            },
            "ui": {
                "gradio_port": 7860,
                "gradio_host": "0.0.0.0"
            }
        }
        
        for key in self._get_leaf_keys(self.config_data):
            self.config_sources[key] = "defaults"
    
    def _get_leaf_keys(self, data: Dict[str, Any], prefix: str = "") -> List[str]:
        """Get all leaf keys in a nested dictionary."""
        result = []
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.extend(self._get_leaf_keys(value, full_key))
            else:
                result.append(full_key)
        return result
    
    def load_from_file(self, file_path: Union[str, Path]) -> bool:
        """
        Load configuration from a file.
        
            file_path: Path to configuration file (YAML or JSON)
            
            bool: True if successful, False otherwise
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path
        
        if not path.exists():
            return False
        
        try:
            with open(path, "r") as f:
                if path.suffix.lower() in (".yaml", ".yml"):
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)
            
            self._update_config_recursive(file_config, str(path))
            return True
        except Exception as e:
            print(f"Error loading configuration from {path}: {str(e)}")
            return False
    
    def load_from_env(self, prefix: str = "AIGEN_") -> int:
        """
        Load configuration from environment variables.
        
            prefix: Prefix for environment variables
            
            int: Number of variables loaded
        """
        count = 0
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace("_", ".")
                self._set_value(config_key, value, f"env:{key}")
                count += 1
        
        return count
    
    def _update_config_recursive(self, source_dict: Dict[str, Any], 
                               source_name: str, prefix: str = "") -> None:
        """Recursively update configuration from a dictionary."""
        for key, value in source_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self._update_config_recursive(value, source_name, full_key)
            else:
                self._set_value(full_key, value, source_name)
    
    def _set_value(self, key: str, value: Any, source: str) -> None:
        """Set a configuration value with source tracking."""
        parts = key.split(".")
        
        current = self.config_data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
        
        self.config_sources[key] = source
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
            key: The key to look up (dot notation for nested values)
            default: Default value if key is not found
            
        """
        parts = key.split(".")
        
        current = self.config_data
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value at runtime.
        
            key: The key to set (dot notation for nested values)
            value: The value to set
        """
        self._set_value(key, value, "runtime")
    
    def save(self, file_path: Union[str, Path]) -> bool:
        """
        Save the current configuration to a file.
        
            file_path: Path to save the configuration to
            
            bool: True if successful, False otherwise
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path
        
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "w") as f:
                if path.suffix.lower() in (".yaml", ".yml"):
                    yaml.dump(self.config_data, f, default_flow_style=False)
                else:
                    json.dump(self.config_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving configuration to {path}: {str(e)}")
            return False