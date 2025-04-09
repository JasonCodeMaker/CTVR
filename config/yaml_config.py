import os
import yaml
from typing import Dict, Any, Optional
from modules.basic_utils import mkdirp
import argparse

class Config:
    """Configuration class that loads from YAML files with seed override capability"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from a YAML file
        
        Args:
            config_path: Path to a YAML configuration file
        """
        # Initialize empty config dictionary
        self.config = {}
        
        # Load config from file
        if config_path and os.path.exists(config_path):
            self.load_yaml(config_path)
        else:
            raise ValueError(f"Config file not found: {config_path}")
        
        # Allow overriding via command line
        self._update_from_cmd_args()
            
        # Process critical paths
        self._process_paths()
    
    def load_yaml(self, config_path: str) -> None:
        """Load configuration from YAML file
        
        Args:
            config_path: Path to the YAML file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Handle base config inheritance if _base_ is specified
        if '_base_' in config:
            base_path = os.path.join(os.path.dirname(config_path), config['_base_'])
            if os.path.exists(base_path):
                self.load_yaml(base_path)
            del config['_base_']
            
        # Update config with current values (overriding base)
        self._update_config_recursive(self.config, config)
    
    def _update_config_recursive(self, target: Dict, source: Dict) -> None:
        """Recursively update configuration dictionary
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._update_config_recursive(target[key], value)
            else:
                target[key] = value
    
    def _update_from_cmd_args(self):
        """Update any parameter from command line arguments using dot notation"""
        # Get all command line arguments
        parser = argparse.ArgumentParser(add_help=False)
        args, unknown = parser.parse_known_args()
        
        # Process unknown arguments which should be in format --key.subkey value
        override_dict = {}
        i = 0
        while i < len(unknown):
            arg = unknown[i]
            if arg.startswith('--'):
                param_name = arg[2:]  # Remove leading --
                
                # Check if there's a value after this argument
                if i + 1 >= len(unknown) or unknown[i + 1].startswith('--'):
                    # Boolean flag
                    override_dict[param_name] = True
                    i += 1
                else:
                    # Get the value and attempt type conversion
                    value = unknown[i + 1]
                    try:
                        # Try to convert to int
                        value = int(value)
                    except ValueError:
                        try:
                            # Try to convert to float
                            value = float(value)
                        except ValueError:
                            # Keep as string if not a number
                            if value.lower() == 'true':
                                value = True
                            elif value.lower() == 'false':
                                value = False
                    
                    override_dict[param_name] = value
                    i += 2
            else:
                i += 1
        
        # Process any non-nested args from original parser
        for key, value in vars(args).items():
            if value is not None:
                override_dict[key] = value
        
        # Update nested config values using dot notation
        for key, value in override_dict.items():
            if '.' in key:
                # Handle nested parameters
                keys = key.split('.')
                current = self.config
                
                # Navigate to the innermost dict
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                # Set the value
                current[keys[-1]] = value
            else:
                # Handle top-level parameters
                self.config[key] = value
    
    def _process_paths(self):
        """Process and create necessary directory paths"""
        output_dir = self.config.get('output_dir', './outputs')
        exp_name = self.config.get('exp_name', 'debug')
        model_path = os.path.join(output_dir, exp_name)
        
        # Update config with computed values
        self.config['model_path'] = model_path
        mkdirp(model_path)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value
    
    def __getattr__(self, name: str) -> Any:
        """Allow accessing config items as attributes
        
        Args:
            name: Attribute name
            
        Returns:
            Configuration value
            
        Raises:
            AttributeError: If attribute not found
        """
        # First check normal attributes
        try:
            return super().__getattribute__(name)
        except AttributeError:
            # Then check in config dictionary
            if name in self.config:
                return self.config[name]
                
            # Not found
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Allow setting config items as attributes
        
        Args:
            name: Attribute name
            value: Value to set
        """
        # Special handling for 'config' attribute
        if name == 'config':
            super().__setattr__(name, value)
        else:
            # Set in both attribute and config dict
            super().__setattr__(name, value)
            self.config[name] = value

    def print_config(self):
        """Print all configuration parameters."""
        print("Configuration Parameters:")
        print("=" * 30)
        for key, value in sorted(self.config.items()):
            print(f"{key.ljust(30)}: {value}")
        print("=" * 30)