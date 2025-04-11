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
        """Update specific parameters from command line arguments"""
        parser = argparse.ArgumentParser(add_help=False)
        
        # general parameters
        parser.add_argument('--eval', action='store_true', help="Evaluation mode")
        
        # model parameters
        parser.add_argument('--eval_path', type=str, help="Path to the checkpoint for evaluation")
        parser.add_argument('--eval_task_id', type=int, help="Task ID to evaluate")
        parser.add_argument('--eval_mode', type=str, choices=['single', 'all'], help="Evaluation mode: 'single' or 'all'")

        # data parameters
        parser.add_argument('--dataset_name', type=str, help="Dataset name")
        parser.add_argument('--videos_dir', type=str, help="Location of videos")
        parser.add_argument('--task_num', type=int, help="Number of tasks")
        parser.add_argument('--path_data', type=str, help="Path to CTVR dataset")

        # experiment parameters
        parser.add_argument('--exp_name', type=str, help="Name of the current experiment")
        parser.add_argument('--output_dir', type=str)

        # system parameters
        parser.add_argument('--seed', type=int, help='Random seed')
        
        args, _ = parser.parse_known_args()
        
        # Update config with command line arguments
        for key, value in vars(args).items():
            if value is not None:
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