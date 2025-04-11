import argparse
import os
from config.yaml_config import Config

class ConfigFactory:
    """Factory class to create configuration objects from YAML files only"""
    
    CONFIG_DIR = "config"
    DEFAULT_CONFIG = "default_config.yaml"
    
    @staticmethod
    def get_config() -> Config:
        """Get configuration based on command-line specified config file or architecture
        
        Returns:
            Config object with loaded configuration
            
        Raises:
            ValueError: If no valid configuration file is found
        """
        # Parse just enough to get the config file or architecture
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--config', type=str, default='config/frame_fusion_moe_config.yaml' ,help='Path to config file')
        args, _ = parser.parse_known_args()
        
        config_path = None
        
        # Try to use specified config file
        if args.config and os.path.exists(args.config):
            config_path = args.config
        else:
            default_config = f"{ConfigFactory.CONFIG_DIR}/{ConfigFactory.DEFAULT_CONFIG}"
            if os.path.exists(default_config):
                config_path = default_config
            else:
                raise ValueError("No valid configuration file found. Please specify a valid config file or ensure the default config exists.")
        
        # Load the configuration
        return Config(config_path)
