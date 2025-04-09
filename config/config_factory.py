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
        parser.add_argument('--arch', type=str, default='frame_fusion_moe', help='Architecture name')
        args, _ = parser.parse_known_args()
        
        config_path = None
        
        # Try to use specified config file
        if args.config and os.path.exists(args.config):
            config_path = args.config
        else:
            # Otherwise use architecture-specific config
            arch_config = f"{ConfigFactory.CONFIG_DIR}/{args.arch}_config.yaml"
            if os.path.exists(arch_config):
                config_path = arch_config
            else:
                # Fall back to default config
                default_config = f"{ConfigFactory.CONFIG_DIR}/{ConfigFactory.DEFAULT_CONFIG}"
                if os.path.exists(default_config):
                    config_path = default_config
        
        # Create and return the config object if a valid path was found
        if config_path:
            return Config(config_path)
        else:
            raise ValueError(f"No configuration file found for architecture '{args.arch}'")