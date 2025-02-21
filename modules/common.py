import os
import pickle
from comet_ml import Experiment

def initialize_experiment(config):
    """Initialize a CometML experiment (generic version)."""
    if not all([config.api_key, config.project_name, config.workspace]):
        raise ValueError("Missing required CometML configuration values")
    
    return Experiment(
        api_key=config.api_key,
        project_name=config.project_name,
        workspace=config.workspace
    )

def load_dataset(path: str):
    """Load dataset from a pickle file (generic function)."""
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data

def get_tokenizer(config):
    """Get an appropriate tokenizer based on configuration (generic function)."""
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    
    if config.huggingface:
        from transformers import CLIPTokenizer
        return CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32",
            TOKENIZERS_PARALLELISM=False,
            clean_up_tokenization_spaces=True
        )
    else:
        from modules.tokenization_clip import SimpleTokenizer
        return SimpleTokenizer()
