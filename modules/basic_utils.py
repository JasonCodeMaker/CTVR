import os
import ujson as json
import numpy as np
import random
import torch
import shutil
from pathlib import Path


def seed_everything(seed_value=-1):
    if seed_value >= 0:
        seed = seed_value
    else:
        seed = random.randint(0, 10000)
    
    # Set seeds for different libraries
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
    # Configure CUDNN for deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed

def load_checkpoint(model: torch.nn.Module, ckpt_path: Path):
    """Load model weights from checkpoint."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    print("Checkpoint loaded successfully")

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]

def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)

def deletedir(p):
    if os.path.exists(p):
        shutil.rmtree(p)