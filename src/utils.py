from pathlib import Path
import random
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path="config.yaml"):
    with open(ROOT / path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)

def prepare_outputs(config):
    for path in config["outputs"].values():
        (ROOT / path).mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
