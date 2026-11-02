import yaml
from types import SimpleNamespace


def load_config(path: str):
    """Load YAML config file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def dict_to_namespace(d):
    """Convert dict to object-like config."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


def save_config(cfg, path: str):
    """Save config for reproducibility."""
    with open(path, "w") as f:
        yaml.dump(cfg, f)