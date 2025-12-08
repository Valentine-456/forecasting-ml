import yaml
from src.utils.path_resolver import CONFIG_DIR

def load_config(name: str):
    path = CONFIG_DIR / name
    with open(path, "r") as f:
        return yaml.safe_load(f)
