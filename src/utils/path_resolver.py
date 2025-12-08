from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "Rodrigues_dane"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
CONFIG_DIR = ROOT / "config"
OUTPUT_DIR = ROOT / "output"
