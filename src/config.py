"""Configuración central del laboratorio."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
CUSTOM_TEST_DIR = DATA_DIR / "custom_test"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"

RANDOM_SEED = 123
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 64
SAMPLES_PER_CLASS = 600
TRAIN_SPLIT = 0.60
VALIDATION_SPLIT = 0.20
TEST_SPLIT = 0.20

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
DISPLAY_NAMES = {**{letter: letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
                 "del": "DELETE", "nothing": "NOTHING", "space": "SPACE"}

for directory in (DATA_DIR, RAW_DIR, CUSTOM_TEST_DIR, OUTPUTS_DIR,
                  FIGURES_DIR, METRICS_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

