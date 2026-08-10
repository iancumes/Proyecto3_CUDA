"""Funciones generales de reproducibilidad y persistencia."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> None:
    """Fija semillas de Python, NumPy y TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        pass


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"No se puede serializar {type(value)!r}")


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

