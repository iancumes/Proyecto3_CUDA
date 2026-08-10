"""Métricas, matrices de confusión, errores y predicción reusable."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow import keras

from .config import DISPLAY_NAMES, FIGURES_DIR, IMAGE_SIZE, METRICS_DIR, PROJECT_ROOT


def classification_metrics(y_true, y_pred, class_names: list[str]) -> tuple[dict, pd.DataFrame, np.ndarray]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=[DISPLAY_NAMES.get(x, x) for x in class_names],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }
    return metrics, report, cm


def top_confusions(cm: np.ndarray, class_names: list[str], n: int = 10) -> pd.DataFrame:
    rows = []
    for i, actual in enumerate(class_names):
        for j, predicted in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                rows.append(
                    {
                        "actual_class": DISPLAY_NAMES.get(actual, actual),
                        "predicted_class": DISPLAY_NAMES.get(predicted, predicted),
                        "errors": int(cm[i, j]),
                    }
                )
    return pd.DataFrame(rows).sort_values("errors", ascending=False).head(n).reset_index(drop=True)


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    labels = [DISPLAY_NAMES.get(x, x) for x in class_names]
    fig, ax = plt.subplots(figsize=(15, 13))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Clase predicha",
        ylabel="Clase real",
        title="Matriz de confusión - mejor modelo",
    )
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def predecir_imagen(ruta: str | Path, modelo: keras.Model, class_indices: dict[str, int], mostrar: bool = True):
    """Carga, normaliza y clasifica una fotografía nueva."""
    ruta = Path(ruta)
    with Image.open(ruta) as source:
        image = source.convert("RGB")
        resized = image.resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    probabilities = modelo.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    index = int(np.argmax(probabilities))
    inverse = {value: key for key, value in class_indices.items()}
    class_name = inverse[index]
    try:
        display_path = str(ruta.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        display_path = str(ruta)
    result = {
        "path": display_path,
        "class_name": DISPLAY_NAMES.get(class_name, class_name),
        "class_index": index,
        "confidence": float(probabilities[index]),
    }
    if mostrar:
        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.title(f"Predicción: {result['class_name']} | Confianza: {result['confidence']:.2%}")
        plt.axis("off")
        plt.show()
    return result


def parse_custom_label(path: Path) -> str | None:
    token = path.stem.split("_")[0].upper()
    aliases = {"DELETE": "del", "DEL": "del", "NOTHING": "nothing", "SPACE": "space"}
    if token in aliases:
        return aliases[token]
    if len(token) == 1 and token in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return token
    return None
