"""Gráficos reproducibles del EDA y entrenamiento."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

from .config import DISPLAY_NAMES, FIGURES_DIR, IMAGE_SIZE, PROJECT_ROOT, RANDOM_SEED


def plot_class_distribution(distribution: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "class_distribution.png"
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(distribution["class"], distribution["image_count"], color="#276FBF")
    ax.set(title="Distribución original de clases", xlabel="Clase", ylabel="Número de imágenes")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _open_rgb(path: str | Path) -> Image.Image:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with Image.open(path) as image:
        return image.convert("RGB")


def plot_examples(audit: pd.DataFrame, class_names: list[str], per_class: int, filename: str) -> Path:
    rng = np.random.default_rng(RANDOM_SEED)
    fig, axes = plt.subplots(len(class_names), per_class, figsize=(3 * per_class, 2.8 * len(class_names)))
    axes = np.atleast_2d(axes)
    for row, class_name in enumerate(class_names):
        candidates = audit.loc[(audit["valid"]) & (audit["class_name"] == class_name), "path"].to_numpy()
        chosen = rng.choice(candidates, size=per_class, replace=False)
        for col, path in enumerate(chosen):
            axes[row, col].imshow(_open_rgb(path))
            axes[row, col].set_title(DISPLAY_NAMES.get(class_name, class_name))
            axes[row, col].axis("off")
    fig.tight_layout()
    output = FIGURES_DIR / filename
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_resize_comparison(image_path: str | Path) -> Path:
    original = _open_rgb(image_path)
    resized = original.resize(IMAGE_SIZE)
    output = FIGURES_DIR / "resize_comparison.png"
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original)
    axes[0].set_title(f"Original: {original.width}×{original.height}×3")
    axes[1].imshow(resized)
    axes[1].set_title(f"Resize: {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}×3")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_augmentation(image_path: str | Path) -> Path:
    image = _open_rgb(image_path).resize(IMAGE_SIZE)
    array = img_to_array(image)
    generator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        zoom_range=0.1,
    )
    iterator = generator.flow(np.expand_dims(array, 0), batch_size=1, seed=RANDOM_SEED)
    output = FIGURES_DIR / "augmentation_examples.png"
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.ravel()
    axes[0].imshow(image)
    axes[0].set_title("Original")
    for i in range(1, 6):
        augmented = next(iterator)[0].astype(np.uint8)
        axes[i].imshow(augmented)
        axes[i].set_title(f"Aumentada {i}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Augmentation moderado, sin reflexión horizontal")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_history(history: dict, experiment_id: str) -> Path:
    output = FIGURES_DIR / f"history_{experiment_id.lower()}.png"
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["accuracy"], label="Train")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[0].set(title="Accuracy", xlabel="Época", ylabel="Accuracy")
    axes[1].plot(epochs, history["loss"], label="Train")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set(title="Categorical cross-entropy", xlabel="Época", ylabel="Loss")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle(experiment_id)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
