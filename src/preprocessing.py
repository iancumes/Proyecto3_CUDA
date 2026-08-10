"""Carga por lotes y preprocesamiento con ImageDataGenerator."""

from __future__ import annotations

from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .config import BATCH_SIZE, IMAGE_SIZE, RANDOM_SEED, TEST_DIR, TRAIN_DIR, VALIDATION_DIR


def build_generators(augmentation: bool = False, shuffle_train: bool = True):
    """Crea generadores; augmentation se aplica exclusivamente a TRAIN."""
    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1,
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    evaluation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    common = dict(
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    train_it = train_datagen.flow_from_directory(
        TRAIN_DIR, shuffle=shuffle_train, seed=RANDOM_SEED, **common
    )
    validation_it = evaluation_datagen.flow_from_directory(
        VALIDATION_DIR, shuffle=False, **common
    )
    test_it = evaluation_datagen.flow_from_directory(TEST_DIR, shuffle=False, **common)
    return train_it, validation_it, test_it


def generator_contract() -> dict:
    return {
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "class_mode": "categorical",
        "normalization": "rescale=1.0/255.0",
        "validation_augmentation": False,
        "test_augmentation": False,
        "horizontal_flip": False,
    }

