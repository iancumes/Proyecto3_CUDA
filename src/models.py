"""Arquitecturas basadas en el ejemplo CNN de gatos y perros."""

from __future__ import annotations

from tensorflow import keras

from .config import IMAGE_SIZE

INPUT_SHAPE = (*IMAGE_SIZE, 3)
N_CLASSES = 29


def _compile(model: keras.Model, learning_rate: float) -> keras.Model:
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn1(learning_rate: float = 0.01) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=INPUT_SHAPE),
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"
            ),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu", kernel_initializer="he_uniform"),
            keras.layers.Dense(N_CLASSES, activation="softmax"),
        ],
        name="cnn_modelo_1",
    )
    return _compile(model, learning_rate)


def build_cnn2(learning_rate: float = 0.01, dropout: float = 0.0) -> keras.Model:
    layers = [
        keras.layers.Input(shape=INPUT_SHAPE),
        keras.layers.Conv2D(
            32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"
        ),
        keras.layers.MaxPooling2D((2, 2)),
    ]
    if dropout:
        layers.append(keras.layers.Dropout(dropout))
    layers.extend(
        [
            keras.layers.Conv2D(
                64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"
            ),
            keras.layers.MaxPooling2D((2, 2)),
        ]
    )
    if dropout:
        layers.append(keras.layers.Dropout(dropout))
    layers.extend(
        [
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu", kernel_initializer="he_uniform"),
            keras.layers.Dense(N_CLASSES, activation="softmax"),
        ]
    )
    model = keras.Sequential(layers, name=f"cnn_modelo_2_dropout_{dropout:g}")
    return _compile(model, learning_rate)


def build_fully_connected(learning_rate: float = 0.01) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=INPUT_SHAPE),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu", kernel_initializer="he_uniform"),
            keras.layers.Dense(N_CLASSES, activation="softmax"),
        ],
        name="fully_connected_128",
    )
    return _compile(model, learning_rate)


def build_fully_connected_variant(learning_rate: float = 0.01) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=INPUT_SHAPE),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu", kernel_initializer="he_uniform"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(N_CLASSES, activation="softmax"),
        ],
        name="fully_connected_256_dropout_01",
    )
    return _compile(model, learning_rate)


def architecture_text(model_key: str) -> str:
    descriptions = {
        "CNN1": "Conv2D(32)-MaxPool-Flatten-Dense(128)-Softmax(29)",
        "CNN2": "Conv2D(32)-MaxPool-Conv2D(64)-MaxPool-Flatten-Dense(128)-Softmax(29)",
        "CNN3_D01": "CNN2 + Dropout(0.1) después de cada MaxPool",
        "CNN3_D02": "CNN2 + Dropout(0.2) después de cada MaxPool",
        "CNN3_D01_LR01": "CNN2 + Dropout(0.1), SGD learning_rate=0.01",
        "CNN_AUG": "Mejor CNN de validación reentrenada con augmentation moderado",
        "FC1": "Flatten-Dense(128)-Softmax(29)",
        "FC2": "Flatten-Dense(256)-Dropout(0.1)-Softmax(29)",
        "SVM_LINEAR": "Flatten-StandardScaler-PCA-SVC(kernel=linear,C=1)",
        "SVM_RBF": "Flatten-StandardScaler-PCA-SVC(kernel=rbf,C=1)",
    }
    return descriptions[model_key]
