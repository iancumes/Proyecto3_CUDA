"""Ejecución y registro de los experimentos del laboratorio."""

from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow import keras

from .config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DISPLAY_NAMES,
    FIGURES_DIR,
    IMAGE_SIZE,
    METRICS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    TEST_DIR,
)
from .evaluation import classification_metrics, save_confusion_matrix, top_confusions
from .models import (
    architecture_text,
    build_cnn1,
    build_cnn2,
    build_fully_connected,
    build_fully_connected_variant,
)
from .preprocessing import build_generators
from .utils import load_json, save_json, set_global_seed
from .visualization import plot_history

EXPERIMENTS_PATH = METRICS_DIR / "experiments.csv"
TEST_EVALUATION_NOTE = "Test evaluado únicamente después de congelar selección."

CNN_CONFIGS = {
    "CNN1": {"builder": build_cnn1, "learning_rate": 0.01, "dropout": 0.0,
             "epochs": 10, "early_stopping": False},
    "CNN2": {"builder": build_cnn2, "learning_rate": 0.01, "dropout": 0.0,
             "epochs": 10, "early_stopping": False},
    "CNN3_D01": {"builder": build_cnn2, "learning_rate": 0.001, "dropout": 0.1,
                 "epochs": 15, "early_stopping": True},
    "CNN3_D02": {"builder": build_cnn2, "learning_rate": 0.001, "dropout": 0.2,
                 "epochs": 15, "early_stopping": True},
    "CNN3_D01_LR01": {"builder": build_cnn2, "learning_rate": 0.01, "dropout": 0.1,
                      "epochs": 15, "early_stopping": True},
    "FC1": {"builder": build_fully_connected, "learning_rate": 0.01, "dropout": 0.0,
            "epochs": 10, "early_stopping": False},
    "FC2": {"builder": build_fully_connected_variant, "learning_rate": 0.01, "dropout": 0.1,
            "epochs": 10, "early_stopping": False},
}


def _model_path(experiment_id: str) -> Path:
    return MODELS_DIR / f"{experiment_id.lower()}.keras"


def _history_path(experiment_id: str) -> Path:
    return METRICS_DIR / f"history_{experiment_id.lower()}.json"


def _load_experiments() -> pd.DataFrame:
    return pd.read_csv(EXPERIMENTS_PATH) if EXPERIMENTS_PATH.exists() else pd.DataFrame()


def _upsert(row: dict) -> None:
    current = _load_experiments()
    if not current.empty:
        current = current[current["experiment_id"] != row["experiment_id"]]
    pd.concat([current, pd.DataFrame([row])], ignore_index=True).to_csv(EXPERIMENTS_PATH, index=False)


def _with_test_evaluation_note(notes: object) -> str:
    """Agrega la nota de test una sola vez, incluso al forzar una reevaluación."""
    text = str(notes).replace(f" {TEST_EVALUATION_NOTE}", "").strip()
    return f"{text} {TEST_EVALUATION_NOTE}".strip()


def _build_from_config(experiment_id: str, config: dict) -> keras.Model:
    if experiment_id == "CNN1":
        return config["builder"](learning_rate=config["learning_rate"])
    if experiment_id.startswith("FC"):
        return config["builder"](learning_rate=config["learning_rate"])
    return config["builder"](
        learning_rate=config["learning_rate"], dropout=config["dropout"]
    )


def train_keras_experiment(
    experiment_id: str,
    config: dict,
    augmentation: bool = False,
    force: bool = False,
    architecture_override: str | None = None,
) -> dict:
    model_path = _model_path(experiment_id)
    history_path = _history_path(experiment_id)
    existing = _load_experiments()
    if (
        not force
        and model_path.exists()
        and history_path.exists()
        and not existing.empty
        and experiment_id in set(existing["experiment_id"])
    ):
        return existing.loc[existing["experiment_id"] == experiment_id].iloc[-1].to_dict()

    set_global_seed(RANDOM_SEED)
    keras.backend.clear_session()
    train_it, validation_it, _ = build_generators(augmentation=augmentation, shuffle_train=True)
    # Para evaluar train usamos un generador sin augmentation y sin shuffle.
    train_eval_it, _, _ = build_generators(augmentation=False, shuffle_train=False)
    model = _build_from_config(experiment_id if experiment_id != "CNN_AUG" else config["base_id"], config)
    callbacks = []
    if config.get("early_stopping"):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        )

    started = time.perf_counter()
    history_obj = model.fit(
        train_it,
        validation_data=validation_it,
        epochs=config["epochs"],
        callbacks=callbacks,
        verbose=1,
    )
    training_time = time.perf_counter() - started
    history = {key: [float(v) for v in values] for key, values in history_obj.history.items()}
    best_epoch = int(np.argmin(history["val_loss"]) + 1)
    train_loss, train_accuracy = model.evaluate(train_eval_it, verbose=0)
    validation_loss, validation_accuracy = model.evaluate(validation_it, verbose=0)
    model.save(model_path)
    save_json(history, history_path)
    plot_history(history, experiment_id)

    row = {
        "experiment_id": experiment_id,
        "model_name": experiment_id,
        "architecture": architecture_override or architecture_text(experiment_id),
        "learning_rate": config["learning_rate"],
        "dropout": config.get("dropout", 0.0),
        "batch_size": BATCH_SIZE,
        "epochs_requested": config["epochs"],
        "epochs_trained": len(history["loss"]),
        "best_epoch": best_epoch,
        "augmentation": bool(augmentation),
        "train_accuracy": float(train_accuracy),
        "validation_accuracy": float(validation_accuracy),
        "train_loss": float(train_loss),
        "validation_loss": float(validation_loss),
        "training_time": float(training_time),
        "parameters": int(model.count_params()),
        "test_accuracy": np.nan,
        "test_loss": np.nan,
        "macro_precision": np.nan,
        "macro_recall": np.nan,
        "macro_f1": np.nan,
        "notes": "Test no consultado durante selección; resultados de test se agregan después.",
    }
    _upsert(row)
    del model, train_it, validation_it, train_eval_it
    keras.backend.clear_session()
    gc.collect()
    return row


def train_all_neural(force: bool = False) -> pd.DataFrame:
    for experiment_id, config in CNN_CONFIGS.items():
        train_keras_experiment(experiment_id, config, force=force)

    experiments = _load_experiments()
    cnn_ids = ["CNN1", "CNN2", "CNN3_D01", "CNN3_D02", "CNN3_D01_LR01"]
    best_base_id = (
        experiments[experiments["experiment_id"].isin(cnn_ids)]
        .sort_values(["validation_accuracy", "validation_loss"], ascending=[False, True])
        .iloc[0]["experiment_id"]
    )
    base = CNN_CONFIGS[best_base_id].copy()
    base.update({"epochs": 15, "early_stopping": True, "base_id": best_base_id})
    train_keras_experiment(
        "CNN_AUG",
        base,
        augmentation=True,
        force=force,
        architecture_override=f"{architecture_text(best_base_id)} + augmentation moderado",
    )
    save_json(
        {
            "best_base_cnn_by_validation": best_base_id,
            "criterion": "Mayor validation_accuracy; validation_loss como desempate",
            "test_used_for_selection": False,
        },
        METRICS_DIR / "cnn_selection_before_test.json",
    )
    return _load_experiments()


def load_flat_split(manifest: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray]:
    subset = manifest[manifest["split"] == split].reset_index(drop=True)
    class_to_index = {name: index for index, name in enumerate(sorted(CLASS_NAMES))}
    x = np.empty((len(subset), IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3), dtype=np.float32)
    y = np.empty(len(subset), dtype=np.int32)
    for i, row in subset.iterrows():
        with Image.open(PROJECT_ROOT / row["split_path"]) as image:
            array = np.asarray(image.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32) / 255.0
        x[i] = array.reshape(-1)
        y[i] = class_to_index[row["class_name"]]
    return x, y


def train_svm_experiments(manifest: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    existing = _load_experiments()
    svm_ids = ["SVM_LINEAR", "SVM_RBF"]
    if (
        not force
        and all((MODELS_DIR / f"{x.lower()}.joblib").exists() for x in svm_ids)
        and not existing.empty
        and set(svm_ids).issubset(set(existing["experiment_id"]))
    ):
        return existing

    x_train, y_train = load_flat_split(manifest, "train")
    x_validation, y_validation = load_flat_split(manifest, "validation")
    for experiment_id, kernel in (("SVM_LINEAR", "linear"), ("SVM_RBF", "rbf")):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler(copy=False)),
                ("pca", PCA(n_components=100, whiten=True, svd_solver="randomized",
                            iterated_power=2, random_state=RANDOM_SEED)),
                ("svm", SVC(kernel=kernel, C=1, cache_size=2048)),
            ]
        )
        started = time.perf_counter()
        pipeline.fit(x_train.copy(), y_train)
        training_time = time.perf_counter() - started
        train_accuracy = float(pipeline.score(x_train.copy(), y_train))
        validation_accuracy = float(pipeline.score(x_validation.copy(), y_validation))
        joblib.dump(pipeline, MODELS_DIR / f"{experiment_id.lower()}.joblib", compress=3)
        _upsert(
            {
                "experiment_id": experiment_id,
                "model_name": f"SVM {kernel.upper()}",
                "architecture": architecture_text(experiment_id),
                "learning_rate": np.nan,
                "dropout": np.nan,
                "batch_size": np.nan,
                "epochs_requested": np.nan,
                "epochs_trained": np.nan,
                "best_epoch": np.nan,
                "augmentation": False,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
                "train_loss": np.nan,
                "validation_loss": np.nan,
                "training_time": training_time,
                "parameters": np.nan,
                "test_accuracy": np.nan,
                "test_loss": np.nan,
                "macro_precision": np.nan,
                "macro_recall": np.nan,
                "macro_f1": np.nan,
                "notes": "StandardScaler y PCA(100) ajustados solo con TRAIN; C=1.",
            }
        )
        del pipeline
        gc.collect()
    del x_train, x_validation, y_train, y_validation
    gc.collect()
    return _load_experiments()


def lock_selection_before_test() -> dict:
    experiments = _load_experiments()
    keras_rows = experiments[experiments["experiment_id"].isin(list(CNN_CONFIGS) + ["CNN_AUG"])]
    best_keras = keras_rows.sort_values(
        ["validation_accuracy", "validation_loss"], ascending=[False, True]
    ).iloc[0]["experiment_id"]
    svm_rows = experiments[experiments["experiment_id"].isin(["SVM_LINEAR", "SVM_RBF"])]
    best_svm = svm_rows.sort_values("validation_accuracy", ascending=False).iloc[0]["experiment_id"]
    best_overall_validation = experiments.sort_values("validation_accuracy", ascending=False).iloc[0]["experiment_id"]
    selection = {
        "best_keras_by_validation": best_keras,
        "best_svm_by_validation": best_svm,
        "best_overall_by_validation": best_overall_validation,
        "criterion": "Mayor validation_accuracy; validation_loss como desempate en Keras",
        "test_used_for_selection": False,
    }
    save_json(selection, METRICS_DIR / "selection_before_test.json")
    return selection


def evaluate_all_on_test(manifest: pd.DataFrame, selection: dict, force: bool = False) -> pd.DataFrame:
    experiments = _load_experiments()
    if (
        not force
        and experiments["test_accuracy"].notna().all()
        and (METRICS_DIR / "final_evaluation_complete.json").exists()
    ):
        experiments["notes"] = experiments["notes"].map(_with_test_evaluation_note)
        experiments.to_csv(EXPERIMENTS_PATH, index=False)
        return experiments

    _, _, test_it = build_generators(augmentation=False, shuffle_train=False)
    class_names = [name for name, index in sorted(test_it.class_indices.items(), key=lambda x: x[1])]
    keras_ids = list(CNN_CONFIGS) + ["CNN_AUG"]
    for experiment_id in keras_ids:
        model = keras.models.load_model(_model_path(experiment_id))
        test_it.reset()
        test_loss, _ = model.evaluate(test_it, verbose=0)
        test_it.reset()
        probabilities = model.predict(test_it, verbose=0)
        y_true = test_it.classes
        y_pred = probabilities.argmax(axis=1)
        metrics, report, cm = classification_metrics(y_true, y_pred, class_names)
        report.to_csv(METRICS_DIR / f"classification_report_{experiment_id.lower()}.csv")
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            METRICS_DIR / f"confusion_matrix_{experiment_id.lower()}.csv"
        )
        row = experiments.loc[experiments["experiment_id"] == experiment_id].iloc[-1].to_dict()
        row.update(metrics)
        row["test_loss"] = float(test_loss)
        row["notes"] = _with_test_evaluation_note(row["notes"])
        _upsert(row)
        if experiment_id == selection["best_keras_by_validation"]:
            np.savez_compressed(
                METRICS_DIR / "best_keras_test_predictions.npz",
                y_true=y_true,
                y_pred=y_pred,
                probabilities=probabilities,
                filepaths=np.array([
                    str(Path(path).resolve().relative_to(PROJECT_ROOT)) for path in test_it.filepaths
                ]),
                class_names=np.array(class_names),
            )
        del model, probabilities
        keras.backend.clear_session()
        gc.collect()

    x_test, y_test = load_flat_split(manifest, "test")
    for experiment_id in ("SVM_LINEAR", "SVM_RBF"):
        pipeline = joblib.load(MODELS_DIR / f"{experiment_id.lower()}.joblib")
        y_pred = pipeline.predict(x_test.copy())
        metrics, report, cm = classification_metrics(y_test, y_pred, sorted(CLASS_NAMES))
        report.to_csv(METRICS_DIR / f"classification_report_{experiment_id.lower()}.csv")
        pd.DataFrame(cm, index=sorted(CLASS_NAMES), columns=sorted(CLASS_NAMES)).to_csv(
            METRICS_DIR / f"confusion_matrix_{experiment_id.lower()}.csv"
        )
        experiments = _load_experiments()
        row = experiments.loc[experiments["experiment_id"] == experiment_id].iloc[-1].to_dict()
        row.update(metrics)
        row["notes"] = _with_test_evaluation_note(row["notes"])
        _upsert(row)
        del pipeline, y_pred
        gc.collect()
    del x_test, y_test

    experiments = _load_experiments()
    best_keras_id = selection["best_keras_by_validation"]
    source_model = _model_path(best_keras_id)
    destination = MODELS_DIR / "best_asl_model.keras"
    shutil.copy2(source_model, destination)
    loaded = keras.models.load_model(destination)
    _, _, verify_test_it = build_generators(augmentation=False, shuffle_train=False)
    batch = next(verify_test_it)[0][:1]
    prediction = loaded.predict(batch, verbose=0)
    save_json(
        {
            "model_source_experiment": best_keras_id,
            "loaded_successfully": True,
            "prediction_shape": list(prediction.shape),
            "probabilities_sum": float(prediction[0].sum()),
        },
        METRICS_DIR / "model_reload_check.json",
    )
    save_json(
        {"completed": True, "selection_was_locked_before_test": True},
        METRICS_DIR / "final_evaluation_complete.json",
    )
    return _load_experiments()


def analyze_best_model(selection: dict) -> dict:
    best_id = selection["best_keras_by_validation"]
    cm_path = METRICS_DIR / f"confusion_matrix_{best_id.lower()}.csv"
    report_path = METRICS_DIR / f"classification_report_{best_id.lower()}.csv"
    cm_df = pd.read_csv(cm_path, index_col=0)
    report = pd.read_csv(report_path, index_col=0)
    cm = cm_df.to_numpy()
    class_names = list(cm_df.index)
    confusions = top_confusions(cm, class_names, n=10)
    confusions.to_csv(METRICS_DIR / "top_10_confusions.csv", index=False)
    save_confusion_matrix(cm, class_names, FIGURES_DIR / "confusion_matrix_best.png")
    per_class = report.loc[[DISPLAY_NAMES.get(x, x) for x in class_names]].copy()
    recalls = per_class["recall"].sort_values()

    def pair_errors(group: list[str]) -> int:
        indices = [class_names.index(x) for x in group]
        return int(sum(cm[i, j] for i in indices for j in indices if i != j))

    analysis = {
        "best_keras_experiment": best_id,
        "lowest_recall": recalls.head(5).to_dict(),
        "highest_recall": recalls.tail(5).sort_values(ascending=False).to_dict(),
        "m_n_s_internal_errors": pair_errors(["M", "N", "S"]),
        "u_v_r_internal_errors": pair_errors(["U", "V", "R"]),
        "top_confusions": confusions.to_dict("records"),
    }
    save_json(analysis, METRICS_DIR / "best_model_analysis.json")
    return analysis
