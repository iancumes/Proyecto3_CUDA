"""Orquestador completo del Laboratorio 3."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras

from .config import (
    BATCH_SIZE,
    CLASS_NAMES,
    CUSTOM_TEST_DIR,
    DISPLAY_NAMES,
    FIGURES_DIR,
    IMAGE_SIZE,
    METRICS_DIR,
    RANDOM_SEED,
    SAMPLES_PER_CLASS,
    PROJECT_ROOT,
)
from .data import audit_dataset, find_train_source, prepare_splits, verify_no_leakage
from .evaluation import parse_custom_label, predecir_imagen
from .experiments import (
    analyze_best_model,
    evaluate_all_on_test,
    lock_selection_before_test,
    train_all_neural,
    train_svm_experiments,
)
from .preprocessing import build_generators, generator_contract
from .utils import save_json, set_global_seed
from .visualization import (
    plot_augmentation,
    plot_class_distribution,
    plot_examples,
    plot_resize_comparison,
)


def create_eda_artifacts(audit: pd.DataFrame) -> None:
    distribution = pd.read_csv(METRICS_DIR / "class_distribution.csv")
    plot_class_distribution(distribution)
    plot_examples(audit, ["A", "B", "C", "D", "E"], 2, "five_letters.png")
    plot_examples(audit, ["A"], 6, "within_class_A.png")
    plot_examples(audit, ["space", "del", "nothing"], 3, "special_classes.png")
    plot_examples(audit, ["M", "N", "S"], 3, "similar_MNS.png")
    plot_examples(audit, ["U", "V", "R"], 3, "similar_UVR.png")
    example_path = audit.loc[(audit["valid"]) & (audit["class_name"] == "A"), "path"].iloc[0]
    plot_resize_comparison(example_path)
    plot_augmentation(example_path)


def create_misclassification_figure() -> None:
    prediction_path = METRICS_DIR / "best_keras_test_predictions.npz"
    if not prediction_path.exists():
        return
    data = np.load(prediction_path, allow_pickle=True)
    y_true, y_pred = data["y_true"], data["y_pred"]
    probabilities, filepaths = data["probabilities"], data["filepaths"]
    class_names = data["class_names"].tolist()
    errors = np.flatnonzero(y_true != y_pred)
    if not len(errors):
        return
    confidence = probabilities[np.arange(len(y_pred)), y_pred]
    selected = errors[np.argsort(confidence[errors])[::-1][:12]]
    rows = []
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for ax, index in zip(axes.ravel(), selected):
        with Image.open(PROJECT_ROOT / filepaths[index]) as image:
            ax.imshow(image.convert("RGB"))
        actual = DISPLAY_NAMES.get(class_names[int(y_true[index])], class_names[int(y_true[index])])
        predicted = DISPLAY_NAMES.get(class_names[int(y_pred[index])], class_names[int(y_pred[index])])
        conf = float(confidence[index])
        ax.set_title(f"Real: {actual}\nPred.: {predicted} ({conf:.1%})", fontsize=9)
        ax.axis("off")
        rows.append(
            {"path": str(filepaths[index]), "actual": actual, "predicted": predicted, "confidence": conf}
        )
    for ax in axes.ravel()[len(selected):]:
        ax.axis("off")
    fig.suptitle("Errores de mayor confianza del mejor modelo Keras")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "misclassified_examples.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(METRICS_DIR / "misclassified_examples.csv", index=False)


def evaluate_custom_photos(selection: dict) -> dict:
    image_paths = sorted(
        p for p in CUSTOM_TEST_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        result = {
            "status": "PENDIENTE: fotografías reales de los integrantes.",
            "image_count": 0,
            "required_location": "data/custom_test/<integrante>/",
            "required_naming": "A_01.jpg, B_01.jpg, ... (mínimo 5 letras distintas por integrante)",
        }
        save_json(result, METRICS_DIR / "custom_test_status.json")
        return result

    _, _, test_it = build_generators(augmentation=False, shuffle_train=False)
    model = keras.models.load_model(Path(__file__).resolve().parents[1] / "outputs/models/best_asl_model.keras")
    rows = []
    for path in image_paths:
        actual = parse_custom_label(path)
        prediction = predecir_imagen(path, model, test_it.class_indices, mostrar=False)
        rows.append(
            {
                "path": str(path),
                "actual": DISPLAY_NAMES.get(actual, actual) if actual else None,
                "predicted": prediction["class_name"],
                "confidence": prediction["confidence"],
                "correct": bool(actual and prediction["class_name"] == DISPLAY_NAMES.get(actual, actual)),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(METRICS_DIR / "custom_test_predictions.csv", index=False)
    labeled = results[results["actual"].notna()]
    status = {
        "status": "COMPLETADO" if len(labeled) == len(results) else "PARCIAL: hay nombres sin etiqueta reconocible",
        "image_count": int(len(results)),
        "labeled_count": int(len(labeled)),
        "accuracy": float(labeled["correct"].mean()) if len(labeled) else None,
    }
    save_json(status, METRICS_DIR / "custom_test_status.json")
    return status


def create_comparison_and_conclusions(selection: dict, best_analysis: dict, custom_status: dict) -> None:
    experiments = pd.read_csv(METRICS_DIR / "experiments.csv")
    display = {
        "CNN1": "CNN Modelo 1",
        "CNN2": "CNN Modelo 2",
        "CNN3_D01": "CNN Dropout 0.1",
        "CNN3_D02": "CNN Dropout 0.2",
        "CNN3_D01_LR01": "CNN Dropout 0.1 (LR 0.01)",
        "CNN_AUG": "CNN + Augmentation",
        "FC1": "Fully Connected",
        "FC2": "Fully Connected 256 + Dropout 0.1",
        "SVM_LINEAR": "SVM Linear",
        "SVM_RBF": "SVM RBF",
    }
    experiments["Modelo"] = experiments["experiment_id"].map(display)
    experiments["gap_train_validation"] = experiments["train_accuracy"] - experiments["validation_accuracy"]
    columns = {
        "train_accuracy": "Train Accuracy",
        "validation_accuracy": "Validation Accuracy",
        "test_accuracy": "Test Accuracy",
        "macro_precision": "Macro Precision",
        "macro_recall": "Macro Recall",
        "macro_f1": "Macro F1",
        "test_loss": "Test Loss",
        "epochs_trained": "Épocas",
        "parameters": "Parámetros",
        "training_time": "Tiempo (s)",
        "augmentation": "Augmentation",
        "notes": "Comentarios",
    }
    comparison = experiments[["Modelo", *columns]].rename(columns=columns)
    comparison.to_csv(METRICS_DIR / "final_comparison.csv", index=False)

    by_id = experiments.set_index("experiment_id")
    best_id = selection["best_overall_by_validation"]
    conclusions = {
        "selection_basis": "Selección congelada usando validation antes de consultar test.",
        "best_model_by_validation": best_id,
        "best_validation_accuracy": float(by_id.loc[best_id, "validation_accuracy"]),
        "best_test_accuracy": float(by_id.loc[best_id, "test_accuracy"]),
        "second_convolution_validation_change": float(
            by_id.loc["CNN2", "validation_accuracy"] - by_id.loc["CNN1", "validation_accuracy"]
        ),
        "dropout_01_vs_cnn2_validation_change": float(
            by_id.loc["CNN3_D01", "validation_accuracy"] - by_id.loc["CNN2", "validation_accuracy"]
        ),
        "dropout_02_vs_cnn2_validation_change": float(
            by_id.loc["CNN3_D02", "validation_accuracy"] - by_id.loc["CNN2", "validation_accuracy"]
        ),
        "better_dropout": (
            "0.1" if by_id.loc["CNN3_D01", "validation_accuracy"] >= by_id.loc["CNN3_D02", "validation_accuracy"] else "0.2"
        ),
        "learning_rate_001_to_01_validation_change": float(
            by_id.loc["CNN3_D01_LR01", "validation_accuracy"]
            - by_id.loc["CNN3_D01", "validation_accuracy"]
        ),
        "augmentation_vs_selected_base_validation_change": float(
            by_id.loc["CNN_AUG", "validation_accuracy"]
            - by_id.loc[
                __import__("json").loads((METRICS_DIR / "cnn_selection_before_test.json").read_text())["best_base_cnn_by_validation"],
                "validation_accuracy",
            ]
        ),
        "fully_connected_vs_best_cnn_validation_change": float(
            by_id.loc["FC1", "validation_accuracy"]
            - by_id.loc[selection["best_keras_by_validation"], "validation_accuracy"]
        ),
        "fully_connected_variant_validation_change": float(
            by_id.loc["FC2", "validation_accuracy"] - by_id.loc["FC1", "validation_accuracy"]
        ),
        "best_svm": selection["best_svm_by_validation"],
        "best_svm_validation_accuracy": float(by_id.loc[selection["best_svm_by_validation"], "validation_accuracy"]),
        "best_model_analysis": best_analysis,
        "custom_photos": custom_status,
    }
    save_json(conclusions, METRICS_DIR / "conclusions.json")


def run_complete_lab(force: bool = False) -> dict:
    set_global_seed(RANDOM_SEED)
    source = find_train_source()
    audit, dataset_summary = audit_dataset(source, force=force)
    create_eda_artifacts(audit)
    manifest = prepare_splits(audit, samples_per_class=SAMPLES_PER_CLASS, force=force)
    leakage = verify_no_leakage(manifest)
    save_json(generator_contract(), METRICS_DIR / "preprocessing_config.json")
    train_all_neural(force=force)
    train_svm_experiments(manifest, force=force)
    selection = lock_selection_before_test()
    experiments = evaluate_all_on_test(manifest, selection, force=force)
    best_analysis = analyze_best_model(selection)
    create_misclassification_figure()
    custom_status = evaluate_custom_photos(selection)
    create_comparison_and_conclusions(selection, best_analysis, custom_status)
    run_summary = {
        "dataset": dataset_summary,
        "samples_per_class": SAMPLES_PER_CLASS,
        "selected_total": int(len(manifest)),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "leakage": leakage,
        "selection": selection,
        "custom_test": custom_status,
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": keras.__version__,
        "random_seed": RANDOM_SEED,
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
    }
    save_json(run_summary, METRICS_DIR / "run_summary.json")
    return run_summary


if __name__ == "__main__":
    run_complete_lab(force=False)
