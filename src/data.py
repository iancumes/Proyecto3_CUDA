"""Auditoría, submuestreo y división reproducible del dataset ASL."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from .config import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    METRICS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_DIR,
    SAMPLES_PER_CLASS,
    TEST_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
)
from .utils import save_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def find_train_source(raw_dir: Path = RAW_DIR) -> Path:
    """Localiza la carpeta que contiene directamente las 29 clases."""
    candidates = [raw_dir / "asl_alphabet_train" / "asl_alphabet_train",
                  raw_dir / "asl_alphabet_train"]
    candidates.extend(p for p in raw_dir.rglob("asl_alphabet_train") if p.is_dir())
    for candidate in candidates:
        subdirs = {p.name for p in candidate.iterdir() if p.is_dir()} if candidate.exists() else set()
        if set(CLASS_NAMES).issubset(subdirs):
            return candidate.resolve()
    raise FileNotFoundError(
        f"No se encontró asl_alphabet_train con sus 29 clases dentro de {raw_dir}."
    )


def image_files(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def audit_dataset(source_dir: Path, force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Inspecciona todos los archivos y guarda un registro verificable."""
    audit_path = METRICS_DIR / "dataset_audit.csv"
    summary_path = METRICS_DIR / "dataset_summary.json"
    if audit_path.exists() and summary_path.exists() and not force:
        audit = pd.read_csv(audit_path)
        import json
        return audit, json.loads(summary_path.read_text(encoding="utf-8"))

    rows: list[dict] = []
    for class_name in CLASS_NAMES:
        class_dir = source_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Falta la clase: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            row = {
                "class_name": class_name,
                "path": str(path.resolve().relative_to(PROJECT_ROOT)),
                "extension": path.suffix.lower(),
                "valid": False,
                "width": np.nan,
                "height": np.nan,
                "channels": np.nan,
                "mode": None,
                "error": None,
            }
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    row.update(
                        valid=True,
                        width=image.width,
                        height=image.height,
                        channels=len(image.getbands()),
                        mode=image.mode,
                    )
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)

    audit = pd.DataFrame(rows)
    audit.to_csv(audit_path, index=False)
    valid = audit[audit["valid"]]
    counts = valid.groupby("class_name").size().reindex(CLASS_NAMES, fill_value=0)
    resolutions = (
        valid.groupby(["width", "height"]).size().sort_values(ascending=False)
        .rename("count").reset_index().to_dict("records")
    )
    summary = {
        "source_dir": str(source_dir.resolve().relative_to(PROJECT_ROOT)),
        "total_files": int(len(audit)),
        "valid_images": int(valid.shape[0]),
        "invalid_images": int((~audit["valid"]).sum()),
        "class_count": int((counts > 0).sum()),
        "class_names": CLASS_NAMES,
        "counts": {k: int(v) for k, v in counts.items()},
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
        "class_difference": int(counts.max() - counts.min()),
        "formats": valid["extension"].value_counts().to_dict(),
        "modes": valid["mode"].value_counts().to_dict(),
        "channels": {str(k): int(v) for k, v in valid["channels"].value_counts().items()},
        "resolutions": resolutions,
    }
    save_json(summary, summary_path)
    distribution = counts.rename("image_count").reset_index()
    distribution["display_class"] = distribution["class_name"].map(DISPLAY_NAMES)
    distribution["percentage"] = 100 * distribution["image_count"] / distribution["image_count"].sum()
    distribution[["display_class", "image_count", "percentage"]].rename(
        columns={"display_class": "class"}
    ).to_csv(METRICS_DIR / "class_distribution.csv", index=False)
    return audit, summary


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _manifest_matches(manifest_path: Path, samples_per_class: int, seed: int) -> bool:
    if not manifest_path.exists():
        return False
    manifest = pd.read_csv(manifest_path)
    expected = samples_per_class * len(CLASS_NAMES)
    return (
        len(manifest) == expected
        and manifest["class_name"].nunique() == len(CLASS_NAMES)
        and set(manifest["split"]) == {"train", "validation", "test"}
        and int(manifest["seed"].iloc[0]) == seed
    )


def prepare_splits(
    audit: pd.DataFrame,
    samples_per_class: int = SAMPLES_PER_CLASS,
    seed: int = RANDOM_SEED,
    force: bool = False,
) -> pd.DataFrame:
    """Selecciona 600 imágenes/clase y crea splits 60/20/20 sin overlap."""
    manifest_path = METRICS_DIR / "split_manifest.csv"
    if not force and _manifest_matches(manifest_path, samples_per_class, seed):
        return pd.read_csv(manifest_path)

    for directory in (TRAIN_DIR, VALIDATION_DIR, TEST_DIR):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    source_root = find_train_source()
    split_dirs = {"train": TRAIN_DIR, "validation": VALIDATION_DIR, "test": TEST_DIR}
    n_train = int(samples_per_class * 0.60)
    n_validation = int(samples_per_class * 0.20)

    valid_audit = audit[audit["valid"]].copy()
    for class_name in CLASS_NAMES:
        paths = valid_audit.loc[valid_audit["class_name"] == class_name, "path"].to_numpy()
        if len(paths) < samples_per_class:
            raise ValueError(f"{class_name}: solo hay {len(paths)} imágenes válidas.")
        selected = rng.choice(paths, size=samples_per_class, replace=False)
        rng.shuffle(selected)
        assignments = (
            ["train"] * n_train
            + ["validation"] * n_validation
            + ["test"] * (samples_per_class - n_train - n_validation)
        )
        for source_str, split in zip(selected, assignments):
            source = PROJECT_ROOT / source_str
            destination = split_dirs[split] / class_name / source.name
            _link_or_copy(source, destination)
            rows.append(
                {
                    "class_name": class_name,
                    "display_class": DISPLAY_NAMES[class_name],
                    "source_path": str(source.relative_to(PROJECT_ROOT)),
                    "relative_source": str(source.relative_to(source_root)),
                    "split": split,
                    "split_path": str(destination.resolve().relative_to(PROJECT_ROOT)),
                    "seed": seed,
                }
            )

    manifest = pd.DataFrame(rows)
    verify_no_leakage(manifest)
    manifest.to_csv(manifest_path, index=False)
    split_table = pd.crosstab(manifest["display_class"], manifest["split"])
    split_table = split_table.reindex(columns=["train", "validation", "test"])
    split_table.to_csv(METRICS_DIR / "split_by_class.csv")
    return manifest


def verify_no_leakage(manifest: pd.DataFrame) -> dict:
    sets = {
        split: set(manifest.loc[manifest["split"] == split, "source_path"])
        for split in ("train", "validation", "test")
    }
    intersections = {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "validation_test": len(sets["validation"] & sets["test"]),
    }
    if any(intersections.values()):
        raise AssertionError(f"Data leakage detectado: {intersections}")
    if manifest["source_path"].duplicated().any():
        raise AssertionError("Hay imágenes fuente repetidas en el manifiesto.")
    counts = pd.crosstab(manifest["class_name"], manifest["split"])
    if counts.shape != (len(CLASS_NAMES), 3) or counts.nunique().max() != 1:
        raise AssertionError("Los splits no son perfectamente estratificados.")
    result = {
        "intersections": intersections,
        "unique_source_images": int(manifest["source_path"].nunique()),
        "total_manifest_rows": int(len(manifest)),
        "all_29_classes_in_each_split": bool((counts > 0).all().all()),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "verified": True,
    }
    save_json(result, METRICS_DIR / "leakage_check.json")
    return result
