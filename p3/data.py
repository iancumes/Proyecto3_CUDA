"""Manifiesto original y preparación de tensores en CPU."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image

from src.config import CLASS_NAMES, IMAGE_SIZE, METRICS_DIR, PROJECT_ROOT
from src.data import audit_dataset, find_train_source, prepare_splits, verify_no_leakage

CLASS_INDEX = {name: index for index, name in enumerate(sorted(CLASS_NAMES))}
SHAPE = (*IMAGE_SIZE, 3)


def decode_image(path):
    """Replica RGB, interpolación nearest y escala del generador original."""
    with Image.open(path) as source:
        image = source.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.NEAREST)
        return np.asarray(image, dtype=np.float32) * np.float32(1 / 255)


def dataset_rows(manifest, split):
    rows = pd.read_csv(manifest)
    return rows.loc[rows["split"] == split].sort_values("relative_source")


def fingerprint(directory):
    digest = hashlib.sha256()
    for path in sorted(Path(directory).glob("*.npy")):
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def prepare_arrays(manifest, out, workers=1, splits=("train", "validation")):
    """Prepara arrays mapeados a disco y devuelve tiempo y huella del resultado."""
    if workers < 1:
        raise ValueError("Se necesita al menos un trabajador.")
    decode_many = lambda paths: map(decode_image, paths)
    if workers > 1:
        from .data_parallel import decode_parallel
        def decode_many(paths):
            return decode_parallel(paths, workers)

    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    counts = {}
    for split in splits:
        rows = dataset_rows(manifest, split)
        if not len(rows):
            raise ValueError(f"El manifiesto no contiene imágenes de {split}.")
        paths = [PROJECT_ROOT / path for path in rows["source_path"]]
        tensor = np.lib.format.open_memmap(
            out / f"{split}_x.npy", mode="w+", dtype=np.float32, shape=(len(paths), *SHAPE)
        )
        for index, array in enumerate(decode_many(paths), start=1):
            tensor[index - 1] = array
            if index % 3000 == 0:
                print(f"{split}: {index}/{len(paths)} imágenes", flush=True)
        tensor.flush()
        del tensor
        np.save(out / f"{split}_y.npy", rows["class_name"].map(CLASS_INDEX).to_numpy(dtype=np.int32))
        counts[split] = len(paths)
    elapsed = perf_counter() - start
    return {"prep_s": elapsed, "workers": workers, "counts": counts, "arrays_sha256": fingerprint(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", action="store_true", help="Auditar y generar el manifiesto de Lab3Datos.")
    parser.add_argument("--manifest", type=Path, default=METRICS_DIR / "split_manifest.csv")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--test", action="store_true", help="Preparar solo el conjunto reservado de prueba.")
    args = parser.parse_args()
    if args.setup:
        audit, _ = audit_dataset(find_train_source())
        print(json.dumps(verify_no_leakage(prepare_splits(audit)), indent=2))
        return
    if args.out is None:
        parser.error("Usa --setup o indica --out.")
    splits = ("test",) if args.test else ("train", "validation")
    metrics = prepare_arrays(args.manifest, args.out, args.workers, splits)
    (args.out / "preparation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
