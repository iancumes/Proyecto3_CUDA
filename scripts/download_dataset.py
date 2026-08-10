"""Descarga y extrae ASL Alphabet desde el endpoint oficial de Kaggle."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
URL = "https://www.kaggle.com/api/v1/datasets/download/grassknoted/asl-alphabet"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if any(RAW.rglob("asl_alphabet_train")):
        print(f"Dataset ya presente en {RAW}; no se duplica.")
        return
    archive = RAW / "asl-alphabet.zip"
    print(f"Descargando {URL} ...")
    with urllib.request.urlopen(URL) as response, archive.open("wb") as target:
        shutil.copyfileobj(response, target)
    print("Extrayendo ...")
    with zipfile.ZipFile(archive) as source:
        source.extractall(RAW)
    archive.unlink()
    print(f"Dataset listo en {RAW}")


if __name__ == "__main__":
    main()

