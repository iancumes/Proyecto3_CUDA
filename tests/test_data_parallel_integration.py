"""Comprueba varios grupos, errores de lectura y la interfaz de preparación."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import pytest

from p3 import data


@pytest.fixture
def manifest(tmp_path):
    rows = []
    pixels = np.arange(13 * 19 * 3, dtype=np.uint16).reshape(19, 13, 3)
    for split in ("train", "validation", "test"):
        for class_index, name in enumerate(data.CLASS_INDEX):
            for sample in range(3):
                path = tmp_path / f"{split}-{name}-{sample}.png"
                image = Image.fromarray(((pixels + class_index * 7 + sample) % 256).astype(np.uint8))
                image = image.convert(("RGB", "L", "RGBA")[sample])
                image.save(path)
                rows.append({
                    "split": split, "class_name": name, "source_path": str(path),
                    "relative_source": f"{name}/{path.name}",
                })
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).sample(frac=1, random_state=123).to_csv(path, index=False)
    return path


def test_multiple_chunks_keep_pixels_labels_and_manifest_order(manifest, tmp_path):
    serial_dir, parallel_dir = tmp_path / "serial", tmp_path / "parallel"
    serial = data.prepare_arrays(manifest, serial_dir)
    parallel = data.prepare_arrays(manifest, parallel_dir, workers=2)

    # 87 imágenes por partición ejercitan tres grupos de hasta 32 archivos.
    assert parallel["counts"] == {"train": 87, "validation": 87}
    assert parallel["arrays_sha256"] == serial["arrays_sha256"]
    for split in ("train", "validation"):
        images = np.load(parallel_dir / f"{split}_x.npy")
        labels = np.load(parallel_dir / f"{split}_y.npy")
        np.testing.assert_array_equal(images, np.load(serial_dir / f"{split}_x.npy"))
        np.testing.assert_array_equal(labels, np.repeat(np.arange(29, dtype=np.int32), 3))
        assert images.shape == (87, 64, 64, 3)
        assert images.dtype == np.float32
        assert labels.dtype == np.int32
        assert 0 <= images.min() <= images.max() <= 1
        rows = pd.read_csv(manifest).query("split == @split").sort_values("relative_source")
        for actual, path in zip(images, rows["source_path"], strict=True):
            np.testing.assert_array_equal(actual, data.decode_image(path))
    assert not (parallel_dir / "test_x.npy").exists()
    assert not (parallel_dir / "test_y.npy").exists()


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("bad_image", ["missing", "corrupt"])
def test_image_errors_reach_the_caller(tmp_path, workers, bad_image):
    image = tmp_path / "invalid.jpg"
    if bad_image == "corrupt":
        image.write_bytes(b"not an image")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{
        "split": "train", "class_name": "A", "source_path": str(image),
        "relative_source": "A/invalid.jpg",
    }]).to_csv(manifest, index=False)

    error = FileNotFoundError if bad_image == "missing" else UnidentifiedImageError
    with pytest.raises(error):
        data.prepare_arrays(manifest, tmp_path / "arrays", workers=workers, splits=("train",))


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_worker_count_is_rejected_before_creating_output(tmp_path, workers):
    out = tmp_path / "arrays"
    with pytest.raises(ValueError, match="trabajador"):
        data.prepare_arrays(tmp_path / "missing.csv", out, workers=workers)
    assert not out.exists()


def test_cli_parallel_preparation_and_reserved_test(manifest, tmp_path):
    output = tmp_path / "cli"
    command = [
        sys.executable, "-m", "p3.data", "--manifest", str(manifest),
        "--workers", "2", "--out", str(output),
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[1], timeout=60,
    )
    metadata = json.loads((output / "preparation.json").read_text())
    assert metadata == json.loads(completed.stdout)
    assert metadata["workers"] == 2
    assert metadata["counts"] == {"train": 87, "validation": 87}
    assert metadata["prep_s"] > 0
    assert metadata["arrays_sha256"] == data.fingerprint(output)

    reserved = tmp_path / "reserved"
    subprocess.run(
        command[:-1] + [str(reserved), "--test"], check=True,
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1], timeout=60,
    )
    metadata = json.loads((reserved / "preparation.json").read_text())
    assert metadata["counts"] == {"test": 87}
    assert {path.name for path in reserved.glob("*.npy")} == {"test_x.npy", "test_y.npy"}
    assert metadata["arrays_sha256"] == data.fingerprint(reserved)
