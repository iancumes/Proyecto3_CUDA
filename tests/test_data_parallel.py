import numpy as np
import pandas as pd
from PIL import Image

from p3 import data


def test_serial_and_parallel_prepare_identical_arrays(tmp_path):
    rows = []
    for split in ("train", "validation", "test"):
        for index, name in enumerate(data.CLASS_INDEX):
            file = tmp_path / f"{split}-{name}.jpg"
            Image.new("RGB", (13, 19), (index * 7, 167, 53)).save(file)
            rows.append({
                "split": split, "class_name": name, "source_path": str(file),
                "relative_source": f"{name}/{file.name}",
            })
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(reversed(rows)).to_csv(manifest, index=False)
    serial = data.prepare_arrays(manifest, tmp_path / "serial")
    parallel = data.prepare_arrays(manifest, tmp_path / "parallel", workers=2)

    assert serial["arrays_sha256"] == parallel["arrays_sha256"]
    assert serial["counts"] == {"train": 29, "validation": 29}
    labels = np.load(tmp_path / "parallel/train_y.npy")
    assert labels.tolist() == list(range(29))
    images = np.load(tmp_path / "parallel/train_x.npy")
    assert images.shape == (29, 64, 64, 3)
    assert images.dtype == np.float32
    assert images.min() >= 0 and images.max() <= 1
    assert not (tmp_path / "parallel/test_x.npy").exists()
