import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skinlesion.utils import load_config  # noqa: E402

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


@pytest.fixture
def fake_cfg(tmp_path):
    """A tiny synthetic HAM10000 (12 coloured-blob images per class) + a fast config pointing at it."""
    raw = tmp_path / "raw" / "HAM10000_images_part_1"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for ci, dx in enumerate(CLASSES):
        for k in range(12):
            img = np.full((60, 80, 3), 40 + 25 * ci, np.uint8)
            cv2.circle(img, (40, 30), 10 + ci * 2, (int(rng.integers(0, 255)), 30 * ci, 200 - 20 * ci), -1)
            iid = f"ISIC_{ci:02d}{k:03d}"
            cv2.imwrite(str(raw / f"{iid}.jpg"), img)
            rows.append({"lesion_id": f"HAM_{ci}{k}", "image_id": iid, "dx": dx,
                         "dx_type": "histo", "age": 50, "sex": "male", "localization": "back"})
    pd.DataFrame(rows).to_csv(tmp_path / "raw" / "HAM10000_metadata.csv", index=False)

    return load_config(ROOT / "configs" / "paper.yaml", [
        f"paths.raw_dir={tmp_path / 'raw'}",
        f"paths.organized_dir={tmp_path / 'organized'}",
        f"paths.subset_dir={tmp_path / 'subset'}",
        f"paths.features_file={tmp_path / 'features.h5'}",
        f"paths.outputs_dir={tmp_path / 'outputs'}",
        "data.images_per_class=10",
        "ml.kfold=3", "ml.image_size=64",
        "cnn.epochs=2", "cnn.image_size=48", "cnn.batch_size=16",
    ])
