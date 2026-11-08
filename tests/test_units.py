"""Fast unit tests for individual blocks (no dataset needed)."""
import numpy as np
import torch

from conftest import ROOT
from skinlesion.data.augmentation import KerasStyleAugmenter, hflip_offline
from skinlesion.data.splits import split_indices
from skinlesion.features import extract_global_features
from skinlesion.models.cnn import SkinLesionCNN, count_parameters
from skinlesion.models.ml_models import build_ml_models
from skinlesion.utils import load_config


def test_config_overrides_and_inheritance():
    cfg = load_config(ROOT / "configs" / "paper.yaml", ["cnn.epochs=3", "data.augment_before_split=false"])
    assert cfg.cnn.epochs == 3 and cfg.data.augment_before_split is False
    var = load_config(ROOT / "configs" / "variations" / "no_leakage.yaml")
    assert var.data.augment_before_split is False and var.cnn.batch_size == 32


def test_global_features_shape():
    imgs = np.random.default_rng(0).integers(0, 255, (3, 220, 220, 3), dtype=np.uint8)
    X = extract_global_features(imgs, bins=8)
    assert X.shape == (3, 512 + 13 + 7)


def test_hflip_doubles_and_keeps_groups():
    imgs = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    im2, lab2, g2, aug = hflip_offline(imgs, np.array(["a", "b"]), np.array(["x", "y"]))
    assert len(im2) == 4 and aug.sum() == 2 and list(g2) == ["x", "y", "x", "y"]
    assert np.array_equal(im2[2], imgs[0][:, ::-1])


def test_group_split_has_no_leakage():
    groups = np.repeat(np.arange(50).astype(str), 2)
    labels = np.repeat(np.array(list("abcde") * 10), 2)
    tr, te = split_indices(labels, groups, 0.2, 0, augment_before_split=False)
    assert not set(groups[tr]) & set(groups[te])
    assert len(te) == 20


def test_augmenter_preserves_shape():
    aug = KerasStyleAugmenter(25, .1, .1, .2, .2, True, seed=0)
    img = np.random.default_rng(0).random((96, 96, 3)).astype(np.float32)
    assert aug(img).shape == img.shape


def test_cnn_matches_table2():
    m = SkinLesionCNN(num_classes=7, image_size=96).eval()
    assert m(torch.zeros(2, 3, 96, 96)).shape == (2, 7)
    # Table 2 by hand: convs 896+18,496+36,928+73,856+147,584, BN γ/β 64+128+128+256+256+2,048,
    # dense 8,192·1024+1,024 = 8,389,632, output 1,024·7+7 = 7,175  →  8,677,447 trainable
    assert count_parameters(m) == 8_677_447


def test_ml_registry_builds_table1():
    cfg = load_config(ROOT / "configs" / "paper.yaml")
    models = build_ml_models(cfg.ml.models)
    assert list(models) == ["LR", "LDA", "KNN", "DT", "RF", "NB", "SVM"]
    assert models["RF"].n_estimators == 200 and models["SVM"].kernel == "linear"
