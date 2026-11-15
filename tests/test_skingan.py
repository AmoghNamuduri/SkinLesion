"""SkinGAN blocks on CPU with tiny dummy networks — no GPU, no StyleGAN clone, no downloads."""
import json
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from conftest import ROOT
from skinlesion.data.splits import class_names, encode_labels, lesion_level_split
from skinlesion.evaluation.memorization import audit, embed
from skinlesion.evaluation.metrics import auc_score, bootstrap_ci, probabilistic_metrics
from skinlesion.gan.dataset_export import center_crop_resize, export_stylegan_dataset, read_archive_labels
from skinlesion.gan.domain_loss import DomainLoss, class_map_for
from skinlesion.gan.dp_training import dp_step_
from skinlesion.gan.generate import generate_dataset
from skinlesion.gan.train_gan import build_command
from skinlesion.gan.visualize import make_grid
from skinlesion.utils import load_config

sys.path.insert(0, str(ROOT / "third_party"))
import setup_stylegan  # noqa: E402


def skingan_cfg(fake_cfg, tmp_path, extra=()):
    over = [f"paths.raw_dir={fake_cfg.paths.raw_dir}", f"paths.splits_dir={tmp_path / 'splits'}",
            f"paths.gan_dataset={tmp_path / 'gan.zip'}", f"paths.gan_runs={tmp_path / 'runs'}",
            "gan_data.resolution=32", *extra]
    return load_config(ROOT / "configs" / "skingan" / "baseline.yaml", over)


def test_configs_inherit():
    base = load_config(ROOT / "configs" / "skingan" / "baseline.yaml")
    sk = load_config(ROOT / "configs" / "skingan" / "skingan.yaml")
    test = load_config(ROOT / "gan_test" / "test_config.yaml")
    assert base.gan.domain_loss.enabled is False and sk.gan.domain_loss.enabled is True
    assert sk.gan.kimg == base.gan.kimg and test.gan.kimg == 100
    assert base.data.classes == ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def test_lesion_level_split_no_leakage():
    rng = np.random.default_rng(0)
    rows = [{"lesion_id": f"L{i}", "image_id": f"I{i}_{j}", "dx": ["nv", "mel", "bkl"][i % 3]}
            for i in range(300) for j in range(rng.integers(1, 4))]
    df = lesion_level_split(pd.DataFrame(rows), 0.1, 0.2, seed=1)
    owners = df.groupby("lesion_id")["split"].nunique()
    assert (owners == 1).all()
    frac = df.drop_duplicates("lesion_id")["split"].value_counts(normalize=True)
    assert abs(frac["test"] - 0.2) < 0.03 and abs(frac["val"] - 0.1) < 0.03
    shuffled = lesion_level_split(pd.DataFrame(rows).sample(frac=1, random_state=3), 0.1, 0.2, seed=1)
    assert df.set_index("image_id")["split"].equals(shuffled.set_index("image_id")["split"])  # order-independent


def test_label_sets():
    cfg = load_config(ROOT / "configs" / "skingan" / "baseline.yaml", ["labels.set=binary"])
    assert class_names(cfg) == ["benign", "malignant"]
    assert list(encode_labels(["mel", "nv", "malignant", "benign"], cfg)) == [1, 0, 1, 0]
    cfg7 = load_config(ROOT / "configs" / "skingan" / "baseline.yaml")
    assert list(encode_labels(["akiec", "vasc"], cfg7)) == [0, 6]


def test_center_crop_resize():
    img = np.zeros((450, 600, 3), np.uint8)
    img[:, 75:525] = 255  # the centred 450×450 square is white
    out = center_crop_resize(img, 64)
    assert out.shape == (64, 64, 3) and out.min() == 255


def test_export_stylegan_archive(fake_cfg, tmp_path):
    cfg = skingan_cfg(fake_cfg, tmp_path)
    info = export_stylegan_dataset(cfg)
    labels, names = read_archive_labels(info["path"])
    assert names == list(cfg.data.classes) and len(labels) == info["images"]
    test_ids = set(pd.read_csv(tmp_path / "splits" / "ham10000_test.csv")["image_id"])
    with zipfile.ZipFile(info["path"]) as zf:
        stored = {n.split("/")[-1].split(".")[0] for n in zf.namelist() if n.endswith(".jpg")}
    assert stored and not stored & test_ids  # the GAN never sees test images


def test_domain_loss_gradients_reach_generator_only():
    torch.manual_seed(0)
    clf = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(4, 7))
    before = [p.clone() for p in clf.parameters()]
    gen = nn.Parameter(torch.zeros(2, 3, 32, 32))  # stand-in for G's output
    loss_fn = DomainLoss(clf, weight=0.1, image_size=16).train()
    c = torch.eye(7)[[1, 4]]
    a = loss_fn(torch.tanh(gen), c)
    b = loss_fn(torch.tanh(gen), c)
    assert torch.allclose(a, b)  # classifier stays in eval mode (no dropout noise)
    a.backward()
    assert gen.grad is not None and gen.grad.abs().sum() > 0
    assert all(p.grad is None for p in clf.parameters())
    assert all(torch.equal(x, y) for x, y in zip(before, clf.parameters()))


def test_domain_loss_binary_class_map():
    clf = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 1))
    cmap = class_map_for(["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"], ["benign", "malignant"], ["mel", "bcc", "akiec"])
    assert cmap == [1, 1, 0, 0, 1, 0, 0]
    loss = DomainLoss(clf, 1.0, 8, cmap)(torch.zeros(3, 3, 8, 8), torch.eye(7)[[0, 2, 4]])
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_build_command_and_spec(fake_cfg, tmp_path):
    cfg = skingan_cfg(fake_cfg, tmp_path, ["gan.kimg=50", "gan.gamma=2"])
    cmd = " ".join(build_command(cfg))
    for flag in ("--cond=1", "--cfg=paper256", "--kimg=50", "--resume=ffhq256", "--target=0.6",
                 "--augpipe=bgcfnc", "--gamma=2", "--mirror=1"):
        assert flag in cmd


def test_stylegan_patch_text():
    loss_src = ("import torch\nfrom torch_utils.ops import conv2d_gradfix\n\n"
                + setup_stylegan.LOSS_GMAIN_OLD + "                more()\n")
    patched = setup_stylegan.patch_loss(loss_src)
    assert "SKINGAN_DOMAIN_LOSS" in patched and "loss_Gmain + loss_domain" in patched
    assert setup_stylegan.patch_loss(patched) == patched
    misc = "x\n" + setup_stylegan.MISC_OLD + setup_stylegan.SAMPLER_OLD
    patched_misc = setup_stylegan.patch_misc(misc)
    assert "shape == tensor.shape" in patched_misc and "super().__init__()" in patched_misc


class DummyG(nn.Module):
    z_dim, c_dim = 8, 7

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8 + 7, 3 * 16 * 16)

    def forward(self, z, c, truncation_psi=1.0, noise_mode="const"):
        return torch.tanh(self.lin(torch.cat([z, c], 1))).view(-1, 3, 16, 16)


def test_generate_dataset_manifest(tmp_path):
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    m = generate_dataset(DummyG(), classes, {"mel": 3, "df": 2}, tmp_path / "syn")
    assert len(m) == 5 and set(m["dx"]) == {"mel", "df"}
    from skinlesion.data.dataset import load_manifest
    df = load_manifest(tmp_path / "syn")
    assert all(__import__("os").path.exists(p) for p in df["path"])


def test_probabilistic_metrics_and_ci():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.eye(3)[y] * 0.8 + 0.2 / 3
    m = probabilistic_metrics(y, p, ["a", "b", "c"])
    assert m["auc"] == pytest.approx(1.0) and m["balanced_accuracy"] == 1.0
    yb = np.array([0, 1] * 50)
    ci = bootstrap_ci(yb, np.c_[1 - yb * 0.9, yb * 0.9], n=50)
    assert ci["low"] <= ci["estimate"] <= ci["high"]
    assert auc_score([0, 1, 0, 1], np.array([0.1, 0.9, 0.2, 0.8])) == 1.0


def test_memorization_detects_copies():
    rng = np.random.default_rng(0)
    train = rng.integers(0, 255, (40, 32, 32, 3), dtype=np.uint8)
    test = rng.integers(0, 255, (40, 32, 32, 3), dtype=np.uint8)
    copies = train[:10].copy()
    rep = audit(embed(copies, "pixels"), embed(train, "pixels"), embed(test, "pixels"))
    assert rep["frac_below"] == 1.0 and rep["syn_median"] < rep["test_median"]


def test_dp_step_clips_then_noises():
    w = nn.Parameter(torch.zeros(1000))
    w.grad = torch.full((1000,), 10.0)
    norm = dp_step_([w], max_norm=1.0, noise_multiplier=0.0)
    assert norm == pytest.approx(316.23, rel=1e-3) and w.grad.norm().item() == pytest.approx(1.0, rel=1e-4)
    w.grad = torch.zeros(1000)
    dp_step_([w], max_norm=1.0, noise_multiplier=2.0, generator=torch.Generator().manual_seed(0))
    assert 1.5 < w.grad.std().item() < 2.5  # noise std = σ·C


def test_make_grid_shape():
    tiles = [[np.zeros((10, 10, 3), np.uint8)] * 3] * 2
    g = make_grid(tiles, ["a", "b"], tile=20, pad=2, label_w=30)
    assert g.shape == (2 * 22 + 2, 30 + 3 * 22 + 2, 3)


def test_classifier_training_and_synthetic_learning(fake_cfg, tmp_path):
    """Domain classifier → generated set → synthetic-learning comparison, end to end on tiny data."""
    pytest.importorskip("torchvision")
    from skinlesion.models.densenet import load_classifier
    from skinlesion.training.domain_classifier import train_domain_classifier
    from skinlesion.training.synthetic_learning import run_synthetic_learning

    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    generate_dataset(DummyG(), classes, {c: 3 for c in classes}, tmp_path / "syn")
    cfg = skingan_cfg(fake_cfg, tmp_path, [
        f"paths.domain_classifier={tmp_path / 'clf' / 'domain_classifier.pt'}", f"paths.outputs_dir={tmp_path / 'out'}",
        "domain_classifier.pretrained=false", "domain_classifier.image_size=32", "domain_classifier.epochs=1",
        "domain_classifier.batch_size=8", "domain_classifier.num_workers=0", "domain_classifier.device=cpu"])
    res = train_domain_classifier(cfg)
    assert 0.0 <= res["test"]["auc"] <= 1.0
    model, meta = load_classifier(res["checkpoint"])
    assert meta["classes"] == classes and meta["image_size"] == 32

    cfg.synthetic_learning = {"sources": {"real": "real", "gan": str(tmp_path / "syn"), "missing": str(tmp_path / "nope")},
                              "train_size": None, "seeds": [0], "select_on_real_val": True, "epochs": 1, "bootstrap": 20}
    out = run_synthetic_learning(cfg)
    assert set(out["sources"]) == {"real", "gan"}  # missing sources are skipped, not fatal
    assert (tmp_path / "out" / "synthetic_learning" / "results.json").exists()


def test_densenet_shapes():
    pytest.importorskip("torchvision")
    from skinlesion.models.densenet import build_densenet, probabilities
    m = build_densenet(2, pretrained=False).eval()
    out = m(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, 1) and probabilities(out).shape == (1, 2)
    assert build_densenet(7, pretrained=False)(torch.zeros(1, 3, 64, 64)).shape == (1, 7)
