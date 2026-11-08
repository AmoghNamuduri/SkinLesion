"""CNN pipeline (Tables 3, 6, 7 and Fig. 5).

Mirrors the paper's Keras recipe:
* 200×7 images (100 originals + h-flips) at 96×96, pixels scaled to [0, 1]
* one-hot labels in alphabetical class order (LabelBinarizer)
* 80/20 split, test split also used as validation data each epoch
* on-the-fly ImageDataGenerator-style augmentation of the training split
* Adam(lr=1e-3, Keras decay=1e-5 per step), batch 32, 150 epochs,
  categorical cross-entropy; the last-epoch model is evaluated
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..data.augmentation import KerasStyleAugmenter
from ..data.dataset import build_image_dataset
from ..data.splits import split_indices
from ..evaluation.metrics import classification_metrics
from ..evaluation.plots import plot_confusion, plot_history
from ..models.cnn import build_cnn, count_parameters
from ..utils.io import ensure_dir, resolve, save_json


class ArrayDataset(Dataset):
    def __init__(self, images: np.ndarray, targets: np.ndarray, augmenter=None):
        self.images = images.astype(np.float32) / 255.0   # normalise to 0–1 (Step 2)
        self.targets = targets.astype(np.int64)
        self.augmenter = augmenter

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img = self.images[i]
        if self.augmenter is not None:
            img = self.augmenter(img)
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), self.targets[i]


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_optimizer(cfg, model):
    if cfg.cnn.optimizer != "adam":
        raise ValueError(f"unsupported optimizer {cfg.cnn.optimizer}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.cnn.learning_rate, betas=(0.9, 0.999), eps=1e-7)
    decay = float(cfg.cnn.get("lr_decay", 0.0) or 0.0)
    # Keras legacy decay: lr_t = lr0 / (1 + decay * iteration)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0 / (1.0 + decay * step))
    return opt, sched


@torch.no_grad()
def predict(model, loader, device, loss_fn):
    model.eval()
    total, loss_sum, preds, ys = 0, 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += loss_fn(logits, y).item() * len(y)
        total += len(y)
        preds.append(logits.argmax(1).cpu())
        ys.append(y.cpu())
    preds, ys = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    return loss_sum / total, float((preds == ys).mean()), preds, ys


def train_one_epoch(model, loader, opt, sched, loss_fn, device):
    """Returns Keras-style running metrics (computed in training mode, as model.fit reports)."""
    model.train()
    total, loss_sum, correct = 0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        sched.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


def run_cnn_pipeline(cfg) -> dict:
    out_dir = ensure_dir(resolve(cfg.paths.outputs_dir) / "cnn")
    classes = list(cfg.data.classes)
    device = pick_device(cfg.cnn.device)

    ds = build_image_dataset(cfg, cfg.cnn.image_size)
    targets = np.array([classes.index(c) for c in ds.labels])
    tr, te = split_indices(ds.labels, ds.groups, cfg.cnn.test_size, cfg.cnn.split_seed, cfg.data.augment_before_split)
    print(f"CNN data: {len(ds)} images ({len(tr)} train / {len(te)} test), device={device}")

    aug = KerasStyleAugmenter.from_config(cfg.cnn.augmentation, seed=cfg.seed) if cfg.cnn.augmentation else None
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(ArrayDataset(ds.images[tr], targets[tr], aug), batch_size=cfg.cnn.batch_size,
                              shuffle=True, generator=g, num_workers=cfg.cnn.num_workers)
    test_loader = DataLoader(ArrayDataset(ds.images[te], targets[te]), batch_size=cfg.cnn.batch_size,
                             num_workers=cfg.cnn.num_workers)

    model = build_cnn(cfg, len(classes)).to(device)
    print(f"model parameters: {count_parameters(model):,}")
    opt, sched = make_optimizer(cfg, model)
    loss_fn = nn.CrossEntropyLoss()  # softmax + categorical cross-entropy

    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr")}
    t0 = time.time()
    for epoch in range(1, cfg.cnn.epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, opt, sched, loss_fn, device)
        vl, va, _, _ = predict(model, test_loader, device, loss_fn)
        for k, v in zip(hist, (tl, ta, vl, va, opt.param_groups[0]["lr"])):
            hist[k].append(v)
        print(f"epoch {epoch:3d}/{cfg.cnn.epochs}  loss {tl:.4f} acc {ta:.4f}  "
              f"val_loss {vl:.4f} val_acc {va:.4f}  [{time.time() - t0:.0f}s]", flush=True)

    vl, va, preds, ys = predict(model, test_loader, device, loss_fn)
    test = classification_metrics([classes[i] for i in ys], [classes[i] for i in preds], labels=classes)
    test["loss"] = vl
    results = {"train": {"accuracy": hist["train_acc"][-1], "loss": hist["train_loss"][-1]},
               "test": test, "history": hist, "n_train": int(len(tr)), "n_test": int(len(te)),
               "train_seconds": time.time() - t0, "parameters": count_parameters(model)}

    torch.save(model.state_dict(), out_dir / "model.pt")
    save_json(results, out_dir / "results.json")
    plot_history(hist, out_dir / "history.png")
    plot_confusion(test["confusion_matrix"], classes, out_dir / "confusion_matrix.png", "CNN — test split")
    print("\n" + test["report"])
    return results
