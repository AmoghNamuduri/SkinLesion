"""Build ``outputs/report.md``: every paper table next to our reproduced numbers."""
from __future__ import annotations

from pathlib import Path

from ..utils.io import load_json, resolve
from . import paper_reference as ref
from .plots import plot_paper_vs_ours


def _fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v:.2f}" if pct else f"{v:.4f}"


def _row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def build_report(outputs_dir: str | Path) -> Path:
    out = resolve(outputs_dir)
    ml = load_json(out / "ml" / "results.json") if (out / "ml" / "results.json").exists() else None
    cnn = load_json(out / "cnn" / "results.json") if (out / "cnn" / "results.json").exists() else None
    md = ["# Reproduction report — Shetty et al. (2022), Sci Rep 12:18134", ""]

    if ml:
        md += ["## Table 4 — ML accuracy (mean 10-fold CV on the training split)", "",
               _row("Setting", "Source", *ref.ML_ORDER), _row(*["---"] * (2 + len(ref.ML_ORDER)))]
        for key, label in (("without_augmentation", "100×7, no aug"), ("with_hflip_augmentation", "200×7, h-flip")):
            paper = ref.TABLE4_ML_CV_ACCURACY[key]
            ours = ml.get(key, {})
            md.append(_row(label, "paper", *[_fmt(paper[m]) for m in ref.ML_ORDER]))
            md.append(_row(label, "**ours**", *[_fmt(ours.get(m, {}).get("cv_mean")) for m in ref.ML_ORDER]))
            plot_paper_vs_ours(ref.ML_ORDER, [paper[m] for m in ref.ML_ORDER],
                               [ours.get(m, {}).get("cv_mean", 0) for m in ref.ML_ORDER],
                               out / "figures" / f"table4_{key}.png", f"Table 4 — {label}")
        md.append("")

    md += ["## Table 5 — accuracy and weighted precision / recall / F1 (%)", "",
           _row("Model", "Acc paper", "Acc ours", "P paper", "P ours", "R paper", "R ours", "F1 paper", "F1 ours"),
           _row(*["---"] * 9)]
    aug = (ml or {}).get("with_hflip_augmentation", {})
    for m, paper in ref.TABLE5_SUMMARY.items():
        if m == "CNN":
            t = (cnn or {}).get("test")
            ours = None if not t else {"accuracy": 100 * cnn["train"]["accuracy"],
                                       "precision": 100 * t["weighted"]["precision"],
                                       "recall": 100 * t["weighted"]["recall"], "f1": 100 * t["weighted"]["f1"]}
        else:
            r = aug.get(m)
            ours = None if not r else {"accuracy": 100 * r["cv_mean"],
                                       "precision": 100 * r["test"]["weighted"]["precision"],
                                       "recall": 100 * r["test"]["weighted"]["recall"],
                                       "f1": 100 * r["test"]["weighted"]["f1"]}
        g = (lambda k: _fmt(ours[k], True)) if ours else (lambda k: "—")
        md.append(_row(m, paper["accuracy"], g("accuracy"), paper["precision"], g("precision"),
                       paper["recall"], g("recall"), paper["f1"], g("f1")))
    md += ["", "_Accuracy column: CV mean for ML models, final-epoch train accuracy for the CNN "
               "(the paper's 94/95.18 % is a training accuracy). P/R/F1 are on the held-out test split._", ""]

    if cnn:
        md += ["## Table 6 — CNN train vs test", "", _row("", "Train paper", "Train ours", "Test paper", "Test ours"),
               _row(*["---"] * 5),
               _row("Accuracy (%)", ref.TABLE6_CNN["train"]["accuracy"], _fmt(100 * cnn["train"]["accuracy"], True),
                    ref.TABLE6_CNN["test"]["accuracy"], _fmt(100 * cnn["test"]["accuracy"], True)),
               _row("Loss", ref.TABLE6_CNN["train"]["loss"], _fmt(cnn["train"]["loss"]),
                    ref.TABLE6_CNN["test"]["loss"], _fmt(cnn["test"]["loss"])), ""]
        md += ["## Table 7 — CNN per-class report (test split)", "",
               _row("Class", "P paper", "P ours", "R paper", "R ours", "F1 paper", "F1 ours", "Support paper", "Support ours"),
               _row(*["---"] * 9)]
        pc = cnn["test"]["per_class"]
        for lab, p in ref.TABLE7_CNN_REPORT.items():
            if lab in pc:
                o = pc[lab]
            else:
                key = "macro" if lab.startswith("macro") else "weighted"
                o = dict(cnn["test"][key], support=sum(v["support"] for v in pc.values()))
            md.append(_row(lab, p["precision"], f"{o['precision']:.2f}", p["recall"], f"{o['recall']:.2f}",
                           p["f1"], f"{o['f1']:.2f}", p["support"], o["support"]))
        md += ["", "Figures: `outputs/cnn/history.png` (Fig. 5), `outputs/cnn/confusion_matrix.png`.", ""]

    path = out / "report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    build_original_report(out)
    return path


def build_original_report(outputs_dir: str | Path) -> Path:
    """``outputs/report-original.md``: only the numbers published in the paper (no results of ours)."""
    out = resolve(outputs_dir)
    t4 = ref.TABLE4_ML_CV_ACCURACY
    md = ["# Original paper results — Shetty et al. (2022), Sci Rep 12:18134", "",
          "_Numbers copied verbatim from the published paper (doi:10.1038/s41598-022-22644-9). "
          "Our reproduced numbers are in `report.md`._", "",
          "## Setup reported in the paper", "",
          "- Dataset: HAM10000, 10,015 images, 7 classes; subset of 100 images/class (700), "
          "horizontal-flip augmentation → 200/class (1,400); 80/20 split (1,120 / 280)",
          "- ML: 220×220 images; global features = colour histogram + Hu moments + Haralick texture (HDF5); "
          "LR, LDA, KNN, DT, RF, NB, SVM; 10-fold cross-validation",
          "- CNN: 96×96×3, pixels scaled to 0–1; Conv32 → 2×Conv64 → 2×Conv128 → Dense 1024 → softmax 7 "
          "(BatchNorm, Dropout 0.25/0.5); Adam, lr 0.001–0.00001, batch 32, 150 epochs, categorical cross-entropy",
          "- Environment: Python 3.7.9, Keras/TensorFlow, Windows 10, Intel i5, 8 GB RAM", "",
          "## Table 4 — accuracy of the machine-learning models", "",
          _row("Setting", *ref.ML_ORDER), _row(*["---"] * (1 + len(ref.ML_ORDER))),
          _row("Without augmentation (100×7)", *[t4["without_augmentation"][m] for m in ref.ML_ORDER]),
          _row("With horizontal flip (200×7)", *[t4["with_hflip_augmentation"][m] for m in ref.ML_ORDER]), "",
          "## Table 5 — accuracy and weighted precision / recall / F1 (%)", "",
          _row("Metric", *ref.TABLE5_SUMMARY), _row(*["---"] * (1 + len(ref.TABLE5_SUMMARY)))]
    for k, label in (("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")):
        md.append(_row(label, *[v[k] for v in ref.TABLE5_SUMMARY.values()]))
    t6 = ref.TABLE6_CNN
    md += ["", "## Table 6 — CNN training and testing", "", _row("", "Train", "Test"), _row("---", "---", "---"),
           _row("Accuracy (%)", t6["train"]["accuracy"], t6["test"]["accuracy"]),
           _row("Loss", t6["train"]["loss"], t6["test"]["loss"]), "",
           "## Table 7 — CNN multi-class classification report (test set)", "",
           _row("Class", "Precision", "Recall", "F1-score", "Support"), _row(*["---"] * 5)]
    for lab, v in ref.TABLE7_CNN_REPORT.items():
        md.append(_row(lab, v["precision"], v["recall"], v["f1"], v["support"]))
    md += ["", "## Other reported results", "",
           f"- Headline claim: CNN accuracy **{t6['train']['accuracy']} %** (this is the training accuracy in Table 6)",
           f"- Pre-trained InceptionV3 for comparison: {ref.INCEPTION_V3_ACCURACY} %",
           "- Best ML model: Random Forest (0.87321 with augmentation)", ""]
    path = out / "report-original.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path
