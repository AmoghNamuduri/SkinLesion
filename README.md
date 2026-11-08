# SkinLesion — reproducing Shetty et al. (2022)

Modular Python reproduction of

> B. Shetty, R. Fernandes, A. P. Rodrigues, R. Chengoden, S. Bhattacharya, K. Lakshmanna.
> **Skin lesion classification of dermoscopic images using machine learning and convolutional neural network.**
> *Scientific Reports* 12, 18134 (2022). doi:[10.1038/s41598-022-22644-9](https://doi.org/10.1038/s41598-022-22644-9) ·
> [PMC9616944](https://pmc.ncbi.nlm.nih.gov/articles/PMC9616944/)

The paper classifies HAM10000 dermoscopic images into 7 classes in two ways:

| Pipeline | What the paper does | Headline result |
|---|---|---|
| **Classical ML** | 220×220 images → global features (HSV colour histogram + Haralick texture + Hu moments) → LR, LDA, KNN, DT, RF, NB, SVM; 10-fold CV | RF 87.3 % (with h-flip augmentation) |
| **Customised CNN** | 96×96 images → 5-conv VGG-style CNN (Table 2), Adam, 150 epochs, random augmentation | 95.18 % train / 86.43 % test |

Both use a class-balanced subset: **100 images per class (700)**, doubled to **200 per class (1,400)** by horizontal flipping, then split 80/20 (1,120 train / 280 test).

---

## Quick start

```bash
pip install -r requirements.txt

# 1. dataset → data/raw/   (see data/README.md; or download it automatically:)
python scripts/download_data.py          # Harvard Dataverse, no login needed

# 2. Step 1 of the paper: class folders + 100-per-class subset
python scripts/prepare_data.py

# 3. reproduce the tables
python scripts/run_ml.py                 # Table 4 + ML columns of Table 5   (~1–2 min)
python scripts/run_cnn.py                # Tables 6, 7 + Fig. 5               (~30 min on a laptop CPU)

# or everything at once
python scripts/run_all.py

# paper vs ours, side by side
python scripts/compare_with_paper.py     # writes outputs/report.md
```

Outputs land in `outputs/`:

| File | Paper equivalent |
|---|---|
| `outputs/report.md` | every table, paper numbers next to ours |
| `outputs/ml/results.json` | Table 4 (CV mean per model, both settings) and Table 5 (test weighted P/R/F1) |
| `outputs/figures/table4_*.png` | Table 4 as paper-vs-ours bar charts |
| `outputs/cnn/results.json` | Tables 6 and 7 |
| `outputs/cnn/history.png` | Fig. 5 (accuracy and loss per epoch) |
| `outputs/cnn/confusion_matrix.png`, `outputs/ml/confusion_*.png` | confusion matrices |
| `outputs/cnn/model.pt` | trained CNN weights |

## Repository layout

```
configs/
  paper.yaml                 every hyper-parameter, annotated with its source table
  variations/no_leakage.yaml example variation (inherits paper.yaml)
src/skinlesion/
  utils/       config (YAML + --set overrides + inheritance), seeding, paths/JSON
  data/        download · metadata · organize (Step 1) · sampling · augmentation (Step 3) · dataset (Step 2) · splits
  features/    descriptors (colour histogram, Haralick, Hu) · extractor (concat + HDF5)
  models/      ml_models (Table 1 registry) · cnn (Table 2)
  training/    ml_trainer (CV + test metrics) · cnn_trainer (Keras-style training loop)
  evaluation/  metrics · plots · paper_reference (published numbers) · report
scripts/       one thin CLI per stage + run_all
tests/         unit tests per block + end-to-end test on synthetic data
mindmap/       interactive map of the codebase (open mindmap/index.html)
data/          dataset placeholder (git-ignored contents)
```

Open **`mindmap/index.html`** in a browser for an interactive map. It groups the files into 9 clusters and draws a line for every dependency. Hover a cluster or file to see its description and functionality, and click to pin the card. The map is generated from the code itself: after you change files, re-run `python mindmap/build_mindmap.py`.

## Modularity — building variations

Every block is swapped through config, not by editing code:

```bash
# one-off overrides
python scripts/run_cnn.py --set cnn.epochs=50 cnn.learning_rate=0.0005
python scripts/run_ml.py  --set ml.histogram_bins=16 ml.scaler=standard

# a named variation = a YAML that inherits from paper.yaml
python scripts/run_all.py --config configs/variations/no_leakage.yaml
```

Extension points:

| To change… | Edit |
|---|---|
| CNN depth/width/dropout | `cnn.architecture.conv_blocks` in YAML (no code) |
| a new CNN (e.g. transfer learning) | add a builder in `models/cnn.py`, select it in `build_cnn` |
| a new classifier | register it in `models/ml_models.ML_REGISTRY` + YAML entry |
| a new image feature | add a function to `features/descriptors.DESCRIPTORS` |
| augmentation | `cnn.augmentation` in YAML, or `data/augmentation.py` |
| split protocol | `data/splits.py` (`data.augment_before_split`) |
| samples per class | `data.images_per_class` |

## Faithfulness notes

**Taken directly from the paper:** HAM10000; 100 images/class; horizontal-flip augmentation to 200/class; 80/20 split; 220×220 for ML and 96×96 for the CNN; pixel scaling to 0–1; one-hot labels in alphabetical class order; colour histogram + Hu moments + Haralick features stored in HDF5; the Table 1 classifier hyper-parameters; 10-fold CV; the full Table 2 architecture; the Table 3 settings (Adam, categorical cross-entropy, 150 epochs, batch 32, learning rate 0.001 decaying toward 0.00001).

**Not stated in the paper. These are the defaults of the public tutorials the pipeline is built on (marked `assumed` in `paper.yaml`):** 8×8×8 HSV bins; MinMax scaling; split seeds (9 for ML, 42 for CNN); ImageDataGenerator ranges (rotation 25°, shift 0.1, shear 0.2, zoom 0.2, horizontal flip); the sampling seed. Table 1 lists "Estimators = 100" for the Decision Tree, but that is not a DT parameter, so DT uses `random_state=9`.

**Framework:** the paper used Keras/TensorFlow 2 on Python 3.7. This repo uses **PyTorch** for the CNN and reproduces the Keras defaults exactly: Glorot-uniform initialization, zero biases, BatchNorm momentum 0.99 / epsilon 1e-3, Conv→ReLU→BN order, Adam epsilon 1e-7, Keras' legacy `decay` learning-rate schedule (`lr / (1 + decay·step)`), and one pass over the 1,120 training images per epoch. Parameter count: 8,677,447.

**How to read the paper's numbers:**
- **Table 4** values are mean 10-fold CV accuracies on the 80 % training split. For example, 0.48929 × 560 = 274 and 0.58125 × 1,120 = 651 exactly.
- The headline **95.18 % is the CNN's training accuracy**. The held-out test accuracy is 86.43 % (Table 6).
- The flip augmentation is applied **before** the split, so an image and its mirror image can end up on opposite sides of it. Colour histograms and Haralick features barely change under a horizontal flip, so this inflates the ML scores, especially RF. `paper.yaml` keeps this behaviour to match the paper; `configs/variations/no_leakage.yaml` keeps each pair on the same side, so you can measure the effect.

Exact numbers will differ somewhat from the paper. The paper samples its 100 images per class at random without giving a seed, uses an unseeded TensorFlow, and evaluates on only 280 test images, where one image is worth 0.36 %. The comparison report puts the two side by side.

## Tests

```bash
python -m pytest -q
```

The tests run in about 20 s and don't need the dataset: `tests/conftest.py` generates a tiny synthetic HAM10000.
