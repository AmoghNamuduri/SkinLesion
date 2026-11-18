# SkinGAN

**SkinGAN = conditional StyleGAN2-ADA + a 7-class classifier that gives feedback to the generator.**
This folder runs the full SkinGAN experiment on Google Colab, as proposed in the paper (Sections III–IV).

```
L_G = L_adv + λ · CrossEntropy( DenseNet(G(z, c)), c )        λ = 0.1, DenseNet frozen
```

## What's in this folder

| File | Purpose |
|---|---|
| `SkinGAN_Colab.ipynb` | the whole pipeline, steps A–F (open this in Colab) |
| `colab_baseline.yaml` | **standard GAN** arm: no classifier feedback |
| `colab_skingan.yaml` | **SkinGAN** arm: identical, with the classifier loss on |
| `colab_experiment.yaml` | the classification comparison (5 training sets, real test images) |
| `compare_grids.py` | picture: real \| standard GAN \| SkinGAN per class, plus progression strips |

The code these files call lives in the library, so it's shared with the rest of the project:

| Piece | Code |
|---|---|
| Lesion-level split, label sets | `src/skinlesion/data/splits.py` |
| 256² StyleGAN training archive | `src/skinlesion/gan/dataset_export.py`, `scripts/export_gan_dataset.py` |
| StyleGAN launcher + multi-session resume | `src/skinlesion/gan/train_gan.py`, `scripts/train_gan.py` |
| **Classifier-guided loss** | `src/skinlesion/gan/domain_loss.py`, hooked into StyleGAN by `third_party/setup_stylegan.py` |
| Domain classifier (DenseNet-121) | `src/skinlesion/models/densenet.py`, `src/skinlesion/training/domain_classifier.py` |
| Synthetic images | `src/skinlesion/gan/generate.py`, `scripts/generate_synthetic.py` |
| Classification experiment | `src/skinlesion/training/synthetic_learning.py`, `scripts/run_synthetic_learning.py` |
| Metrics, report, memorization check | `src/skinlesion/evaluation/` |

## Experimental design

1. **Shared checkpoint:** both GANs start from the same snapshot (the finished `gan_test` run).
2. **Standard GAN** and **SkinGAN** each train `EXTRA_KIMG` more (default 300), with identical settings except
   the classifier loss, so any difference comes from the classifier feedback alone.
3. **Domain classifier:** DenseNet-121 (ImageNet) trained on the real training split; frozen inside SkinGAN.
4. **Classification experiment:** the same DenseNet-121 is trained on each training set and tested on the
   same real test images:

| Training set | Rare classes topped up to 1,000 with |
|---|---|
| `real` | nothing (nv capped at 1,500) |
| `traditional_aug` | copies of real images (+ flips, rotations, crops during training) |
| `real_plus_gan` | standard-GAN images |
| `real_plus_skingan` | SkinGAN images |
| `skingan_only` | synthetic images only (SpineGAN-style check) |

Metrics: accuracy, balanced accuracy, macro F1, recall for df / vasc / akiec, AUC with 95% bootstrap CI.
GAN quality: FID (optional). Privacy: memorization check (optional).

## How to run (Google Colab)

**Before the first session**
1. Let the `gan_test` run finish; its last snapshot becomes the shared starting point.
2. Commit and push this folder plus the updated `src/` files, because the notebook clones the repo from GitHub.

**Every session**
1. Colab → **File → Upload notebook** → `skinGan/SkinGAN_Colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU.**
3. Run cells **0–5** (settings, GPU, Drive + code, StyleGAN setup, data, shared checkpoint).
4. Run the **next unfinished step**. Every step skips itself when done and resumes after a disconnect, so
   when in doubt, just run the steps in order:

| Step | What | T4 time (estimate) | Sessions |
|---|---|---|---|
| A | domain classifier | ~20–40 min | 1 |
| B | standard GAN, +300 kimg | ~7 h | 1–2 |
| C | SkinGAN, +300 kimg | ~8–9 h | 1–2 |
| D | synthetic images | ~10 min | every session that runs E or F |
| E | classification experiment (1 seed) | ~1 h | 1 |
| F | report + pictures | ~5 min | — |

**About 17–19 GPU-hours in total**, spread over roughly 3–4 Colab sessions. To spend less, set `EXTRA_KIMG = 200`
in cell 0 (about 11–12 h for B + C). Use the same value for both GANs.

**Outputs** (Google Drive, `MyDrive/SkinGAN/skingan_outputs/`):
`domain_classifier/` (model + results) · `gan/baseline/`, `gan/skingan/` (snapshots + progress grids) ·
`synthetic_learning/results.json` · `report-skingan.md` · `figures/` (comparison pictures, ROC curves).

**Tips**
- Keep the tab open during B and C (free Colab disconnects idle sessions).
- Run the optional "free Drive space" cell after B and C; each snapshot is a few hundred MB.
- In step C's log, check for `+ SkinGAN domain loss` and `[SkinGAN] domain loss on`. That confirms the
  classifier feedback is active.
- The classifier-loss path has been tested on CPU with small stand-in networks, but not yet in a full GPU run.
  If step C errors, paste the message and it can be patched like the earlier StyleGAN fixes.
