# Original paper results — Shetty et al. (2022), Sci Rep 12:18134

_Numbers copied verbatim from the published paper (doi:10.1038/s41598-022-22644-9). Our reproduced numbers are in `report.md`._

## Setup reported in the paper

- Dataset: HAM10000, 10,015 images, 7 classes; subset of 100 images/class (700), horizontal-flip augmentation → 200/class (1,400); 80/20 split (1,120 / 280)
- ML: 220×220 images; global features = colour histogram + Hu moments + Haralick texture (HDF5); LR, LDA, KNN, DT, RF, NB, SVM; 10-fold cross-validation
- CNN: 96×96×3, pixels scaled to 0–1; Conv32 → 2×Conv64 → 2×Conv128 → Dense 1024 → softmax 7 (BatchNorm, Dropout 0.25/0.5); Adam, lr 0.001–0.00001, batch 32, 150 epochs, categorical cross-entropy
- Environment: Python 3.7.9, Keras/TensorFlow, Windows 10, Intel i5, 8 GB RAM

## Table 4 — accuracy of the machine-learning models

| Setting | LR | LDA | KNN | DT | RF | NB | SVM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Without augmentation (100×7) | 0.48929 | 0.35179 | 0.41786 | 0.44286 | 0.58571 | 0.34107 | 0.42143 |
| With horizontal flip (200×7) | 0.58125 | 0.57589 | 0.48393 | 0.6866 | 0.87321 | 0.363393 | 0.53125 |

## Table 5 — accuracy and weighted precision / recall / F1 (%)

| Metric | CNN | RF | DT | LR | LDA | SVM | KNN | NB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 94 | 87 | 68 | 58 | 57 | 53 | 48 | 36 |
| Precision | 88 | 94 | 75 | 56 | 62 | 54 | 54 | 49 |
| Recall | 85 | 94 | 74 | 55 | 58 | 50 | 50 | 36 |
| F1-score | 86 | 94 | 74 | 55 | 54 | 50 | 50 | 35 |

## Table 6 — CNN training and testing

|  | Train | Test |
| --- | --- | --- |
| Accuracy (%) | 95.18 | 86.43 |
| Loss | 0.1483 | 0.469 |

## Table 7 — CNN multi-class classification report (test set)

| Class | Precision | Recall | F1-score | Support |
| --- | --- | --- | --- | --- |
| akiec | 0.89 | 0.68 | 0.77 | 37 |
| bcc | 0.55 | 0.88 | 0.67 | 33 |
| bkl | 0.89 | 0.94 | 0.92 | 35 |
| df | 0.88 | 0.81 | 0.84 | 43 |
| mel | 0.97 | 0.89 | 0.93 | 44 |
| nv | 0.93 | 0.86 | 0.89 | 44 |
| vasc | 0.95 | 0.89 | 0.92 | 44 |
| macro avg | 0.87 | 0.85 | 0.85 | 280 |
| weighted avg | 0.88 | 0.85 | 0.86 | 280 |

## Other reported results

- Headline claim: CNN accuracy **95.18 %** (this is the training accuracy in Table 6)
- Pre-trained InceptionV3 for comparison: 95.14 %
- Best ML model: Random Forest (0.87321 with augmentation)
