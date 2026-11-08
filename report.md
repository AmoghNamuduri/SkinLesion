# Reproduction report — Shetty et al. (2022), Sci Rep 12:18134

## Table 4 — ML accuracy (mean 10-fold CV on the training split)

| Setting | Source | LR | LDA | KNN | DT | RF | NB | SVM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100×7, no aug | paper | 0.4893 | 0.3518 | 0.4179 | 0.4429 | 0.5857 | 0.3411 | 0.4214 |
| 100×7, no aug | **ours** | 0.4161 | 0.3518 | 0.3393 | 0.3839 | 0.5411 | 0.3464 | 0.4000 |
| 200×7, h-flip | paper | 0.5813 | 0.5759 | 0.4839 | 0.6866 | 0.8732 | 0.3634 | 0.5312 |
| 200×7, h-flip | **ours** | 0.5214 | 0.6714 | 0.4313 | 0.8187 | 0.8670 | 0.4339 | 0.5339 |

## Table 5 — accuracy and weighted precision / recall / F1 (%)

| Model | Acc paper | Acc ours | P paper | P ours | R paper | R ours | F1 paper | F1 ours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN | 94 | 92.32 | 88 | 82.10 | 85 | 80.71 | 86 | 80.70 |
| RF | 87 | 86.70 | 94 | 92.52 | 94 | 92.14 | 94 | 92.00 |
| DT | 68 | 81.88 | 75 | 89.43 | 74 | 89.29 | 74 | 89.15 |
| LR | 58 | 52.14 | 56 | 56.97 | 55 | 56.43 | 55 | 56.43 |
| LDA | 57 | 67.14 | 62 | 63.88 | 58 | 62.50 | 54 | 62.42 |
| SVM | 53 | 53.39 | 54 | 59.30 | 50 | 57.50 | 50 | 57.66 |
| KNN | 48 | 43.12 | 54 | 51.88 | 50 | 47.14 | 50 | 48.01 |
| NB | 36 | 43.39 | 49 | 57.40 | 36 | 42.14 | 35 | 39.96 |

_Accuracy column: CV mean for ML models, final-epoch train accuracy for the CNN (the paper's 94/95.18 % is a training accuracy). P/R/F1 are on the held-out test split._

## Table 6 — CNN train vs test

|  | Train paper | Train ours | Test paper | Test ours |
| --- | --- | --- | --- | --- |
| Accuracy (%) | 95.18 | 92.32 | 86.43 | 80.71 |
| Loss | 0.1483 | 0.2102 | 0.469 | 0.8041 |

## Table 7 — CNN per-class report (test split)

| Class | P paper | P ours | R paper | R ours | F1 paper | F1 ours | Support paper | Support ours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| akiec | 0.89 | 0.74 | 0.68 | 0.85 | 0.77 | 0.80 | 37 | 41 |
| bcc | 0.55 | 0.68 | 0.88 | 0.77 | 0.67 | 0.72 | 33 | 39 |
| bkl | 0.89 | 0.94 | 0.94 | 0.71 | 0.92 | 0.81 | 35 | 45 |
| df | 0.88 | 0.93 | 0.81 | 0.68 | 0.84 | 0.78 | 43 | 40 |
| mel | 0.97 | 0.80 | 0.89 | 0.88 | 0.93 | 0.83 | 44 | 40 |
| nv | 0.93 | 0.75 | 0.86 | 0.83 | 0.89 | 0.79 | 44 | 36 |
| vasc | 0.95 | 0.88 | 0.89 | 0.95 | 0.92 | 0.91 | 44 | 39 |
| macro avg | 0.87 | 0.82 | 0.85 | 0.81 | 0.85 | 0.81 | 280 | 280 |
| weighted avg | 0.88 | 0.82 | 0.85 | 0.81 | 0.86 | 0.81 | 280 | 280 |

Figures: `outputs/cnn/history.png` (Fig. 5), `outputs/cnn/confusion_matrix.png`.
