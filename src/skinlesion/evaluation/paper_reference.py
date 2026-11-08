"""Numbers published in Shetty et al. (2022), copied verbatim from the paper's tables.

Notes on what each number measures (worked out from the paper's text):
* Table 4: mean 10-fold cross-validation accuracy on the 80 % training split
  (e.g. 0.48929 × 560 = 274 and 0.58125 × 1120 = 651 exactly).
* Table 5: accuracy plus weighted precision/recall/F1 (%), for the augmented 200×7 setting.
* Table 6: final-epoch CNN train metrics and held-out test metrics. The headline
  "95.18 %" is the TRAIN accuracy; the test accuracy is 86.43 %.
* Table 7: per-class report of the CNN on the 280-image test set.
"""

ML_ORDER = ["LR", "LDA", "KNN", "DT", "RF", "NB", "SVM"]

TABLE4_ML_CV_ACCURACY = {
    "without_augmentation": {"LR": 0.48929, "LDA": 0.35179, "KNN": 0.41786, "DT": 0.44286,
                             "RF": 0.58571, "NB": 0.34107, "SVM": 0.42143},
    "with_hflip_augmentation": {"LR": 0.58125, "LDA": 0.57589, "KNN": 0.48393, "DT": 0.68660,
                                "RF": 0.87321, "NB": 0.363393, "SVM": 0.53125},
}

# percentages
TABLE5_SUMMARY = {
    "CNN": {"accuracy": 94, "precision": 88, "recall": 85, "f1": 86},
    "RF":  {"accuracy": 87, "precision": 94, "recall": 94, "f1": 94},
    "DT":  {"accuracy": 68, "precision": 75, "recall": 74, "f1": 74},
    "LR":  {"accuracy": 58, "precision": 56, "recall": 55, "f1": 55},
    "LDA": {"accuracy": 57, "precision": 62, "recall": 58, "f1": 54},
    "SVM": {"accuracy": 53, "precision": 54, "recall": 50, "f1": 50},
    "KNN": {"accuracy": 48, "precision": 54, "recall": 50, "f1": 50},
    "NB":  {"accuracy": 36, "precision": 49, "recall": 36, "f1": 35},
}

TABLE6_CNN = {"train": {"accuracy": 95.18, "loss": 0.1483}, "test": {"accuracy": 86.43, "loss": 0.4690}}

TABLE7_CNN_REPORT = {
    "akiec": {"precision": 0.89, "recall": 0.68, "f1": 0.77, "support": 37},
    "bcc":   {"precision": 0.55, "recall": 0.88, "f1": 0.67, "support": 33},
    "bkl":   {"precision": 0.89, "recall": 0.94, "f1": 0.92, "support": 35},
    "df":    {"precision": 0.88, "recall": 0.81, "f1": 0.84, "support": 43},
    "mel":   {"precision": 0.97, "recall": 0.89, "f1": 0.93, "support": 44},
    "nv":    {"precision": 0.93, "recall": 0.86, "f1": 0.89, "support": 44},
    "vasc":  {"precision": 0.95, "recall": 0.89, "f1": 0.92, "support": 44},
    "macro avg":    {"precision": 0.87, "recall": 0.85, "f1": 0.85, "support": 280},
    "weighted avg": {"precision": 0.88, "recall": 0.85, "f1": 0.86, "support": 280},
}

INCEPTION_V3_ACCURACY = 95.14
