"""skinlesion — modular reproduction of Shetty et al. (2022),
"Skin lesion classification of dermoscopic images using machine learning
and convolutional neural network", Scientific Reports 12:18134.

Sub-packages
------------
data        download, organise, sample, augment, split HAM10000
features    global feature descriptors (colour histogram, Hu moments, Haralick)
models      classical ML model registry + the customised CNN
training    training / cross-validation loops for both pipelines
evaluation  metrics, plots, paper reference numbers and comparison report
utils       config loading, seeding, filesystem helpers
"""

__version__ = "0.1.0"
