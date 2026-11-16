"""Train the DenseNet-121 domain classifier on real HAM10000 (issue 09). GPU recommended.

    python scripts/train_domain_classifier.py
    python scripts/train_domain_classifier.py --set labels.set=binary domain_classifier.epochs=10
"""
from _bootstrap import SKINGAN_CONFIG, parse_args

from skinlesion.training.domain_classifier import train_domain_classifier

if __name__ == "__main__":
    _, cfg = parse_args(__doc__, default_config=SKINGAN_CONFIG)
    train_domain_classifier(cfg)
