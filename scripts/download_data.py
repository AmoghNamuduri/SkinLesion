"""Step 0 — download and extract HAM10000.

    python scripts/download_data.py                        # Harvard Dataverse (no login)
    python scripts/download_data.py --set download.source=kaggle
"""
from _bootstrap import parse_args

from skinlesion.data.download import download

if __name__ == "__main__":
    _, cfg = parse_args(__doc__)
    download(cfg)
