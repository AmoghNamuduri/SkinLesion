"""Rebuild outputs/report.md (paper tables vs our results) from saved results.json files."""
from _bootstrap import parse_args

from skinlesion.evaluation.report import build_report

if __name__ == "__main__":
    _, cfg = parse_args(__doc__)
    path = build_report(cfg.paths.outputs_dir)
    print(path.read_text(encoding="utf-8"))
