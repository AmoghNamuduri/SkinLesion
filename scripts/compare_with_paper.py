"""Rebuild outputs/report.md (paper tables vs our results) from saved results.json files,
and outputs/report-skingan.md if any SkinGAN results exist."""
from _bootstrap import parse_args

from skinlesion.evaluation.report import build_report, build_skingan_report

if __name__ == "__main__":
    _, cfg = parse_args(__doc__)
    path = build_report(cfg.paths.outputs_dir)
    print(path.read_text(encoding="utf-8"))
    skingan = build_skingan_report(cfg.paths.outputs_dir)
    if skingan:
        print("\nSkinGAN report ->", skingan)
