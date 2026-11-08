"""End-to-end smoke test on a synthetic HAM10000 — checks every block connects."""
from skinlesion.data.metadata import load_metadata
from skinlesion.data.organize import organize_by_class
from skinlesion.data.sampling import materialize_subset, sample_per_class
from skinlesion.evaluation.report import build_report
from skinlesion.training.cnn_trainer import run_cnn_pipeline
from skinlesion.training.ml_trainer import run_ml_pipeline
from skinlesion.utils.io import resolve


def test_full_pipeline(fake_cfg):
    cfg = fake_cfg
    meta = load_metadata(cfg.paths.raw_dir)
    assert len(meta) == 84
    organize_by_class(meta, cfg.paths.organized_dir)
    sub = sample_per_class(meta, list(cfg.data.classes), cfg.data.images_per_class, cfg.data.sampling_seed)
    materialize_subset(sub, cfg.paths.subset_dir)

    ml = run_ml_pipeline(cfg)
    assert ml["without_augmentation"]["n_train"] + ml["without_augmentation"]["n_test"] == 70
    assert ml["with_hflip_augmentation"]["n_train"] + ml["with_hflip_augmentation"]["n_test"] == 140

    cnn = run_cnn_pipeline(cfg)
    assert len(cnn["history"]["train_loss"]) == 2
    assert cnn["n_test"] == 28

    report = build_report(cfg.paths.outputs_dir).read_text(encoding="utf-8")
    assert "Table 4" in report and "Table 7" in report
    assert (resolve(cfg.paths.outputs_dir) / "cnn" / "history.png").exists()
