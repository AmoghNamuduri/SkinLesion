# GAN test — preview what SkinGAN produces

A short (~100 kimg) run of the **baseline conditional StyleGAN2-ADA** on HAM10000, using exactly the
SkinGAN pipeline (`src/skinlesion/gan/`, `scripts/train_gan.py`, `configs/skingan/baseline.yaml`).
It checks that the GPU setup works and gives a first look at generated lesions.

| File | What it is |
|---|---|
| `SkinGAN_Colab_Test.ipynb` | the Colab notebook (setup → data → train → pictures) |
| `test_config.yaml` | baseline config with a 100 kimg budget and snapshots every 20 kimg |
| `make_real_grid.py` | laptop-only preview of the real 256² training images per class (no GPU) |
| `show_results.py` | turns a run into `progression.png`, `real_vs_generated.png`, `class_grid.png` |
| `output/` | local previews (git-ignored) |

## 1 · Optional: look at the input on your laptop
```bash
python gan_test/make_real_grid.py        # -> gan_test/output/real_samples.png
```

## 2 · Run the test on Google Colab
1. **Push the repo to GitHub** (the notebook clones it). It must be public, or change `REPO_URL` in the notebook.
2. Open [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** →
   choose `gan_test/SkinGAN_Colab_Test.ipynb` (or **File → Open notebook → GitHub** and paste the repo URL).
3. **Runtime → Change runtime type → T4 GPU → Save.**
4. Run the cells top to bottom (**Runtime → Run all**, or Shift+Enter one at a time):
   - **Cell 2** asks for Google Drive access → allow it. Outputs go to `MyDrive/SkinGAN/`.
   - **Cell 4**, first time only: downloads HAM10000 (~2.7 GB) from Harvard Dataverse and builds the 256² archive (~10 min).
     It's saved to Drive, so later sessions just copy it.
   - **Cell 5** trains for about 1.5–2 h on a T4. Keep the tab open; Colab's free tier disconnects idle sessions.
     Progress grids appear in `MyDrive/SkinGAN/runs/baseline_test/<run>/fakes*.png` every 20 kimg.
   - **Cell 6** shows the pictures.
5. **If Colab disconnects:** reconnect, set the GPU runtime again, run cells 0–4, then cell 5.
   It resumes from the newest snapshot on Drive (`RESUME = True`).

## What you'll get
- `real_samples.png`: what the GAN imitates (one row per class, count in brackets)
- `progression.png`: the same samples at 0, 20, 40, … kimg; faces at the start (FFHQ transfer) turning into skin lesions
- `real_vs_generated.png`: per class, 4 real | 4 generated (the main slide)
- `class_grid.png`: 8 generated samples for each of the 7 classes

At 100 kimg, expect recognisable lesion colour and texture but not final quality. The real baseline uses
1,000 kimg (~6.5 h on a V100, roughly 16–19 h of T4 sessions).

## Troubleshooting
| Symptom | Fix |
|---|---|
| `nvidia-smi` not found / "no GPU" | Runtime → Change runtime type → GPU, then rerun from cell 1 |
| warnings about `conv2d_gradfix` / `grid_sample_gradfix` "not supported" | harmless on new PyTorch; StyleGAN falls back to standard ops |
| "Failed to build CUDA kernels; falling back to slow reference implementation" | training still works, ~30–50 % slower; check that `pip install ninja` ran |
| `CUDA out of memory` | add `gan.batch=8` to the `--set` list in cell 5 |
| download in cell 4 fails | upload `ham10000_256_7class.zip` (made locally with `python scripts/export_gan_dataset.py`) to `MyDrive/SkinGAN/data/` and rerun cell 4 |
| numpy / pickle errors when loading the FFHQ model | `!pip install -q "numpy<2"`, then Runtime → Restart session, and rerun from cell 2 |
