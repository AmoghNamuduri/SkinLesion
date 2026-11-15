# third_party/

`stylegan2-ada-pytorch/` is NVIDIA's official [StyleGAN2-ADA (PyTorch)](https://github.com/NVlabs/stylegan2-ada-pytorch),
cloned on demand rather than committed:

```bash
python third_party/setup_stylegan.py          # clone + apply SkinGAN patches
python third_party/setup_stylegan.py --check  # show patch status
```

It is released under the NVIDIA Source Code License (non-commercial research use); see its `LICENSE.txt`.

Our only changes, applied by `setup_stylegan.py` and marked `# [SkinGAN patch]`:

| File | Change | Why |
|---|---|---|
| `training/loss.py` | adds `λ · CE(classifier(G(z,c)), c)` to the generator loss when `$SKINGAN_DOMAIN_LOSS` is set | the SkinGAN domain loss (roadmap issue 10); the baseline runs unchanged |
| `torch_utils/misc.py` | skip parameters whose shape differs when loading a checkpoint | lets a conditional model start from NVIDIA's unconditional FFHQ-256 weights (`--resume=ffhq256`) |

The rest of SkinGAN lives in `src/skinlesion/gan/` and is launched with `scripts/train_gan.py`.
