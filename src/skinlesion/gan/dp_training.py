"""Differentially-private training utilities (roadmap issue 16, stretch).

SpineGAN's "DP" variant adds Gaussian noise to the gradients and *then* clips the
gradient norm, on the whole batch. DP-SGD (Abadi et al. 2016) requires the opposite:
clip each example's gradient to norm C, sum, then add N(0, σ²C²) noise once. With
SpineGAN's batch size of 1, per-example clipping is just clipping the step's gradient,
so the corrected step is ``clip → noise``, implemented in ``dp_step_``.

Only the discriminator touches real images, so only D needs the private step.
Hooking ``dp_step_`` into StyleGAN's training loop is left to issue 16.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def dp_step_(params, max_norm: float, noise_multiplier: float, batch_size: int = 1,
             generator: torch.Generator | None = None) -> float:
    """In place: clip the total gradient norm to ``max_norm``, then add N(0, (σ·C)²) noise and
    average over ``batch_size``. Correct DP-SGD when each call sees one example's gradient
    (batch size 1, as in SpineGAN). Returns the pre-clipping norm."""
    params = [p for p in params if p.grad is not None]
    total = torch.norm(torch.stack([p.grad.norm(2) for p in params]), 2).item() if params else 0.0
    scale = min(1.0, max_norm / (total + 1e-12))
    for p in params:
        p.grad.mul_(scale)
        noise = torch.randn(p.grad.shape, generator=generator, device="cpu").to(p.grad.device)
        p.grad.add_(noise * noise_multiplier * max_norm).div_(batch_size)
    return total


def epsilon(n: int, batch_size: int, noise_multiplier: float, steps: int, delta: float | None = None) -> float:
    """ε for DP-SGD via Opacus' RDP accountant (``pip install opacus``)."""
    from opacus.accountants import RDPAccountant

    acc = RDPAccountant()
    acc.history = [(noise_multiplier, batch_size / n, steps)]
    return acc.get_epsilon(delta=delta or 1.0 / n)


def steps_for_kimg(kimg: float, batch_size: int) -> int:
    """Number of discriminator steps for a StyleGAN budget in kimg."""
    return math.ceil(kimg * 1000 / batch_size)
