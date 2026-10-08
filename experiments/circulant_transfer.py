"""Directional cross-module transfer demo for the circulant backend."""

from __future__ import annotations

import torch

from flowcore import CirculantKroneckerSubstrate


def main() -> None:
    torch.manual_seed(0)
    substrate = CirculantKroneckerSubstrate(
        state_dim=32,
        block_size=4,
        stability_scale=0.95,
    )

    batch, time = 1, 8
    injection = torch.zeros(batch, time, 32)
    injection[:, 0, :4] = torch.tensor([1.0, -0.5, 0.25, 0.75])

    route = torch.zeros(batch, time, substrate.num_blocks)
    route[..., 1] = 1.0
    hold = torch.ones_like(route)

    with torch.no_grad():
        par = substrate(injection, route, hold, mode="parallel")
        seq = substrate(injection, route, hold, mode="sequential")

    print(f"max parallel/sequential error: {(par-seq).abs().max().item():.3e}")
    state = par.view(batch, time, substrate.num_blocks, substrate.block_size)
    energy = state.square().sum(dim=-1).sqrt()[0]
    for t in range(time):
        values = " ".join(f"{v:.4f}" for v in energy[t].tolist())
        print(f"t={t:02d}: {values}")


if __name__ == "__main__":
    main()
