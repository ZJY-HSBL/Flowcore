"""Demonstrate exact cross-module transfer in the v0.3 spectral backend."""

from __future__ import annotations

import torch

from flowcore import SpectralKroneckerSubstrate


def main() -> None:
    torch.manual_seed(0)
    substrate = SpectralKroneckerSubstrate(
        state_dim=32,
        block_size=4,
        stability_scale=0.95,
        mixing_basis_seed=7,
    )

    batch, time = 1, 8
    injection = torch.zeros(batch, time, 32)
    injection[:, 0, :4] = torch.tensor([1.0, -0.5, 0.25, 0.75])

    # Unequal modal gains create physical-space cross-module mixing.
    route = torch.linspace(0.1, 0.9, substrate.num_blocks)
    route = route.view(1, 1, -1).expand(batch, time, -1)
    hold = torch.ones_like(route)

    with torch.no_grad():
        parallel = substrate(injection, route, hold, mode="parallel")
        sequential = substrate(injection, route, hold, mode="sequential")

    error = (parallel - sequential).abs().max().item()
    states = parallel.view(batch, time, substrate.num_blocks, substrate.block_size)
    energy = states.square().sum(dim=-1).sqrt()[0]

    print(f"max parallel/sequential error: {error:.3e}")
    print("module energy by time:")
    for t in range(time):
        values = " ".join(f"{v:.4f}" for v in energy[t].tolist())
        print(f"t={t:02d}: {values}")


if __name__ == "__main__":
    main()
