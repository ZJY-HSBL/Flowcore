"""Long-horizon numerical stability sweep for composable affine dynamics.

This is deliberately a diagnostics experiment, not a claim that a spectral or
row-norm bound fully characterizes learned FlowCore dynamics.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from flowcore.diagnostics import relative_state_error, state_diagnostics
from flowcore.operators import parallel_affine_scan, sequential_affine_scan


def csv_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",")]


def csv_floats(value: str) -> list[float]:
    return [float(x) for x in value.split(",")]


def make_diagonal(batch: int, time: int, dim: int, gain: float, device: torch.device):
    # Small temporal variation avoids testing only the trivial constant operator.
    jitter = 0.005 * torch.randn(batch, time, dim, device=device)
    a = torch.full((batch, time, dim), gain, device=device) + jitter
    b = 0.01 * torch.randn(batch, time, dim, device=device)
    h0 = torch.randn(batch, dim, device=device)
    return a, b, h0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", type=csv_ints, default=[128, 512, 2048, 8192])
    p.add_argument("--gains", type=csv_floats, default=[0.70, 0.90, 0.99, 1.001])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output", type=Path, default=Path("results/stability.csv"))
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    rows = []
    torch.manual_seed(0)
    for length in args.lengths:
        for gain in args.gains:
            a, b, h0 = make_diagonal(args.batch, length, args.dim, gain, device)
            with torch.inference_mode():
                seq = sequential_affine_scan(a, b, h0, kind="diagonal")
                par = parallel_affine_scan(a, b, h0, kind="diagonal")
            diag = state_diagnostics(par)
            row = {
                "length": length,
                "gain": gain,
                "relative_scan_error": relative_state_error(seq, par),
                "max_abs_state": diag.max_abs,
                "rms_state": diag.rms,
                "final_rms": diag.final_rms,
                "finite_fraction": diag.finite_fraction,
                "growth_ratio": diag.growth_ratio,
            }
            rows.append(row)
            print(
                f"T={length:5d} gain={gain:6.3f} err={row['relative_scan_error']:.3e} "
                f"max={diag.max_abs:.3e} growth={diag.growth_ratio:.3e} "
                f"finite={diag.finite_fraction:.4f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
