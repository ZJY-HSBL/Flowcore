"""FlowCore v0.6: arbitrary simultaneous routing stress test.

Unlike v0.5, all modules contain independent payloads at once.  A single route
operator must apply a full permutation to the module axis.  This directly tests
whether a routing family can represent non-translation-structured communication.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from flowcore import (
    CirculantKroneckerSubstrate,
    ParallelFlowSubstrate,
    SpectralKroneckerSubstrate,
    arbitrary_permutation_bank,
    block_projection_error,
    circulant_projection_error,
    cyclic_permutation_bank,
    make_permutation_batch,
    make_permutation_controls,
    orthogonal_mixing_basis,
    permutation_loss,
    permutation_metrics,
    spectral_projection_error,
)


BACKENDS = ("block", "spectral_kronecker", "circulant_kronecker")
FAMILIES = ("cyclic", "arbitrary")


class TaskRoutePolicy(nn.Module):
    """Same controller bandwidth for every backend: task id -> M route values."""

    def __init__(self, num_tasks: int, num_modules: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_tasks, num_modules))

    def forward(self, task_id: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits[task_id])


@dataclass(slots=True)
class StressResult:
    family: str
    backend: str
    seed: int
    mse: float
    relative_mse: float
    slot_accuracy: float
    parallel_error: float
    analytic_projection_error_mean: float
    analytic_projection_error_max: float


def parse_int_csv(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_substrate(backend: str, modules: int, local_dim: int, device: torch.device):
    kwargs = dict(
        state_dim=modules * local_dim,
        block_size=local_dim,
        stability_scale=1.0,
    )
    if backend == "block":
        module = ParallelFlowSubstrate(**kwargs)
    elif backend == "spectral_kronecker":
        module = SpectralKroneckerSubstrate(
            **kwargs,
            mixing_basis_seed=17,
        )
    elif backend == "circulant_kronecker":
        module = CirculantKroneckerSubstrate(**kwargs)
    else:
        raise ValueError(backend)
    return module.to(device)


@torch.no_grad()
def calibrate_identity_like(substrate) -> None:
    if isinstance(substrate, ParallelFlowSubstrate):
        eye = torch.eye(
            substrate.block_size,
            device=substrate.raw_blocks.device,
            dtype=substrate.raw_blocks.dtype,
        )
        substrate.raw_blocks.copy_(eye.unsqueeze(0).expand_as(substrate.raw_blocks))
        substrate.raw_strength.fill_(8.0)
    elif isinstance(substrate, SpectralKroneckerSubstrate):
        substrate.raw_local.copy_(
            torch.eye(
                substrate.block_size,
                device=substrate.raw_local.device,
                dtype=substrate.raw_local.dtype,
            )
        )
        substrate.raw_local_strength.fill_(8.0)
        substrate.raw_mode_strength.fill_(8.0)
    elif isinstance(substrate, CirculantKroneckerSubstrate):
        substrate.raw_local.copy_(
            torch.eye(
                substrate.block_size,
                device=substrate.raw_local.device,
                dtype=substrate.raw_local.dtype,
            )
        )
        substrate.raw_local_strength.fill_(8.0)
        substrate.raw_kernel_prior.zero_()


def projection_errors(
    backend: str,
    permutations: torch.Tensor,
    *,
    basis: torch.Tensor,
) -> list[float]:
    values = []
    for permutation in permutations:
        if backend == "block":
            value = block_projection_error(permutation)
        elif backend == "spectral_kronecker":
            value = spectral_projection_error(
                permutation,
                basis,
                positive_unit_gain=True,
            )
        elif backend == "circulant_kronecker":
            value = circulant_projection_error(permutation)
        else:
            raise ValueError(backend)
        values.append(value)
    return values


def make_bank(family: str, tasks: int, modules: int, permutation_seed: int):
    if family == "cyclic":
        return cyclic_permutation_bank(tasks, modules)
    if family == "arbitrary":
        return arbitrary_permutation_bank(
            tasks, modules, seed=permutation_seed
        )
    raise ValueError(family)


def run_one(
    family: str,
    backend: str,
    seed: int,
    permutations: torch.Tensor,
    args,
    device: torch.device,
) -> StressResult:
    torch.manual_seed(seed)
    substrate = build_substrate(
        backend, args.modules, args.local_dim, device
    )
    calibrate_identity_like(substrate)
    for parameter in substrate.parameters():
        parameter.requires_grad_(False)

    policy = TaskRoutePolicy(
        permutations.shape[0], args.modules
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    train_generator = torch.Generator(device=device)
    train_generator.manual_seed(seed + 3000)

    for _ in range(args.steps):
        batch = make_permutation_batch(
            args.batch,
            args.train_time,
            permutations,
            args.local_dim,
            device=device,
            generator=train_generator,
        )
        route_at_transfer = policy(batch.task_id)
        route, hold = make_permutation_controls(
            route_at_transfer, args.train_time
        )
        states = substrate(
            batch.injection, route, hold, mode="parallel"
        )
        loss = permutation_loss(
            states,
            batch.target,
            args.modules,
            args.local_dim,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy.parameters(), args.grad_clip
        )
        optimizer.step()

    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(seed + 4000)
    batch = make_permutation_batch(
        args.eval_batch,
        args.long_time,
        permutations,
        args.local_dim,
        device=device,
        generator=eval_generator,
    )
    route_at_transfer = policy(batch.task_id)
    route, hold = make_permutation_controls(
        route_at_transfer, args.long_time
    )
    with torch.inference_mode():
        parallel = substrate(
            batch.injection, route, hold, mode="parallel"
        )
        sequential = substrate(
            batch.injection, route, hold, mode="sequential"
        )
    metrics = permutation_metrics(
        parallel,
        batch.payload,
        batch.target,
        batch.task_id,
        permutations,
        parallel_reference=sequential,
    )

    basis = orthogonal_mixing_basis(
        args.modules, seed=17
    )
    errors = projection_errors(
        backend, permutations, basis=basis
    )
    return StressResult(
        family=family,
        backend=backend,
        seed=seed,
        mse=metrics.mse,
        relative_mse=metrics.relative_mse,
        slot_accuracy=metrics.slot_accuracy,
        parallel_error=metrics.max_parallel_error or 0.0,
        analytic_projection_error_mean=statistics.fmean(errors),
        analytic_projection_error_max=max(errors),
    )


def summarize(results: list[StressResult]):
    summary = {}
    for family in FAMILIES:
        summary[family] = {}
        for backend in BACKENDS:
            rows = [
                row
                for row in results
                if row.family == family and row.backend == backend
            ]
            if not rows:
                continue
            summary[family][backend] = {
                "mse_mean": statistics.fmean(r.mse for r in rows),
                "mse_std": statistics.stdev([r.mse for r in rows]) if len(rows) > 1 else 0.0,
                "relative_mse_mean": statistics.fmean(r.relative_mse for r in rows),
                "slot_accuracy_mean": statistics.fmean(r.slot_accuracy for r in rows),
                "parallel_error_max": max(r.parallel_error for r in rows),
                "analytic_projection_error_mean": statistics.fmean(
                    r.analytic_projection_error_mean for r in rows
                ),
            }
    return summary


def print_summary(summary) -> None:
    print(
        "\nfamily     backend                  mse       rel_mse   slot_acc   analytic_err"
    )
    for family in FAMILIES:
        for backend in BACKENDS:
            if backend not in summary.get(family, {}):
                continue
            row = summary[family][backend]
            print(
                f"{family:10s} {backend:24s} "
                f"{row['mse_mean']:9.5f} "
                f"{row['relative_mse_mean']:9.5f} "
                f"{row['slot_accuracy_mean']:9.4f} "
                f"{row['analytic_projection_error_mean']:12.5f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=parse_int_csv, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--modules", type=int, default=6)
    parser.add_argument("--local-dim", type=int, default=3)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--train-time", type=int, default=4)
    parser.add_argument("--long-time", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--permutation-seed", type=int, default=123)
    parser.add_argument(
        "--families",
        default="cyclic,arbitrary",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v06_arbitrary_routing.json"),
    )
    args = parser.parse_args()

    families = tuple(
        value.strip() for value in args.families.split(",") if value.strip()
    )
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise SystemExit(f"unknown families: {sorted(unknown)}")
    if "cyclic" in families and args.tasks > args.modules - 1:
        raise SystemExit("cyclic tasks must be <= modules-1")

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    results: list[StressResult] = []
    for family in families:
        permutations = make_bank(
            family,
            args.tasks,
            args.modules,
            args.permutation_seed,
        )
        print(f"\n{family} permutation bank:")
        print(permutations)
        for backend in BACKENDS:
            for seed in args.seeds:
                result = run_one(
                    family,
                    backend,
                    seed,
                    permutations,
                    args,
                    device,
                )
                results.append(result)
                print(
                    f"{family:10s} {backend:24s} seed={seed} "
                    f"mse={result.mse:.5f} "
                    f"slot={result.slot_accuracy:.4f} "
                    f"analytic={result.analytic_projection_error_mean:.4f}"
                )

    summary = summarize(results)
    print_summary(summary)
    payload = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "results": [asdict(row) for row in results],
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
