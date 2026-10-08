"""FlowCore v0.5: matched cross-module substrate comparison.

This benchmark isolates the dynamic substrate.  A payload enters only physical
module 0.  A task-specific route policy receives the requested destination and
must make the payload physically arrive there at t=1.  The state is then held.

Default mode freezes the substrate after identity-like calibration and trains the
same-size route table for every backend.  This compares routing expressiveness
without allowing one substrate to win by learning more hidden parameters.

Examples
--------
Quick CPU smoke:

    python experiments/compare_substrates.py --steps 30 --seeds 0 --batch 32

Longer run:

    python experiments/compare_substrates.py --steps 400 --seeds 0,1,2,3,4 \
        --batch 256 --long-time 64 --runtime-repeats 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from flowcore import (
    CirculantKroneckerSubstrate,
    ParallelFlowSubstrate,
    SpectralKroneckerSubstrate,
)
from flowcore.evaluation import (
    make_transport_batch,
    make_transport_controls,
    parameter_count,
    transport_loss,
    transport_metrics,
)


BACKENDS = ("block", "spectral_kronecker", "circulant_kronecker")


class DestinationRoutePolicy(nn.Module):
    """Same parameterization for every substrate: destination -> M route values."""

    def __init__(self, num_modules: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_modules, num_modules))

    def forward(self, destination: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits[destination])


@dataclass(slots=True)
class RunResult:
    backend: str
    seed: int
    trainable_substrate: bool
    substrate_params: int
    route_policy_params: int
    train_target_mse: float
    long_target_mse: float
    long_leakage_mse: float
    long_delivery_fraction: float
    long_source_fraction: float
    parallel_error: float
    parallel_ms: float
    sequential_ms: float
    sequential_over_parallel: float
    final_loss: float


def parse_int_csv(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_substrate(
    backend: str,
    num_modules: int,
    local_dim: int,
    device: torch.device,
):
    state_dim = num_modules * local_dim
    kwargs = dict(
        state_dim=state_dim,
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
        raise ValueError(f"unknown backend: {backend}")
    return module.to(device)


@torch.no_grad()
def calibrate_identity_like(substrate) -> None:
    """Remove avoidable local-dynamics confounds before the routing comparison."""

    if isinstance(substrate, ParallelFlowSubstrate):
        substrate.raw_blocks.zero_()
        eye = torch.eye(
            substrate.block_size,
            device=substrate.raw_blocks.device,
            dtype=substrate.raw_blocks.dtype,
        )
        substrate.raw_blocks.copy_(
            eye.unsqueeze(0).expand_as(substrate.raw_blocks)
        )
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
    else:
        raise TypeError(type(substrate))


def build_controls(
    policy: DestinationRoutePolicy,
    destination: torch.Tensor,
    time_steps: int,
    num_modules: int,
):
    route_transfer = policy(destination)
    return make_transport_controls(
        destination,
        route_transfer,
        time_steps,
        num_modules,
    )


def timed_forward(
    substrate,
    injection,
    route,
    hold,
    mode: str,
    repeats: int,
    device: torch.device,
) -> float:
    with torch.inference_mode():
        for _ in range(2):
            substrate(injection, route, hold, mode=mode)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            substrate(injection, route, hold, mode=mode)
        synchronize(device)
    return 1000.0 * (time.perf_counter() - start) / repeats


def run_one(
    backend: str,
    seed: int,
    args,
    device: torch.device,
) -> RunResult:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    substrate = build_substrate(
        backend, args.modules, args.local_dim, device
    )
    calibrate_identity_like(substrate)
    for parameter in substrate.parameters():
        parameter.requires_grad_(args.train_substrate)

    policy = DestinationRoutePolicy(args.modules).to(device)
    optimized = list(policy.parameters())
    if args.train_substrate:
        optimized += [
            p for p in substrate.parameters() if p.requires_grad
        ]
    optimizer = torch.optim.Adam(optimized, lr=args.lr)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1000)

    final_loss = float("nan")
    for step in range(args.steps):
        batch = make_transport_batch(
            args.batch,
            args.train_time,
            args.modules,
            args.local_dim,
            device=device,
            generator=generator,
        )
        route, hold = build_controls(
            policy,
            batch.destination,
            args.train_time,
            args.modules,
        )
        states = substrate(
            batch.injection, route, hold, mode="parallel"
        )
        loss = transport_loss(
            states,
            batch.destination,
            batch.payload,
            args.modules,
            args.local_dim,
            leakage_weight=args.leakage_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimized, args.grad_clip)
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    eval_gen = torch.Generator(device=device)
    eval_gen.manual_seed(seed + 2000)
    train_batch = make_transport_batch(
        args.eval_batch,
        args.train_time,
        args.modules,
        args.local_dim,
        device=device,
        generator=eval_gen,
    )
    route, hold = build_controls(
        policy,
        train_batch.destination,
        args.train_time,
        args.modules,
    )
    with torch.inference_mode():
        train_states = substrate(
            train_batch.injection, route, hold, mode="parallel"
        )
    train_metrics = transport_metrics(
        train_states,
        train_batch.destination,
        train_batch.payload,
        args.modules,
        args.local_dim,
    )

    long_batch = make_transport_batch(
        args.eval_batch,
        args.long_time,
        args.modules,
        args.local_dim,
        device=device,
        generator=eval_gen,
    )
    long_route, long_hold = build_controls(
        policy,
        long_batch.destination,
        args.long_time,
        args.modules,
    )
    with torch.inference_mode():
        parallel = substrate(
            long_batch.injection,
            long_route,
            long_hold,
            mode="parallel",
        )
        sequential = substrate(
            long_batch.injection,
            long_route,
            long_hold,
            mode="sequential",
        )
    long_metrics = transport_metrics(
        parallel,
        long_batch.destination,
        long_batch.payload,
        args.modules,
        args.local_dim,
        parallel_reference=sequential,
    )

    runtime_batch = make_transport_batch(
        args.runtime_batch,
        args.long_time,
        args.modules,
        args.local_dim,
        device=device,
        generator=eval_gen,
    )
    runtime_route, runtime_hold = build_controls(
        policy,
        runtime_batch.destination,
        args.long_time,
        args.modules,
    )
    parallel_ms = timed_forward(
        substrate,
        runtime_batch.injection,
        runtime_route,
        runtime_hold,
        "parallel",
        args.runtime_repeats,
        device,
    )
    sequential_ms = timed_forward(
        substrate,
        runtime_batch.injection,
        runtime_route,
        runtime_hold,
        "sequential",
        args.runtime_repeats,
        device,
    )

    return RunResult(
        backend=backend,
        seed=seed,
        trainable_substrate=args.train_substrate,
        substrate_params=parameter_count(substrate),
        route_policy_params=parameter_count(policy),
        train_target_mse=train_metrics.target_mse,
        long_target_mse=long_metrics.target_mse,
        long_leakage_mse=long_metrics.leakage_mse,
        long_delivery_fraction=long_metrics.delivery_fraction,
        long_source_fraction=long_metrics.source_fraction,
        parallel_error=long_metrics.max_parallel_error or 0.0,
        parallel_ms=parallel_ms,
        sequential_ms=sequential_ms,
        sequential_over_parallel=sequential_ms / max(parallel_ms, 1e-12),
        final_loss=final_loss,
    )


def summarize(results: list[RunResult]) -> dict[str, dict[str, float]]:
    metrics = (
        "train_target_mse",
        "long_target_mse",
        "long_leakage_mse",
        "long_delivery_fraction",
        "parallel_error",
        "parallel_ms",
        "sequential_ms",
        "sequential_over_parallel",
    )
    summary: dict[str, dict[str, float]] = {}
    for backend in BACKENDS:
        rows = [r for r in results if r.backend == backend]
        if not rows:
            continue
        values: dict[str, float] = {}
        for name in metrics:
            xs = [float(getattr(row, name)) for row in rows]
            values[f"{name}_mean"] = statistics.fmean(xs)
            values[f"{name}_std"] = (
                statistics.stdev(xs) if len(xs) > 1 else 0.0
            )
        summary[backend] = values
    return summary


def print_table(summary: dict[str, dict[str, float]]) -> None:
    print(
        "\nbackend                 long_mse    delivery    leakage     "
        "par_err      par_ms    seq/par"
    )
    for backend in BACKENDS:
        if backend not in summary:
            continue
        row = summary[backend]
        print(
            f"{backend:24s} "
            f"{row['long_target_mse_mean']:10.5f} "
            f"{row['long_delivery_fraction_mean']:10.4f} "
            f"{row['long_leakage_mse_mean']:10.5f} "
            f"{row['parallel_error_mean']:10.2e} "
            f"{row['parallel_ms_mean']:9.3f} "
            f"{row['sequential_over_parallel_mean']:8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends",
        default=",".join(BACKENDS),
        help="comma-separated subset of block,spectral_kronecker,circulant_kronecker",
    )
    parser.add_argument("--seeds", type=parse_int_csv, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--runtime-batch", type=int, default=64)
    parser.add_argument("--modules", type=int, default=8)
    parser.add_argument("--local-dim", type=int, default=2)
    parser.add_argument("--train-time", type=int, default=4)
    parser.add_argument("--long-time", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--leakage-weight", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--runtime-repeats", type=int, default=10)
    parser.add_argument("--train-substrate", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v05_substrate_compare.json"),
    )
    args = parser.parse_args()

    requested = tuple(x.strip() for x in args.backends.split(",") if x.strip())
    unknown = set(requested) - set(BACKENDS)
    if unknown:
        raise SystemExit(f"unknown backends: {sorted(unknown)}")

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    print(
        f"device={device} modules={args.modules} local_dim={args.local_dim} "
        f"steps={args.steps} frozen_substrate={not args.train_substrate}"
    )
    results: list[RunResult] = []
    for backend in requested:
        for seed in args.seeds:
            result = run_one(backend, seed, args, device)
            results.append(result)
            print(
                f"{backend:24s} seed={seed} "
                f"long_mse={result.long_target_mse:.5f} "
                f"delivery={result.long_delivery_fraction:.4f} "
                f"err={result.parallel_error:.2e}"
            )

    summary = summarize(results)
    print_table(summary)

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
