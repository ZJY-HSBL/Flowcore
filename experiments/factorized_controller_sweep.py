"""FlowCore v0.9: factorized discrete-controller bandwidth sweep.

The permutation substrate is unchanged from v0.7/v0.8.  This experiment scans
controller factorization rank to find the minimum parameter budget that still
recovers the arbitrary routing tasks exactly.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from flowcore import (
    FactorizedTaskPermutationController,
    arbitrary_permutation_bank,
    geometric_temperature,
    make_permutation_batch,
    parallel_monomial_scan,
    permutation_metrics,
    sequential_monomial_scan,
)


@dataclass(slots=True)
class RankResult:
    rank: int
    seed: int
    soft_mse: float
    hard_mse: float
    slot_accuracy: float
    entry_accuracy: float
    task_exact_accuracy: float
    parallel_error: float
    controller_params: int
    full_controller_params: int
    route_vector_baseline_params: int


def parse_int_csv(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def train_rank(rank: int, seed: int, args, device: torch.device) -> RankResult:
    torch.manual_seed(seed)
    permutations = arbitrary_permutation_bank(
        args.tasks, args.modules, seed=args.permutation_seed
    ).to(device)

    controller = FactorizedTaskPermutationController(
        args.tasks,
        args.modules,
        rank,
        sinkhorn_iterations=args.sinkhorn_iterations,
    ).to(device)
    optimizer = torch.optim.Adam(
        controller.parameters(), lr=args.lr
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 7000 + rank * 100)

    for step in range(args.steps):
        batch = make_permutation_batch(
            args.batch,
            2,
            permutations,
            args.local_dim,
            device=device,
            generator=generator,
        )
        temperature = geometric_temperature(
            step,
            args.steps,
            start=args.temperature_start,
            end=args.temperature_end,
        )
        soft = controller.soft_matrix(
            batch.task_id,
            temperature=temperature,
        )
        routed = torch.matmul(soft, batch.payload)
        mse = (routed - batch.target).square().mean()
        entropy = -(
            soft.clamp_min(1e-9) * soft.clamp_min(1e-9).log()
        ).sum(dim=-1).mean()
        loss = mse + args.entropy_weight * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            controller.parameters(), args.grad_clip
        )
        optimizer.step()

    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(seed + 8000 + rank * 100)
    batch = make_permutation_batch(
        args.eval_batch,
        args.time,
        permutations,
        args.local_dim,
        device=device,
        generator=eval_generator,
    )

    with torch.inference_mode():
        soft = controller.soft_matrix(
            batch.task_id,
            temperature=args.temperature_end,
        )
        soft_routed = torch.matmul(soft, batch.payload)
        soft_mse = float(
            (soft_routed - batch.target).square().mean().cpu()
        )

        task_ids = torch.arange(args.tasks, device=device)
        predicted_tasks = controller.hard_permutation(
            task_ids,
            temperature=args.temperature_end,
        )
        entry_accuracy = float(
            (predicted_tasks == permutations).float().mean().cpu()
        )
        task_exact_accuracy = float(
            (predicted_tasks == permutations)
            .all(dim=-1)
            .float()
            .mean()
            .cpu()
        )

        predicted = predicted_tasks[batch.task_id]
        identity = torch.arange(
            args.modules, device=device, dtype=torch.long
        ).view(1, 1, args.modules)
        operator_permutation = identity.expand(
            args.eval_batch,
            args.time,
            args.modules,
        ).clone()
        operator_permutation[:, 1] = predicted

        gain = torch.ones(
            args.eval_batch,
            args.time,
            args.modules,
            device=device,
        )
        local = torch.eye(
            args.local_dim, device=device
        ).view(1, 1, args.local_dim, args.local_dim)
        local = local.expand(
            args.eval_batch,
            args.time,
            args.local_dim,
            args.local_dim,
        )
        bias = batch.injection.view(
            args.eval_batch,
            args.time,
            args.modules,
            args.local_dim,
        )
        h0 = torch.zeros(
            args.eval_batch,
            args.modules,
            args.local_dim,
            device=device,
        )
        parallel = parallel_monomial_scan(
            operator_permutation, gain, local, bias, h0
        )
        sequential = sequential_monomial_scan(
            operator_permutation, gain, local, bias, h0
        )
        flat_parallel = parallel.reshape(
            args.eval_batch,
            args.time,
            args.modules * args.local_dim,
        )
        flat_sequential = sequential.reshape_as(flat_parallel)
        metrics = permutation_metrics(
            flat_parallel,
            batch.payload,
            batch.target,
            batch.task_id,
            permutations,
            parallel_reference=flat_sequential,
        )

    params = sum(p.numel() for p in controller.parameters())
    return RankResult(
        rank=rank,
        seed=seed,
        soft_mse=soft_mse,
        hard_mse=metrics.mse,
        slot_accuracy=metrics.slot_accuracy,
        entry_accuracy=entry_accuracy,
        task_exact_accuracy=task_exact_accuracy,
        parallel_error=metrics.max_parallel_error or 0.0,
        controller_params=params,
        full_controller_params=args.tasks * args.modules * args.modules,
        route_vector_baseline_params=args.tasks * args.modules,
    )


def summarize(results: list[RankResult]):
    summary = {}
    ranks = sorted({r.rank for r in results})
    for rank in ranks:
        rows = [r for r in results if r.rank == rank]
        summary[str(rank)] = {
            "controller_params": rows[0].controller_params,
            "compression_vs_full": (
                rows[0].controller_params / rows[0].full_controller_params
            ),
            "soft_mse_mean": statistics.fmean(r.soft_mse for r in rows),
            "hard_mse_mean": statistics.fmean(r.hard_mse for r in rows),
            "slot_accuracy_mean": statistics.fmean(
                r.slot_accuracy for r in rows
            ),
            "entry_accuracy_mean": statistics.fmean(
                r.entry_accuracy for r in rows
            ),
            "task_exact_accuracy_mean": statistics.fmean(
                r.task_exact_accuracy for r in rows
            ),
            "parallel_error_max": max(
                r.parallel_error for r in rows
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=parse_int_csv, default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=parse_int_csv, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--modules", type=int, default=6)
    parser.add_argument("--local-dim", type=int, default=3)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--time", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iterations", type=int, default=20)
    parser.add_argument("--permutation-seed", type=int, default=123)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v09_factorized_rank_sweep.json"),
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    results = []
    for rank in args.ranks:
        for seed in args.seeds:
            result = train_rank(rank, seed, args, device)
            results.append(result)
            print(
                f"rank={rank} seed={seed} "
                f"params={result.controller_params} "
                f"hard_mse={result.hard_mse:.6f} "
                f"slot={result.slot_accuracy:.4f} "
                f"entry={result.entry_accuracy:.4f} "
                f"task_exact={result.task_exact_accuracy:.4f}"
            )

    summary = summarize(results)
    print("\nrank summary")
    for rank in args.ranks:
        row = summary[str(rank)]
        print(
            f"rank={rank} params={row['controller_params']} "
            f"full_ratio={row['compression_vs_full']:.3f} "
            f"hard_mse={row['hard_mse_mean']:.6f} "
            f"task_exact={row['task_exact_accuracy_mean']:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "config": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "results": [asdict(r) for r in results],
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
