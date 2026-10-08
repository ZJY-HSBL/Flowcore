"""FlowCore v0.8: learn a discrete permutation controller from task loss.

Training uses only reconstruction loss through a Sinkhorn-relaxed routing matrix.
No permutation labels are used in the loss. Evaluation projects the learned
matrix to an exact assignment and executes it with the monomial parallel scan.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from flowcore import (
    TaskPermutationController,
    arbitrary_permutation_bank,
    geometric_temperature,
    make_permutation_batch,
    parallel_monomial_scan,
    permutation_metrics,
    sequential_monomial_scan,
)


@dataclass(slots=True)
class LearnedRouteResult:
    seed: int
    soft_mse: float
    hard_mse: float
    hard_slot_accuracy: float
    entry_accuracy: float
    task_exact_accuracy: float
    parallel_error: float
    controller_params: int
    route_vector_baseline_params: int


def parse_int_csv(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def train_one(seed: int, args, device: torch.device) -> LearnedRouteResult:
    torch.manual_seed(seed)
    permutations = arbitrary_permutation_bank(
        args.tasks, args.modules, seed=args.permutation_seed
    ).to(device)

    controller = TaskPermutationController(
        args.tasks,
        args.modules,
        sinkhorn_iterations=args.sinkhorn_iterations,
    ).to(device)
    optimizer = torch.optim.Adam(
        controller.parameters(), lr=args.lr
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 5000)

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
            batch.task_id, temperature=temperature
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
    eval_generator.manual_seed(seed + 6000)
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

        task_ids = torch.arange(
            args.tasks, device=device
        )
        predicted_tasks = controller.hard_permutation(
            task_ids,
            temperature=args.temperature_end,
        )
        entry_accuracy = float(
            (predicted_tasks == permutations)
            .float()
            .mean()
            .cpu()
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
            args.modules,
            device=device,
            dtype=torch.long,
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
            operator_permutation,
            gain,
            local,
            bias,
            h0,
        )
        sequential = sequential_monomial_scan(
            operator_permutation,
            gain,
            local,
            bias,
            h0,
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

    return LearnedRouteResult(
        seed=seed,
        soft_mse=soft_mse,
        hard_mse=metrics.mse,
        hard_slot_accuracy=metrics.slot_accuracy,
        entry_accuracy=entry_accuracy,
        task_exact_accuracy=task_exact_accuracy,
        parallel_error=metrics.max_parallel_error or 0.0,
        controller_params=sum(
            p.numel() for p in controller.parameters()
        ),
        route_vector_baseline_params=args.tasks * args.modules,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=parse_int_csv, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--modules", type=int, default=6)
    parser.add_argument("--local-dim", type=int, default=3)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--time", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.15)
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
        default=Path("results/v08_discrete_controller.json"),
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    results = [
        train_one(seed, args, device)
        for seed in args.seeds
    ]
    for result in results:
        print(
            f"seed={result.seed} "
            f"soft_mse={result.soft_mse:.6f} "
            f"hard_mse={result.hard_mse:.6f} "
            f"slot={result.hard_slot_accuracy:.4f} "
            f"entry={result.entry_accuracy:.4f} "
            f"task_exact={result.task_exact_accuracy:.4f} "
            f"par_err={result.parallel_error:.2e}"
        )

    summary = {
        "soft_mse_mean": statistics.fmean(r.soft_mse for r in results),
        "hard_mse_mean": statistics.fmean(r.hard_mse for r in results),
        "hard_slot_accuracy_mean": statistics.fmean(
            r.hard_slot_accuracy for r in results
        ),
        "entry_accuracy_mean": statistics.fmean(
            r.entry_accuracy for r in results
        ),
        "task_exact_accuracy_mean": statistics.fmean(
            r.task_exact_accuracy for r in results
        ),
        "parallel_error_max": max(r.parallel_error for r in results),
        "controller_params": results[0].controller_params,
        "route_vector_baseline_params": results[0].route_vector_baseline_params,
    }
    print(summary)

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
