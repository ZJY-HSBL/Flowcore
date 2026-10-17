"""FlowCore v0.14: exact global-assignment margin against local objectives.

Compare five eight-parameter controllers on exactly identical training batches:
  - reconstruction: v0.12 soft Sinkhorn payload reconstruction MSE;
  - margin: bidirectional hard-negative margin using correspondence derived
    solely from the same target/source payloads used for reconstruction;
  - hybrid: reconstruction plus weighted local margin;
  - global_margin: full-permutation highest-scoring incorrect rival;
  - global_hybrid: reconstruction plus weighted global margin.

All models share the same uniform initialization, parameterization, Adam
settings, seed, and held-out sets. Zero-param uniform quantile and full-distance
baselines are evaluated alongside them.

CAUTION: Although no additional data are introduced, the margin loss explicitly
extracts correspondence labels from the provided payload pairs. This is a
stronger, more direct form of supervision than the reconstruction objective.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from flowcore import geometric_temperature
from flowcore.invariant_router import InvariantPermutationController
from flowcore.route_losses import (
    bidirectional_hard_negative_margin,
    global_permutation_hinge,
    infer_payload_correspondence,
)
from flowcore.structural_controller import QuantileInvariantPermutationController
from structural_metric_stress import evaluate, fresh_matching_batch, floats, ints


OBJECTIVES = ("reconstruction", "margin", "hybrid", "global_margin", "global_hybrid")
BASELINES = ("uniform_quantile", "full_distance")


@dataclass(slots=True)
class LossRow:
    seed: int
    controller: str
    modules: int
    noise: float
    slot_accuracy: float
    permutation_exact: float
    payload_mse: float


def train_seed(seed: int, args, device: torch.device) -> list[LossRow]:
    torch.manual_seed(seed)
    models = {
        objective: QuantileInvariantPermutationController(
            bins=args.bins, learnable=True,
            sinkhorn_iterations=args.sinkhorn_iterations,
        ).to(device)
        for objective in OBJECTIVES
    }
    models.update({
        "uniform_quantile": QuantileInvariantPermutationController(
            bins=args.bins, learnable=False,
            sinkhorn_iterations=args.sinkhorn_iterations,
        ).to(device),
        "full_distance": InvariantPermutationController().to(device),
    })

    # All learned models begin at exactly the same uniform metric.
    initial = models["reconstruction"].raw_weights.detach().clone()
    for name in OBJECTIVES:
        torch.testing.assert_close(models[name].raw_weights.detach(), initial)

    optimizers = {
        name: torch.optim.Adam(models[name].parameters(), lr=args.lr)
        for name in OBJECTIVES
    }
    generator = torch.Generator(device=device)
    generator.manual_seed(30000 + seed)

    for step in range(args.steps):
        noise = args.train_noise_min + (
            args.train_noise_max - args.train_noise_min
        ) * torch.rand((), device=device, generator=generator).item()
        source, target, payload, target_payload, _ = fresh_matching_batch(
            args.batch, args.train_modules, args.context_dim, args.local_dim,
            noise, device=device, generator=generator,
        )
        correspondence = infer_payload_correspondence(payload, target_payload)
        temp = geometric_temperature(
            step, args.steps,
            start=args.temperature_start, end=args.temperature_end,
        )

        for name in OBJECTIVES:
            model = models[name]
            scores = model.score_matrix(source, target)
            if name in ("reconstruction", "hybrid", "global_hybrid"):
                # Stable log-space Sinkhorn. No exponentiation of raw logits.
                from flowcore import sinkhorn_matrix
                soft = sinkhorn_matrix(
                    scores, temperature=temp, iterations=args.sinkhorn_iterations
                )
                reconstruction = (torch.matmul(soft, payload) - target_payload).square().mean()
            else:
                reconstruction = None

            if name in ("margin", "hybrid"):
                margin_term = bidirectional_hard_negative_margin(
                    scores, correspondence, margin=args.margin,
                )
            else:
                margin_term = None

            if name in ("global_margin", "global_hybrid"):
                global_margin = global_permutation_hinge(
                    scores, correspondence, margin=args.margin,
                    max_modules=args.max_exact_modules,
                )
            else:
                global_margin = None

            if name == "reconstruction":
                loss = reconstruction
            elif name == "margin":
                loss = margin_term
            elif name == "hybrid":
                loss = reconstruction + args.margin_weight * margin_term
            elif name == "global_margin":
                loss = global_margin
            else:
                loss = reconstruction + args.margin_weight * global_margin

            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

    for name in OBJECTIVES:
        weights = models[name].metric_weights().detach().cpu().tolist()
        print(
            f"seed={seed} {name:14s} weights="
            + ",".join(f"{value:.3f}" for value in weights)
        )

    rows: list[LossRow] = []
    for modules in args.eval_modules:
        for noise in args.eval_noises:
            for name in OBJECTIVES + BASELINES:
                metric = evaluate(
                    models[name], name, seed=seed, modules=modules,
                    noise=noise, args=args, device=device,
                )
                rows.append(LossRow(**asdict(metric)))
    return rows


def aggregate(rows: list[LossRow], args) -> dict:
    summary = {}
    for modules in args.eval_modules:
        for noise in args.eval_noises:
            for name in OBJECTIVES + BASELINES:
                subset = [
                    row for row in rows
                    if row.modules == modules and row.noise == noise
                    and row.controller == name
                ]
                key = f"{modules}/{noise:.3f}/{name}"
                summary[key] = {
                    "slot_accuracy_mean": statistics.fmean(
                        row.slot_accuracy for row in subset
                    ),
                    "slot_accuracy_std": statistics.stdev(
                        row.slot_accuracy for row in subset
                    ) if len(subset) > 1 else 0.0,
                    "permutation_exact_mean": statistics.fmean(
                        row.permutation_exact for row in subset
                    ),
                    "payload_mse_mean": statistics.fmean(
                        row.payload_mse for row in subset
                    ),
                }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=ints, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--eval-batch", type=int, default=64)
    parser.add_argument("--train-modules", type=int, default=6)
    parser.add_argument("--eval-modules", type=ints, default=[6, 8])
    parser.add_argument("--context-dim", type=int, default=6)
    parser.add_argument("--local-dim", type=int, default=3)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--train-noise-min", type=float, default=0.03)
    parser.add_argument("--train-noise-max", type=float, default=0.25)
    parser.add_argument("--eval-noises", type=floats, default=[0.03, 0.12, 0.20, 0.30])
    parser.add_argument("--temperature-start", type=float, default=0.5)
    parser.add_argument("--temperature-end", type=float, default=0.12)
    parser.add_argument("--sinkhorn-iterations", type=int, default=20)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--max-exact-modules", type=int, default=8)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("results/v14_global_assignment.json")
    )
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if max(args.eval_modules) > 12 or min(args.eval_modules) < 3:
        raise SystemExit("evaluation module counts must be between 3 and 12")
    if args.train_modules > args.max_exact_modules:
        raise SystemExit("global assignment training exceeds exhaustive module limit")
    if args.margin < 0 or args.margin_weight < 0:
        raise SystemExit("margin and margin_weight must be >=0")

    rows: list[LossRow] = []
    for seed in args.seeds:
        rows.extend(train_seed(seed, args, device))

    summary = aggregate(rows, args)
    print("modules noise objective          slot    exact    payload_mse")
    for key, metric in summary.items():
        modules, noise, name = key.split("/")
        print(
            f"{modules:>7} {noise:>5} {name:18s} "
            f"{metric['slot_accuracy_mean']:.4f} "
            f"{metric['permutation_exact_mean']:.4f} "
            f"{metric['payload_mse_mean']:.5f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "results": [asdict(row) for row in rows],
                "summary": summary,
            },
            indent=2,
        ) + "\n"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
