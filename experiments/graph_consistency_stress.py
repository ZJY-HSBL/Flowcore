"""FlowCore v0.15: pairwise geometric graph consistency and route confidence.

Matched zero-parameter methods, all using *the same* held-out random sets:
  fingerprint: original sorted-distance matching;
  graph_unweighted: fingerprint initialization + global pairwise consistency
                    objective refined by improving two-module swaps;
  graph_shortedge: same refinement with a fixed short-edge reliability weight.

No permutation labels, task identity, test-noise value, or extra information are
given to any router. These are small-M deterministic baselines, not learned
semantic routers. Confidence measures normalized local swap stability, **not**
a calibrated correctness probability. We evaluate whether retaining only the
top 50% by confidence improves full-assignment accuracy.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from flowcore.graph_routing import (
    graph_assignment_cost,
    normalized_distances,
    refine_geometric_assignment,
)
from flowcore.invariant_router import InvariantPermutationController
from structural_metric_stress import fresh_matching_batch, ints, floats


@dataclass(slots=True)
class GraphRow:
    seed: int
    modules: int
    noise: float
    controller: str
    slot_accuracy: float
    permutation_exact: float
    payload_mse: float
    confidence_mean: float
    top50_exact: float
    top50_slot_accuracy: float


def evaluate_one(seed: int, modules: int, noise: float, args, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(
        50000 + seed * 10000 + modules * 100 + int(round(noise * 1000))
    )
    source, target, payload, target_payload, expected = fresh_matching_batch(
        args.eval_batch, modules, args.context_dim, args.local_dim,
        noise, device=device, generator=generator,
    )
    with torch.inference_mode():
        fingerprint = InvariantPermutationController().hard_permutation(
            source, target
        )
        source_dist = normalized_distances(source)
        target_dist = normalized_distances(target)
        methods = {}
        for name, floor, rounds in (
            ("fingerprint", 0.0, 0),
            ("graph_unweighted", 0.0, args.rounds),
            ("graph_shortedge", args.reliability_floor, args.rounds),
        ):
            predicted, _, confidence = refine_geometric_assignment(
                source_dist,
                target_dist,
                fingerprint,
                reliability_floor=floor,
                rounds=rounds,
            )
            methods[name] = (predicted, confidence)

    results = []
    for name, (predicted, confidence) in methods.items():
        slot = (predicted == expected).float().mean(dim=1)
        exact = (predicted == expected).all(dim=1).float()
        reconstructed = payload.gather(
            1, predicted.unsqueeze(-1).expand(-1, -1, args.local_dim)
        )
        mse = (reconstructed - target_payload).square().mean().item()

        # No threshold is fitted to labels: select a preset 50%-coverage
        # fraction by descending stability confidence. This is selective
        # performance, not calibrated P(correct | confidence).
        count = max(1, args.eval_batch // 2)
        indices = confidence.argsort(descending=True)[:count]
        results.append(
            GraphRow(
                seed=seed,
                modules=modules,
                noise=noise,
                controller=name,
                slot_accuracy=slot.mean().item(),
                permutation_exact=exact.mean().item(),
                payload_mse=mse,
                confidence_mean=confidence.mean().item(),
                top50_exact=exact.index_select(0, indices).mean().item(),
                top50_slot_accuracy=slot.index_select(0, indices).mean().item(),
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=ints, default=[0, 1, 2])
    parser.add_argument("--modules", type=ints, default=[6, 8])
    parser.add_argument("--noises", type=floats, default=[0.03, 0.12, 0.20, 0.30])
    parser.add_argument("--eval-batch", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=6)
    parser.add_argument("--local-dim", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--reliability-floor", type=float, default=0.15)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/v15_graph_confidence.json")
    )
    args = parser.parse_args()
    if not args.seeds or not args.modules or not args.noises:
        parser.error("non-empty seeds, module counts and noise levels required")
    if min(args.modules) < 3 or max(args.modules) > 12:
        parser.error("module count must be in [3, 12] for exact reference matching")
    if min(args.noises) < 0 or args.reliability_floor < 0 or args.rounds < 0:
        parser.error("noise, reliability floor and rounds must be nonnegative")
    if args.eval_batch < 2:
        parser.error("eval batch must contain at least two samples")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")

    rows = []
    for seed in args.seeds:
        for modules in args.modules:
            for noise in args.noises:
                rows.extend(evaluate_one(seed, modules, noise, args, device))

    aggregate = {}
    for modules in args.modules:
        for noise in args.noises:
            for method in ("fingerprint", "graph_unweighted", "graph_shortedge"):
                found = [
                    row for row in rows
                    if row.modules == modules and row.noise == noise
                    and row.controller == method
                ]
                key = f"{modules}/{noise:.3f}/{method}"
                aggregate[key] = {
                    attr + "_mean": statistics.fmean(getattr(row, attr) for row in found)
                    for attr in (
                        "slot_accuracy", "permutation_exact",
                        "payload_mse", "confidence_mean",
                        "top50_exact", "top50_slot_accuracy",
                    )
                }
    print("modules noise method               slot    exact   selected50_exact  confidence")
    for key, values in aggregate.items():
        modules, noise, method = key.split("/")
        print(
            f"{modules:>7} {noise:>5} {method:20s} "
            f"{values['slot_accuracy_mean']:.4f}  "
            f"{values['permutation_exact_mean']:.4f}  "
            f"{values['top50_exact_mean']:.4f}  "
            f"{values['confidence_mean_mean']:.4f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "config": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "results": [asdict(row) for row in rows],
                "summary": aggregate,
                "confidence_note": "Stability ranking only; not calibrated probability",
            },
            indent=2,
        ) + "\n"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
