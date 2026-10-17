"""FlowCore v0.11: learned fixed-view matching vs invariant OOD-view baseline.

Train the v0.10 learned controller with fixed source/target observation views.
Then evaluate on both the training view pair and a wholly new orthogonal view
pair. Compare against a zero-parameter distance-fingerprint matching controller.

The invariant method is a deliberately analytical baseline; it is not trained
and must not be described as learned semantic generalization.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from flowcore import MatchingPermutationController, geometric_temperature
from flowcore.invariant_router import InvariantPermutationController
from context_route_generalization import make_batch, orthogonal_view


@dataclass(slots=True)
class OODRow:
    seed: int
    controller: str
    view: str
    noise: float
    slot_accuracy: float
    permutation_exact: float
    hard_payload_mse: float


def parse_csv_ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v.strip()]


def parse_csv_floats(s: str) -> list[float]:
    return [float(v) for v in s.split(",") if v.strip()]


def evaluate(
    controller,
    source_view,
    target_view,
    *,
    seed: int,
    args,
    device,
    view: str,
    noise: float,
    controller_name: str,
) -> OODRow:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 50000 + int(noise * 1000) + (0 if view == "trained" else 10000))
    (
        source_context, target_context, payload, target_payload, permutation,
    ) = make_batch(
        args.eval_batch,
        args.modules,
        args.local_dim,
        args.context_dim,
        source_view,
        target_view,
        noise,
        device=device,
        generator=gen,
    )
    with torch.inference_mode():
        predicted = controller.hard_permutation(source_context, target_context)
        index = predicted.unsqueeze(-1).expand(-1, -1, args.local_dim)
        routed = torch.gather(payload, 1, index)
        return OODRow(
            seed=seed,
            controller=controller_name,
            view=view,
            noise=noise,
            slot_accuracy=float((predicted == permutation).float().mean().cpu()),
            permutation_exact=float(
                (predicted == permutation).all(dim=-1).float().mean().cpu()
            ),
            hard_payload_mse=float((routed - target_payload).square().mean().cpu()),
        )


def run_seed(seed: int, args, device: torch.device) -> list[OODRow]:
    torch.manual_seed(seed)
    train_source = orthogonal_view(args.context_dim, 100 + seed, device)
    train_target = orthogonal_view(args.context_dim, 200 + seed, device)
    heldout_source = orthogonal_view(args.context_dim, 10000 + seed, device)
    heldout_target = orthogonal_view(args.context_dim, 20000 + seed, device)

    learned = MatchingPermutationController(
        args.context_dim, args.rank, sinkhorn_iterations=args.sinkhorn_iterations
    ).to(device)
    opt = torch.optim.Adam(learned.parameters(), lr=args.lr)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 9000)
    for step in range(args.steps):
        src, tgt, payload, target_payload, _ = make_batch(
            args.batch,
            args.modules,
            args.local_dim,
            args.context_dim,
            train_source,
            train_target,
            args.train_noise,
            device=device,
            generator=gen,
        )
        temperature = geometric_temperature(
            step, args.steps, start=1.0, end=args.temperature_end
        )
        soft = learned.soft_matrix(src, tgt, temperature=temperature)
        routed = torch.matmul(soft, payload)
        mse = (routed - target_payload).square().mean()
        entropy = -(soft.clamp_min(1e-9) * soft.clamp_min(1e-9).log()).sum(-1).mean()
        loss = mse + args.entropy_weight * entropy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learned.parameters(), 5.0)
        opt.step()

    invariant = InvariantPermutationController().to(device)
    rows = []
    for view, source_view, target_view in [
        ("trained", train_source, train_target),
        ("unseen", heldout_source, heldout_target),
    ]:
        for noise in args.noises:
            for model, name in ((learned, "learned_fixed_view"), (invariant, "invariant_distance")):
                rows.append(
                    evaluate(
                        model, source_view, target_view,
                        seed=seed, args=args, device=device,
                        view=view, noise=noise, controller_name=name
                    )
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seeds", type=parse_csv_ints, default=[0, 1, 2])
    parser.add_argument("--noises", type=parse_csv_floats, default=[0.03, 0.06, 0.12, 0.20])
    parser.add_argument("--train-noise", type=float, default=0.03)
    parser.add_argument("--modules", type=int, default=6)
    parser.add_argument("--local-dim", type=int, default=3)
    parser.add_argument("--context-dim", type=int, default=6)
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--temperature-end", type=float, default=0.05)
    parser.add_argument("--entropy-weight", type=float, default=0.005)
    parser.add_argument("--sinkhorn-iterations", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/v11_unseen_views.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    rows = []
    for seed in args.seeds:
        rows.extend(run_seed(seed, args, device))

    summary = {}
    for controller in ("learned_fixed_view", "invariant_distance"):
        for view in ("trained", "unseen"):
            for noise in args.noises:
                selected = [
                    r for r in rows
                    if r.controller == controller and r.view == view and r.noise == noise
                ]
                key = f"{controller}/{view}/{noise:.3f}"
                summary[key] = {
                    "slot_accuracy_mean": statistics.fmean(r.slot_accuracy for r in selected),
                    "permutation_exact_mean": statistics.fmean(r.permutation_exact for r in selected),
                    "hard_payload_mse_mean": statistics.fmean(r.hard_payload_mse for r in selected),
                }

    print("controller            view     noise   slot     exact    mse")
    for key, data in summary.items():
        controller, view, noise = key.split("/")
        print(
            f"{controller:21s} {view:8s} {noise:5s} "
            f"{data['slot_accuracy_mean']:.4f}   "
            f"{data['permutation_exact_mean']:.4f}   "
            f"{data['hard_payload_mse_mean']:.5f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "results": [asdict(row) for row in rows],
                "summary": summary,
            },
            indent=2,
        ) + "\n"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
