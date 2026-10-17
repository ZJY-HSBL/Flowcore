"""FlowCore v0.12: learned invariant-structure metric under new views/sizes.

Train on M=6 under fresh, independent random orthogonal views, shifts and
positive global scales at every batch, using only payload reconstruction loss.
Evaluate on freshly generated M=6 and unseen M=8 point sets.  Compare three
independently evaluated controllers on identical data:
  1) learned 8-weight quantile-distance metric;
  2) uniform-weight quantile-distance metric (architecture-matched baseline);
  3) original v0.11 full sorted-distance analytical matching baseline.

No task IDs, permutation labels, or fixed observation bases are used in training.
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
from flowcore.structural_controller import QuantileInvariantPermutationController


@dataclass(slots=True)
class MetricRow:
    seed: int
    controller: str
    modules: int
    noise: float
    slot_accuracy: float
    permutation_exact: float
    payload_mse: float


def ints(s: str) -> list[int]:
    return [int(value.strip()) for value in s.split(",") if value.strip()]


def floats(s: str) -> list[float]:
    return [float(value.strip()) for value in s.split(",") if value.strip()]


def fresh_views(
    n: int, dim: int, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    matrices = torch.randn(n, dim, dim, device=device, generator=generator)
    q, _ = torch.linalg.qr(matrices)
    return q


def fresh_matching_batch(
    batch_size: int,
    modules: int,
    descriptor_dim: int,
    payload_dim: int,
    noise: float,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if modules < 3:
        raise ValueError("modules must be >=3")

    latent = torch.randn(
        batch_size, modules, descriptor_dim, device=device, generator=generator
    )
    payload = torch.randn(
        batch_size, modules, payload_dim, device=device, generator=generator
    )
    permutation = torch.stack(
        [
            torch.randperm(modules, device=device, generator=generator)
            for _ in range(batch_size)
        ]
    )
    source_view = fresh_views(batch_size, descriptor_dim, device, generator)
    target_view = fresh_views(batch_size, descriptor_dim, device, generator)

    target_latent = torch.gather(
        latent, 1,
        permutation.unsqueeze(-1).expand(-1, -1, descriptor_dim),
    )
    source = torch.bmm(latent, source_view)
    target = torch.bmm(target_latent, target_view)

    # Independent translation and global scale, both removed by normalized
    # within-set distance descriptors. Noise is added after view transforms.
    source_scale = (0.8 + 0.4 * torch.rand(
        batch_size, 1, 1, device=device, generator=generator
    ))
    target_scale = (0.8 + 0.4 * torch.rand(
        batch_size, 1, 1, device=device, generator=generator
    ))
    source = source * source_scale + torch.randn(
        batch_size, 1, descriptor_dim, device=device, generator=generator
    )
    target = target * target_scale + torch.randn(
        batch_size, 1, descriptor_dim, device=device, generator=generator
    )

    if noise > 0:
        source = source + noise * torch.randn(
            source.shape, device=device, generator=generator
        )
        target = target + noise * torch.randn(
            target.shape, device=device, generator=generator
        )

    target_payload = torch.gather(
        payload, 1,
        permutation.unsqueeze(-1).expand(-1, -1, payload_dim),
    )
    return source, target, payload, target_payload, permutation


@torch.no_grad()
def evaluate(
    controller,
    controller_name: str,
    *,
    seed: int,
    modules: int,
    noise: float,
    args,
    device: torch.device,
) -> MetricRow:
    generator = torch.Generator(device=device)
    # Exactly the same held-out examples for all controllers.
    generator.manual_seed(
        50000 + seed * 10000 + modules * 100 + int(round(noise * 1000))
    )
    source, target, payload, target_payload, expected = fresh_matching_batch(
        args.eval_batch, modules, args.context_dim, args.local_dim, noise,
        device=device, generator=generator,
    )
    predicted = controller.hard_permutation(source, target)
    predicted_payload = torch.gather(
        payload, 1, predicted.unsqueeze(-1).expand(-1, -1, args.local_dim)
    )
    return MetricRow(
        seed=seed,
        controller=controller_name,
        modules=modules,
        noise=noise,
        slot_accuracy=float((predicted == expected).float().mean().cpu()),
        permutation_exact=float(
            (predicted == expected).all(dim=-1).float().mean().cpu()
        ),
        payload_mse=float(
            (predicted_payload - target_payload).square().mean().cpu()
        ),
    )


def run_seed(seed: int, args, device: torch.device) -> list[MetricRow]:
    torch.manual_seed(seed)
    learned = QuantileInvariantPermutationController(
        bins=args.bins, learnable=True,
        sinkhorn_iterations=args.sinkhorn_iterations,
    ).to(device)
    uniform = QuantileInvariantPermutationController(
        bins=args.bins, learnable=False,
        sinkhorn_iterations=args.sinkhorn_iterations,
    ).to(device)
    original = InvariantPermutationController().to(device)
    optimizer = torch.optim.Adam(learned.parameters(), lr=args.lr)

    generator = torch.Generator(device=device)
    generator.manual_seed(30000 + seed)
    for step in range(args.steps):
        noise = float(
            args.train_noise_min + (args.train_noise_max - args.train_noise_min)
            * torch.rand((), generator=generator, device=device).item()
        )
        source, target, payload, target_payload, _ = fresh_matching_batch(
            args.batch, args.train_modules,
            args.context_dim, args.local_dim, noise,
            device=device, generator=generator,
        )
        temp = geometric_temperature(
            step, args.steps,
            start=args.temperature_start, end=args.temperature_end,
        )
        soft = learned.soft_matrix(source, target, temperature=temp)
        reconstruction = torch.matmul(soft, payload)
        loss = (reconstruction - target_payload).square().mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learned.parameters(), 5.0)
        optimizer.step()

    print(
        "seed=" + str(seed) + " learned metric weights="
        + ",".join(f"{x:.3f}" for x in learned.metric_weights().detach().cpu().tolist())
    )

    rows = []
    for modules in args.eval_modules:
        for noise in args.eval_noises:
            for controller, name in (
                (learned, "learned_quantile"),
                (uniform, "uniform_quantile"),
                (original, "full_distance"),
            ):
                rows.append(
                    evaluate(
                        controller, name,
                        seed=seed, modules=modules,
                        noise=noise, args=args, device=device,
                    )
                )
    return rows


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
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/v12_structural_metric.json"),
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    if max(args.eval_modules) > 12:
        raise SystemExit("small-M exact hard assignment supports at most 12 modules")
    if args.train_noise_min < 0 or args.train_noise_max < args.train_noise_min:
        raise SystemExit("invalid training noise range")
    rows = []
    for seed in args.seeds:
        rows.extend(run_seed(seed, args, device))

    summary = {}
    for modules in args.eval_modules:
        for noise in args.eval_noises:
            for controller in ("learned_quantile", "uniform_quantile", "full_distance"):
                selected = [
                    r for r in rows
                    if r.modules == modules and r.noise == noise
                    and r.controller == controller
                ]
                key = f"{modules}/{noise:.3f}/{controller}"
                summary[key] = {
                    "slot_accuracy_mean": statistics.fmean(
                        r.slot_accuracy for r in selected
                    ),
                    "permutation_exact_mean": statistics.fmean(
                        r.permutation_exact for r in selected
                    ),
                    "payload_mse_mean": statistics.fmean(
                        r.payload_mse for r in selected
                    ),
                }
    print("modules noise controller       slot    exact    payload_mse")
    for key, result in summary.items():
        modules, noise, name = key.split("/")
        print(
            f"{int(modules):7d} {noise:>5s} {name:17s} "
            f"{result['slot_accuracy_mean']:.4f} "
            f"{result['permutation_exact_mean']:.4f} "
            f"{result['payload_mse_mean']:.5f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "config": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "results": [asdict(r) for r in rows],
                "summary": summary,
            },
            indent=2,
        ) + "\n"
    )
    print("wrote", args.output)


if __name__ == "__main__":
    main()
