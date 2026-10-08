"""FlowCore v0.10: content-conditioned routing without task IDs.

Every batch contains fresh random module addresses and a fresh random permutation.
The controller sees source descriptors in one observation basis and target
descriptors in another basis. It must infer the permutation from matching
content, not from a stored task index.

Training uses only payload reconstruction through a Sinkhorn-relaxed route.
Evaluation hard-projects the route and executes it with the exact monomial scan.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from flowcore import (
    MatchingPermutationController,
    geometric_temperature,
    parallel_monomial_scan,
    permutation_metrics,
    sequential_monomial_scan,
)


@dataclass(slots=True)
class ContextRouteResult:
    seed: int
    soft_mse: float
    hard_mse: float
    slot_accuracy: float
    permutation_accuracy: float
    parallel_error: float
    higher_noise_slot_accuracy: float
    controller_params: int


def parse_int_csv(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def orthogonal_view(dim: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    q, _ = torch.linalg.qr(
        torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    )
    return q.float().to(device)


def make_batch(
    batch_size: int,
    modules: int,
    local_dim: int,
    context_dim: int,
    source_view: torch.Tensor,
    target_view: torch.Tensor,
    noise: float,
    *,
    device: torch.device,
    generator: torch.Generator,
):
    latent = torch.randn(
        batch_size, modules, context_dim,
        device=device, generator=generator,
    )
    payload = torch.randn(
        batch_size, modules, local_dim,
        device=device, generator=generator,
    )
    permutation = torch.stack(
        [
            torch.randperm(modules, device=device, generator=generator)
            for _ in range(batch_size)
        ]
    )

    source_context = latent @ source_view
    target_latent = torch.gather(
        latent,
        1,
        permutation.unsqueeze(-1).expand(-1, -1, context_dim),
    )
    target_context = target_latent @ target_view

    if noise > 0:
        source_context = source_context + noise * torch.randn(
            source_context.shape,
            device=device,
            generator=generator,
        )
        target_context = target_context + noise * torch.randn(
            target_context.shape,
            device=device,
            generator=generator,
        )

    target_payload = torch.gather(
        payload,
        1,
        permutation.unsqueeze(-1).expand(-1, -1, local_dim),
    )
    return (
        source_context,
        target_context,
        payload,
        target_payload,
        permutation,
    )


def hard_execute(
    predicted: torch.Tensor,
    payload: torch.Tensor,
    target_payload: torch.Tensor,
    true_permutation: torch.Tensor,
    *,
    time_steps: int,
):
    batch_size, modules, local_dim = payload.shape
    device = payload.device

    identity = torch.arange(
        modules, device=device, dtype=torch.long
    ).view(1, 1, modules)
    operator_permutation = identity.expand(
        batch_size, time_steps, modules
    ).clone()
    operator_permutation[:, 1] = predicted

    gain = torch.ones(
        batch_size, time_steps, modules, device=device
    )
    local = torch.eye(
        local_dim, device=device
    ).view(1, 1, local_dim, local_dim).expand(
        batch_size, time_steps, local_dim, local_dim
    )
    bias = torch.zeros(
        batch_size, time_steps, modules, local_dim, device=device
    )
    bias[:, 0] = payload
    h0 = torch.zeros(
        batch_size, modules, local_dim, device=device
    )

    parallel = parallel_monomial_scan(
        operator_permutation, gain, local, bias, h0
    )
    sequential = sequential_monomial_scan(
        operator_permutation, gain, local, bias, h0
    )

    flat_parallel = parallel.reshape(
        batch_size, time_steps, modules * local_dim
    )
    flat_sequential = sequential.reshape_as(flat_parallel)

    # permutation_metrics needs a task bank/task id; here every sample has its
    # own unseen permutation, so compute the equivalent metrics directly.
    final = parallel[:, -1]
    hard_mse = float(
        (final - target_payload).square().mean().cpu()
    )
    pairwise = (
        final[:, :, None, :] - payload[:, None, :, :]
    ).square().mean(dim=-1)
    recovered = pairwise.argmin(dim=-1)
    slot_accuracy = float(
        (recovered == true_permutation).float().mean().cpu()
    )
    exact = float(
        (predicted == true_permutation)
        .all(dim=-1).float().mean().cpu()
    )
    parallel_error = float(
        (flat_parallel - flat_sequential).abs().amax().cpu()
    )
    return hard_mse, slot_accuracy, exact, parallel_error


def train_one(seed: int, args, device: torch.device) -> ContextRouteResult:
    torch.manual_seed(seed)
    source_view = orthogonal_view(
        args.context_dim, 100 + seed, device
    )
    target_view = orthogonal_view(
        args.context_dim, 200 + seed, device
    )

    controller = MatchingPermutationController(
        args.context_dim,
        args.rank,
        sinkhorn_iterations=args.sinkhorn_iterations,
    ).to(device)
    optimizer = torch.optim.Adam(
        controller.parameters(), lr=args.lr
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 9000)

    for step in range(args.steps):
        (
            source_context,
            target_context,
            payload,
            target_payload,
            _,
        ) = make_batch(
            args.batch,
            args.modules,
            args.local_dim,
            args.context_dim,
            source_view,
            target_view,
            args.noise,
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
            source_context,
            target_context,
            temperature=temperature,
        )
        routed = torch.matmul(soft, payload)
        mse = (routed - target_payload).square().mean()
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
    eval_generator.manual_seed(seed + 10000)
    (
        source_context,
        target_context,
        payload,
        target_payload,
        permutation,
    ) = make_batch(
        args.eval_batch,
        args.modules,
        args.local_dim,
        args.context_dim,
        source_view,
        target_view,
        args.noise,
        device=device,
        generator=eval_generator,
    )

    with torch.inference_mode():
        soft = controller.soft_matrix(
            source_context,
            target_context,
            temperature=args.temperature_end,
        )
        soft_mse = float(
            (torch.matmul(soft, payload) - target_payload)
            .square().mean().cpu()
        )
        predicted = controller.hard_permutation(
            source_context,
            target_context,
            temperature=args.temperature_end,
        )
        hard_mse, slot, exact, par_err = hard_execute(
            predicted,
            payload,
            target_payload,
            permutation,
            time_steps=args.time,
        )

        (
            noisy_source,
            noisy_target,
            noisy_payload,
            noisy_target_payload,
            noisy_permutation,
        ) = make_batch(
            args.eval_batch,
            args.modules,
            args.local_dim,
            args.context_dim,
            source_view,
            target_view,
            args.noise * args.noise_multiplier,
            device=device,
            generator=eval_generator,
        )
        noisy_predicted = controller.hard_permutation(
            noisy_source,
            noisy_target,
            temperature=args.temperature_end,
        )
        _, noisy_slot, _, _ = hard_execute(
            noisy_predicted,
            noisy_payload,
            noisy_target_payload,
            noisy_permutation,
            time_steps=args.time,
        )

    return ContextRouteResult(
        seed=seed,
        soft_mse=soft_mse,
        hard_mse=hard_mse,
        slot_accuracy=slot,
        permutation_accuracy=exact,
        parallel_error=par_err,
        higher_noise_slot_accuracy=noisy_slot,
        controller_params=sum(
            p.numel() for p in controller.parameters()
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=parse_int_csv, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--modules", type=int, default=6)
    parser.add_argument("--local-dim", type=int, default=3)
    parser.add_argument("--context-dim", type=int, default=6)
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--time", type=int, default=16)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--noise-multiplier", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--entropy-weight", type=float, default=0.005)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iterations", type=int, default=20)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v10_context_routing.json"),
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    results = [train_one(seed, args, device) for seed in args.seeds]
    for r in results:
        print(
            f"seed={r.seed} soft_mse={r.soft_mse:.6f} "
            f"hard_mse={r.hard_mse:.6f} slot={r.slot_accuracy:.4f} "
            f"perm_exact={r.permutation_accuracy:.4f} "
            f"noise2x_slot={r.higher_noise_slot_accuracy:.4f} "
            f"par_err={r.parallel_error:.2e}"
        )

    summary = {
        "soft_mse_mean": statistics.fmean(r.soft_mse for r in results),
        "hard_mse_mean": statistics.fmean(r.hard_mse for r in results),
        "slot_accuracy_mean": statistics.fmean(
            r.slot_accuracy for r in results
        ),
        "permutation_accuracy_mean": statistics.fmean(
            r.permutation_accuracy for r in results
        ),
        "higher_noise_slot_accuracy_mean": statistics.fmean(
            r.higher_noise_slot_accuracy for r in results
        ),
        "parallel_error_max": max(r.parallel_error for r in results),
        "controller_params": results[0].controller_params,
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
