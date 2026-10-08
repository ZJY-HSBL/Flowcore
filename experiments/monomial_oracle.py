"""FlowCore v0.7: monomial oracle on arbitrary permutation routing.

This experiment does not claim an end-to-end differentiable permutation
controller.  It supplies the requested discrete permutation directly to the
monomial substrate to test the algebraic representational ceiling.
"""

from __future__ import annotations

import argparse

import torch

from flowcore import (
    arbitrary_permutation_bank,
    make_permutation_batch,
    parallel_monomial_scan,
    permutation_metrics,
    sequential_monomial_scan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modules", type=int, default=6)
    parser.add_argument("--local-dim", type=int, default=3)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--time", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    permutations = arbitrary_permutation_bank(
        args.tasks, args.modules, seed=123
    ).to(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    batch = make_permutation_batch(
        args.batch,
        args.time,
        permutations,
        args.local_dim,
        device=device,
        generator=generator,
    )

    identity = torch.arange(
        args.modules, device=device, dtype=torch.long
    ).view(1, 1, args.modules)
    operator_permutation = identity.expand(
        args.batch, args.time, args.modules
    ).clone()
    operator_permutation[:, 1] = permutations[batch.task_id]

    gain = torch.ones(
        args.batch,
        args.time,
        args.modules,
        device=device,
    )
    eye = torch.eye(
        args.local_dim, device=device
    ).view(1, 1, args.local_dim, args.local_dim)
    local = eye.expand(
        args.batch,
        args.time,
        args.local_dim,
        args.local_dim,
    )
    bias = batch.injection.view(
        args.batch,
        args.time,
        args.modules,
        args.local_dim,
    )

    h0 = torch.zeros(
        args.batch,
        args.modules,
        args.local_dim,
        device=device,
    )
    with torch.inference_mode():
        parallel = parallel_monomial_scan(
            operator_permutation, gain, local, bias, h0
        )
        sequential = sequential_monomial_scan(
            operator_permutation, gain, local, bias, h0
        )

    flat_parallel = parallel.reshape(
        args.batch, args.time, args.modules * args.local_dim
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
    print(f"mse={metrics.mse:.8f}")
    print(f"relative_mse={metrics.relative_mse:.8f}")
    print(f"slot_accuracy={metrics.slot_accuracy:.6f}")
    print(f"parallel_error={metrics.max_parallel_error:.3e}")


if __name__ == "__main__":
    main()
