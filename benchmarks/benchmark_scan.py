"""Microbenchmark sequential recurrence vs vectorized parallel scan.

Dense/block scan has more arithmetic work than a serial matvec recurrence; the
point of this script is to expose the trade-off rather than promise universal
speedups on every device/sequence length.
"""

from __future__ import annotations

import argparse
import time

import torch

from flowcore.operators import parallel_affine_scan, sequential_affine_scan


def bench(fn, *args, warmup=5, repeats=20, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    if args[0].is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn(*args, **kwargs)
    if args[0].is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--time", type=int, default=1024)
    p.add_argument("--blocks", type=int, default=32)
    p.add_argument("--block-size", type=int, default=4)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shape = (args.batch, args.time, args.blocks, args.block_size, args.block_size)
    a = 0.05 * torch.randn(*shape, device=device)
    b = torch.randn(args.batch, args.time, args.blocks, args.block_size, device=device)
    h0 = torch.zeros(args.batch, args.blocks, args.block_size, device=device)

    ts = bench(sequential_affine_scan, a, b, h0, kind="block")
    tp = bench(parallel_affine_scan, a, b, h0, kind="block")
    print(f"device={device} sequential={ts*1e3:.2f}ms parallel_scan={tp*1e3:.2f}ms")
    print(f"ratio sequential/parallel={ts/tp:.3f}x")


if __name__ == "__main__":
    main()
