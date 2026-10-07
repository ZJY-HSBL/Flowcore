"""Benchmark sequential, Hillis-Steele, and work-efficient affine scans.

Examples
--------
CPU smoke test::

    python benchmarks/benchmark_scan.py --device cpu --lengths 128,512,2048

CUDA sweep with torch.compile::

    python benchmarks/benchmark_scan.py --device cuda --lengths 256,1024,4096 \
        --batch 16 --blocks 32 --block-size 4 --compile

The benchmark reports wall-clock latency, state-steps/s and (on CUDA) peak
allocated memory.  Algorithmic composition counts are reported separately from
runtime because lower dependency depth does not guarantee a speedup.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from flowcore.operators import (
    make_compiled_scan,
    parallel_affine_scan,
    scan_composition_count,
    sequential_affine_scan,
)


@dataclass(slots=True)
class Result:
    device: str
    dtype: str
    algorithm: str
    compiled: bool
    batch: int
    time: int
    blocks: int
    block_size: int
    milliseconds: float
    state_steps_per_second: float
    composition_count: int | None
    peak_memory_mb: float | None


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stable_block_operators(batch: int, time_steps: int, blocks: int, d: int, device, dtype):
    # Near-identity stable dynamics make the benchmark representative of long
    # memory without creating artificial overflow at large T.
    eye = torch.eye(d, device=device, dtype=dtype).view(1, 1, 1, d, d)
    noise = 0.01 * torch.randn(batch, time_steps, blocks, d, d, device=device, dtype=dtype)
    a = 0.94 * eye + noise
    row = a.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
    a = 0.98 * a / row
    b = 0.01 * torch.randn(batch, time_steps, blocks, d, device=device, dtype=dtype)
    h0 = torch.zeros(batch, blocks, d, device=device, dtype=dtype)
    return a, b, h0


def timed(fn, a, b, h0, *, warmup: int, repeats: int, device: torch.device):
    for _ in range(warmup):
        fn(a, b, h0)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn(a, b, h0)
    synchronize(device)
    elapsed = (time.perf_counter() - start) / repeats
    peak = None
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    return elapsed, peak


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--lengths", type=parse_csv_ints, default=[128, 512, 2048])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--blocks", type=int, default=16)
    p.add_argument("--block-size", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise SystemExit("float16 CPU matmul is not a useful benchmark; use float32/bfloat16")

    results: list[Result] = []
    for time_steps in args.lengths:
        a, bias, h0 = stable_block_operators(
            args.batch, time_steps, args.blocks, args.block_size, device, dtype
        )

        cases: list[tuple[str, bool, object, int | None]] = [
            (
                "sequential",
                False,
                lambda aa, bb, hh: sequential_affine_scan(aa, bb, hh, kind="block"),
                None,
            ),
            (
                "hillis_steele",
                False,
                lambda aa, bb, hh: parallel_affine_scan(
                    aa, bb, hh, kind="block", algorithm="hillis_steele"
                ),
                scan_composition_count(time_steps, "hillis_steele"),
            ),
            (
                "work_efficient",
                False,
                lambda aa, bb, hh: parallel_affine_scan(
                    aa, bb, hh, kind="block", algorithm="work_efficient"
                ),
                scan_composition_count(time_steps, "work_efficient"),
            ),
        ]
        if args.compile:
            cases.append(
                (
                    "work_efficient",
                    True,
                    make_compiled_scan(kind="block", algorithm="work_efficient"),
                    scan_composition_count(time_steps, "work_efficient"),
                )
            )

        print(f"\nT={time_steps} batch={args.batch} blocks={args.blocks} d={args.block_size}")
        for name, compiled, fn, composition_count in cases:
            with torch.inference_mode():
                seconds, peak = timed(
                    fn, a, bias, h0, warmup=args.warmup, repeats=args.repeats, device=device
                )
            throughput = args.batch * time_steps / seconds
            result = Result(
                device=str(device),
                dtype=args.dtype,
                algorithm=name,
                compiled=compiled,
                batch=args.batch,
                time=time_steps,
                blocks=args.blocks,
                block_size=args.block_size,
                milliseconds=1000 * seconds,
                state_steps_per_second=throughput,
                composition_count=composition_count,
                peak_memory_mb=peak,
            )
            results.append(result)
            compile_label = "+compile" if compiled else ""
            work = "serial recurrence" if composition_count is None else f"{composition_count} comp"
            mem = "" if peak is None else f" peak={peak:.1f}MB"
            print(
                f"  {name:15s}{compile_label:9s} {result.milliseconds:9.3f} ms  "
                f"{throughput:12.0f} state-step/s  {work}{mem}"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
