# FlowCore v0.2 — parallelization kernel report

FlowCore v0.2 focuses on a narrow question: can the discrete dynamic substrate be
represented by an associative operator family whose time evolution has logarithmic
dependency depth **without** paying unnecessary `O(T log T)` composition work?

## 1. Work-efficient affine scan

v0.1 used an inclusive Hillis-Steele scan.  Its dependency depth is `O(log T)`,
but the number of affine compositions is `O(T log T)`.

v0.2 adds a Blelloch upsweep/downsweep scan.  It pads the sequence to a power of
two, computes an exclusive prefix in a tree, and converts it to an inclusive
prefix.  For a power-of-two sequence length `T`, the implementation performs:

```text
upsweep     T - 1

downsweep   T - 1

inclusive   T

----------------

total       3T - 2 affine compositions
```

Thus the algorithm has `O(T)` composition work and `O(log T)` dependency depth.
Hillis-Steele remains available as a reference backend.

## 2. Numerical equivalence

The test suite now checks:

- dense sequential vs parallel state equivalence;
- block sequential vs parallel state equivalence;
- diagonal sequential vs parallel state equivalence;
- sequential vs parallel gradient equivalence;
- Blelloch vs Hillis-Steele equivalence for non-power-of-two sequence lengths;
- stable 1025-step diagonal dynamics remain finite and agree with the sequential
  reference.

Local reference run:

```text
11 passed
```

## 3. `torch.compile`

`make_compiled_scan()` creates an opt-in compiled scan callable.  Compilation is
not enabled implicitly because compile latency can dominate short experiments.
The primary target is repeated, static-shape execution.

A local CPU smoke run (PyTorch 2.10.0, small block shape; **not** a general speed
claim) produced:

```text
T=128, batch=2, blocks=4, block_size=2
sequential                    1.161 ms
Hillis-Steele                 0.489 ms
work-efficient Blelloch       0.516 ms
work-efficient + compile      0.293 ms
```

Different warmup states, hardware, shapes and compiler versions can change the
ordering.  CUDA measurements are still required before claiming production
speedups.

A second uncompiled CPU smoke sweep with the same tiny block geometry showed the
expected scaling trend:

```text
T=512   sequential 4.891 ms   Hillis-Steele 1.507 ms   work-efficient 1.069 ms
T=1024  sequential 10.481 ms  Hillis-Steele 2.793 ms   work-efficient 2.187 ms
T=2048  sequential 22.373 ms  Hillis-Steele 4.466 ms   work-efficient 3.567 ms
```

This is encouraging evidence that reducing scan work matters even before a CUDA
kernel, but it is still a single CPU environment and should not be generalized.

## 4. Algorithmic work example

At `T=4096`:

```text
work-efficient: 3*T - 2 = 12,286 compositions
Hillis-Steele:            45,057 compositions
```

This is an algorithmic composition count, not a FLOP count.  A block composition
is more expensive than a serial block matvec, so runtime must still be measured.

## 5. Long-horizon diagnostic smoke test

`experiments/stability_sweep.py` sweeps sequence length and diagonal gain while
comparing the parallel trajectory with the sequential recurrence.

One local float32 CPU run gave the following representative observations:

- gains 0.70, 0.90 and 0.99 remained finite through 2048 steps;
- parallel/sequential relative max error stayed below roughly `5e-7` in those
  stable regimes;
- gain 1.001 also stayed finite in this short smoke run, but state magnitude grew
  strongly by 2048 steps (growth ratio about `7.9x`).

This is a numerical diagnostic, not a theorem about learned FlowCore stability.
It demonstrates why learned dynamics need explicit monitoring near the critical
regime.

## 6. Current conclusion

v0.2 strengthens the parallelization hypothesis at the **kernel** level:

1. affine Flow operators form an associative family;
2. state and gradient results match serial recurrence in tested regimes;
3. time dependency depth can be logarithmic;
4. operator-composition work can also be linear in sequence length;
5. `torch.compile` can compile the work-efficient implementation on the tested
   CPU environment.

What remains unproven is the important engineering question: whether the full
FlowCore model provides an end-to-end CUDA throughput/efficiency advantage at
useful state sizes and sequence lengths.  That is the next benchmark target.
