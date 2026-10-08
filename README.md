# FlowCore

**FlowCore** is an experimental PyTorch library for building **parallelizable
dynamic substrates** from composable discrete-time operators.

The project starts from a simple idea: keep the benefits of persistent recurrent
state and dynamic Route/Hold control, but express the substrate as an associative
operator sequence that can be evaluated with a parallel prefix scan.

> Status: research prototype.  This repository implements the mathematical and
> software kernel, not the full long-term cognitive architecture.

## Core abstraction

For each time step, FlowCore constructs an affine transition

```text
h[t+1] = A[t] h[t] + b[t]
```

with

```text
A[t] <- long-term W + Route + Hold
b[t] <- multi-port B_k injection + Hold
```

Affine transitions compose associatively:

```text
(A2,b2) o (A1,b1) = (A2 A1, A2 b1 + b2)
```

so an entire sequence can be evaluated through a vectorized parallel scan.

## Architecture

```text
input/context
     |
  Encoder
     |
     +------------------+
     |                  |
Port selector      Route/Hold controller
     |                  |
 B1 B2 ... Bk            |
     |                  |
     +------ injection --+
              |
      ParallelFlowSubstrate
      (block affine operators)
              |
        parallel scan
              |
          persistent state
              |
           Readout
```

The base model is exactly parallelizable because the controller is exogenous to
current recurrent state.  `RefinedFlowCoreModel` adds state-conditioned control
using a fixed number of predict-correct scan rounds.

## Why block operators?

Dense affine scans are useful as a correctness reference but composing dense
matrices costs too much for large state sizes.  The default substrate uses many
small block-local operators.  This gives a closed operator family with practical
vectorized composition while leaving room for richer hierarchical cross-module
routing later.

## Quick start

```bash
pip install -e .
python examples/toy_delayed_copy.py
```

Or:

```python
import torch
from flowcore import FlowConfig, FlowCoreModel

cfg = FlowConfig(
    input_dim=16,
    output_dim=4,
    state_dim=256,
    block_size=8,
    num_ports=8,
    active_ports=2,
    context_dim=32,
)
model = FlowCoreModel(cfg)

x = torch.randn(8, 128, 16)
context = torch.randn(8, 32)
out = model(x, context=context, mode="parallel")
print(out.output.shape)  # [8, 128, 4]
```

The sequential reference should agree with the parallel scan:

```python
p = model(x, context=context, mode="parallel").states
s = model(x, context=context, mode="sequential").states
print((p - s).abs().max())
```

## Implemented through v0.7

- dense/block/diagonal affine operator composition
- differentiable work-efficient Blelloch parallel scan (`O(T)` compositions, `O(log T)` depth)
- Hillis-Steele scan retained as a reference backend
- opt-in `torch.compile` scan wrapper
- sequential reference recurrence
- block-local trainable long-term substrate `W`
- dynamic Route and Hold controls
- top-k normalized multi-port injection `B_k`
- disjoint candidate input basins
- one-pass exact FlowCore model
- predict-correct state-conditioned refinement model
- Route/Hold policy-anchor buffer
- correctness and gradient-equivalence tests
- delayed-memory example and configurable scan microbenchmark
- long-horizon stability diagnostics and v0.2 kernel report
- exact spectral-Kronecker cross-module routing backend
- shared-basis global communication with elementwise modal composition
- end-to-end `substrate_kind="spectral_kronecker"` model option
- directional circulant/FFT cross-module routing backend
- exact frequency-domain composition with cyclic transport\n- matched physical cross-module transport benchmark\n- equal-size destination-to-Route policy comparison across all three substrates\n- direct delivery/leakage/persistence/runtime metrics
- simultaneous multi-source permutation stress benchmark
- analytic projection-error ceilings for block/spectral/circulant routing
- exact monomial permutation x gain operator family
- work-efficient parallel scan for arbitrary one-to-one module routing

## v0.2 parallelization status

FlowCore now uses a work-efficient Blelloch scan by default.  For power-of-two
sequence length `T`, the tree performs `3T-2` affine compositions, versus the
`O(T log T)` work of the v0.1 Hillis-Steele implementation.  The test suite
checks state and gradient equivalence against serial recurrence.

`torch.compile` is available through `make_compiled_scan()` and the benchmark
script can compare sequential, Hillis-Steele, Blelloch and compiled Blelloch
execution.  CPU smoke results are documented in
[`docs/V0_2_REPORT.md`](docs/V0_2_REPORT.md); CUDA results are still required
before making general performance claims.

```bash
python benchmarks/benchmark_scan.py --device auto --lengths 128,512,2048
python experiments/stability_sweep.py --lengths 128,512,2048
```

## v0.3 cross-module routing

The original block backend is exactly scan-friendly but has no communication
between blocks inside a scan pass.  v0.3 adds an optional structured backend:

```python
cfg = FlowConfig(
    input_dim=16,
    output_dim=4,
    state_dim=256,
    block_size=8,
    num_ports=8,
    context_dim=32,
    substrate_kind="spectral_kronecker",
)
model = FlowCoreModel(cfg)
```

It represents state as modules x local-state and uses

```text
H[t+1] = R[t] H[t] L[t]^T + B[t]
R[t]   = Q diag(g[t]) Q^T
```

All routing matrices share the orthogonal basis `Q`, so composition is exact:

```text
g[2:1] = g[2] * g[1]
L[2:1] = L[2] @ L[1]
```

This creates real cross-module signal transfer while retaining an associative
operator family.  The trade-off is expressiveness: v0.3 routing is structured
and simultaneously diagonalizable, not an arbitrary directed sparse graph.

See [`docs/V0_3_REPORT.md`](docs/V0_3_REPORT.md) and run:

```bash
python experiments/cross_module_transfer.py
```

## v0.4 directional FFT routing

v0.4 adds a second exact cross-module backend:

```python
cfg = FlowConfig(
    input_dim=16,
    output_dim=4,
    state_dim=256,
    block_size=8,
    num_ports=8,
    context_dim=32,
    substrate_kind="circulant_kronecker",
)
```

Module routing is represented by a real circular kernel `k[t]`.  The runtime
uses its FFT spectrum:

```text
R[t] x = IFFT( FFT(k[t]) * FFT(x) )
```

and composition is exact:

```text
spectrum[2:1] = spectrum[2] * spectrum[1]
```

Unlike the v0.3 shared-orthogonal-basis backend, a circulant kernel can encode
directional cyclic movement between modules.  The remaining restriction is
translation structure: every module uses the same relative routing offsets.

See [`docs/V0_4_REPORT.md`](docs/V0_4_REPORT.md).

## v0.5 matched substrate benchmark

v0.5 adds a common physical transport task.  A payload enters only module 0 and
the requested destination must contain that payload after one routing event.
There is no learned readout that can bypass the substrate.

By default, all three substrates are calibrated to identity-like local dynamics
and frozen; each receives the same `M x M` destination-to-Route policy.

```bash
python experiments/compare_substrates.py \
  --steps 400 \
  --seeds 0,1,2,3,4 \
  --modules 8 \
  --local-dim 2 \
  --long-time 64
```

The JSON output reports target MSE, non-target leakage, physical delivery
fraction, persistence, parallel/sequential agreement and runtime.  Use
`--train-substrate` for the joint-learning regime.

A three-seed CI reference profile (4 modules, 1D local state, 120 route-policy
updates, frozen substrates) produced:

| backend | long MSE | target energy delivery |
| --- | ---: | ---: |
| block | 0.97018 | 0.00% |
| spectral-Kronecker | 0.55690 | 12.56% |
| circulant-Kronecker | 0.00320 | 99.88% |

This is a deliberately narrow transport benchmark, not a general sequence-model
ranking.  At the short T=16 CPU profile, the parallel scan was also slower than
the sequential recurrence for all three backends.

See [`docs/V0_5_REPORT.md`](docs/V0_5_REPORT.md).

## v0.6 arbitrary routing stress test

v0.6 replaces the single-source transport assumption with simultaneous payloads
in every module.  One dynamic operator must implement an entire permutation.

Two task families are compared:

- cyclic shifts, which are an exact positive control for circulant routing;
- non-cyclic random permutations, which deliberately violate translation
  structure.

The experiment also computes the best analytic projection of every target
permutation into the current block, shared-basis spectral and circulant matrix
families.  This separates optimization failure from representation limits.

```bash
python experiments/arbitrary_routing_stress.py \
  --steps 250 \
  --seeds 0,1,2 \
  --modules 6 \
  --local-dim 3 \
  --tasks 4
```

A three-seed CI reference run confirmed the intended stress pattern:

| family | backend | MSE | slot accuracy | analytic error |
| --- | --- | ---: | ---: | ---: |
| cyclic | circulant | 0.00575 | 99.09% | 0.0000 |
| arbitrary | circulant | 0.65722 | 27.47% | 0.81384 |

The arbitrary failure tracks a large analytic projection error, so the next
operator family must increase representational freedom rather than only tuning
optimization.

See [`docs/V0_6_REPORT.md`](docs/V0_6_REPORT.md).

## v0.7 monomial routing primitive

v0.7 adds an exact algebraic primitive for arbitrary one-to-one routing:

```text
output[i] = gain[i] * input[permutation[i]]
```

Permutation/gain operators are closed under composition, so arbitrary
permutations can participate in the same work-efficient temporal scan without
materializing dense module matrices.

```bash
python experiments/monomial_oracle.py
```

This is deliberately an oracle-level substrate primitive, not yet a normal
`FlowCoreModel` backend.  The unresolved problem is how a learned controller
should produce discrete permutations while preserving exact execution.

The CI oracle on four non-cyclic 6-module permutations produced exactly:

```text
MSE             0
relative MSE    0
slot accuracy   100%
parallel error  0
```

This proves the algebraic substrate can express the v0.6 failure cases exactly.
It does **not** prove that a neural controller can learn the required discrete
permutations.

See [`docs/V0_7_REPORT.md`](docs/V0_7_REPORT.md).

## Deliberate limitations

FlowCore does **not** claim that a dynamic substrate automatically yields
AGI, emergent brain regions, or compute savings.  In particular:

1. exact affine scan requires a structured operator family;
2. block-local scan does not yet implement arbitrary dynamic cross-block edges;
3. top-k input ports are sparse semantically, but custom sparse execution kernels
   are future work;
4. a parallel scan can do more total arithmetic than a serial recurrence, so
   speedups are hardware- and shape-dependent;
5. state-conditioned routing is handled by a small number of global refinement
   passes rather than a fully recursive controller.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design and
[`docs/ROADMAP.md`](docs/ROADMAP.md) for the experimental path.

## Tests

```bash
pytest
```

The test suite verifies both state outputs and gradients of the parallel scan
against the sequential recurrence.

## License

MIT
