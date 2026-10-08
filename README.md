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

## Implemented through v0.3

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

## Deliberate limitations

FlowCore v0.2 does **not** claim that a dynamic substrate automatically yields
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
