# FlowCore v0.5 — matched substrate comparison protocol

v0.5 stops adding operator families and introduces a common benchmark for the
three existing dynamic substrates:

- block-local affine;
- shared-basis spectral-Kronecker;
- directional circulant-Kronecker.

The goal is to test the riverbed itself, not the surrounding encoder/readout.

## 1. Physical transport task

Each sample contains a payload p and a requested destination module d.

The payload is injected only into physical module 0 at t=0:

```text
source module 0: payload
all other modules: zero
```

At t=1 the route policy must move the payload to destination d.  From t>=2,
Hold is zero, so the resulting state should persist.

The target is not a learned readout.  Success is measured directly in the
physical substrate state:

```text
final_state[d] == payload
```

This prevents a decoder from bypassing the routing problem.

## 2. Matched route policy

Every backend receives the same-size policy:

```text
destination id -> M route logits
```

with exactly M x M trainable parameters.

By default, the substrate parameters are calibrated to identity-like local
dynamics and then frozen.  Only the destination-to-route table is trained.

This isolates routing expressiveness.  The command-line flag
`--train-substrate` enables a second regime where both the route policy and
the substrate may adapt.

## 3. Metrics

The benchmark records:

- target-module MSE;
- energy leakage into non-target modules;
- fraction of final state energy in the target module;
- fraction remaining in the original source module;
- parallel vs sequential maximum absolute error;
- parallel runtime;
- sequential runtime;
- sequential/parallel runtime ratio;
- substrate and policy parameter counts.

It also evaluates a horizon longer than the training horizon.  Because Hold is
zero after the transfer event, this tests state persistence without repeatedly
applying the transport route.

## 4. Why this task separates the current operator families

The block backend has no cross-module edge inside a scan pass.  On this task it
cannot physically move source-module information to another module.

The spectral backend can globally mix modules through

\[
Q\,\mathrm{diag}(g)\,Q^T,
\]

but its positive modal-gain parameterization is constrained.

The circulant backend can learn destination-dependent cyclic offsets and is
directional, but remains translation structured.

The benchmark therefore probes a real expressiveness hierarchy rather than only
checking numerical scan equivalence.

## 5. Reproducible command

A multi-seed CPU run:

```bash
python experiments/compare_substrates.py \
  --steps 400 \
  --seeds 0,1,2,3,4 \
  --batch 256 \
  --eval-batch 1024 \
  --modules 8 \
  --local-dim 2 \
  --train-time 4 \
  --long-time 64 \
  --runtime-repeats 20
```

Joint policy+substrate training:

```bash
python experiments/compare_substrates.py \
  --steps 400 \
  --seeds 0,1,2,3,4 \
  --train-substrate
```

Results are written as JSON with per-seed rows and mean/std summaries.

## 6. CI reference result

GitHub Actions run 37742036277 executed a reproducible CPU reference profile:

```text
steps=120
seeds=0,1,2
modules=4
local_dim=1
train_time=4
long_time=16
frozen substrate
same 4 x 4 destination-to-Route policy for every backend
```

The three-seed means were:

| backend | long target MSE | delivery fraction | leakage MSE | max parallel error | parallel ms | sequential/parallel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| block | 0.97018 | 0.0000 | 0.00076 | 0 | 0.340 | 0.608 |
| spectral-Kronecker | 0.55690 | 0.1256 | 0.18382 | 9.54e-7 | 0.710 | 0.836 |
| circulant-Kronecker | 0.00320 | 0.9988 | 0.00035 | 0 | 0.609 | 0.749 |

Per-seed long-horizon target MSE:

```text
block:                 0.79615, 1.13655, 0.97786
spectral-Kronecker:    0.46811, 0.65818, 0.54443
circulant-Kronecker:   0.00250, 0.00376, 0.00334
```

The controlled result supports three narrow conclusions.

First, the block substrate behaves as expected: because it has no cross-module
operator, target-module delivery is exactly zero on this physical transport
task.

Second, the shared-basis spectral backend can transfer information but is
strongly constrained by the positive modal-gain family used in v0.5.

Third, directional circulant routing is almost exact on the task it is
structurally matched to: about 99.88% of final state energy reaches the requested
module with target MSE around 3.2e-3.

The parallel/sequential errors remain at float32 noise level.  Runtime tells the
opposite story from accuracy: at this very short T=16 CPU profile,
sequential/parallel is below 1 for every backend.  The tree scan is therefore
slower than the simple recurrence here.  v0.5 does not support a speed claim;
the parallel path is expected to matter only at larger horizons/hardware
parallelism.

The full JSON output is uploaded by CI as the `v05-ci-reference` workflow
artifact.

## 7. Interpretation rule

v0.5 should not be used to claim that one backend is universally superior.

A strong result only supports the narrower statement that a substrate is better
suited to this controlled transport problem under the matched policy/training
budget.  Real sequence modeling still requires later high-dimensional tasks.
