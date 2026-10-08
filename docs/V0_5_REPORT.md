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

## 6. Interpretation rule

v0.5 should not be used to claim that one backend is universally superior.

A strong result only supports the narrower statement that a substrate is better
suited to this controlled transport problem under the matched policy/training
budget.  Real sequence modeling still requires later high-dimensional tasks.
