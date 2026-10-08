# FlowCore v0.6 — arbitrary routing stress test

v0.5 showed that the circulant backend is extremely effective when one payload
must be moved from a fixed source to a requested destination.  That benchmark is
structurally aligned with cyclic translation: one relative offset is enough.

v0.6 attacks that assumption directly.

## 1. Simultaneous multi-source routing

At t=0 every physical module contains an independent payload:

```text
M0: p0
M1: p1
M2: p2
...
```

At t=1 a single global routing operator must realize a full permutation:

```text
output[i] = input[permutation[i]]
```

All later time steps use Hold=0, so the routed state should persist unchanged.

Because several source payloads move simultaneously, a controller cannot solve
the task merely by choosing one source-to-destination relative shift.

## 2. Positive control vs stress family

The experiment trains on two separate task families.

### Cyclic

Every target permutation is a pure circular shift.  A circulant operator should
represent this family exactly.

### Arbitrary

Target permutations are random, unique and explicitly rejected if they are a
cyclic shift.  These permutations generally cannot be represented by one
circulant routing matrix.

The same task-id -> M route-value table is used for every backend.

## 3. Analytic projection error

Optimization failure is not enough evidence of a structural limitation, so v0.6
also computes the best matrix approximation inside each current operator family.

For block routing, the target permutation is projected onto diagonal matrices.

For circulant routing, the target is projected onto the linear span of cyclic
shift matrices.

For spectral routing, the target is projected onto

\[
Q\operatorname{diag}(g)Q^T
\]

using the same fixed basis Q and clamping modal gain to the current [0,1]
parameter regime.

The reported normalized Frobenius error is

\[
\frac{\|P-\hat P\|_F}{\|P\|_F}.
\]

This provides a representation ceiling independent of SGD.

## 4. Metrics

The benchmark records:

- final value MSE;
- relative MSE normalized by target energy;
- slot accuracy: which original payload is closest to each output module;
- parallel/sequential maximum error;
- analytic projection error.

Slot accuracy is especially useful because it asks whether the operator recovered
the correct routing assignment even when amplitudes are imperfect.

## 5. Reproducible command

```bash
python experiments/arbitrary_routing_stress.py \
  --steps 250 \
  --seeds 0,1,2 \
  --modules 6 \
  --local-dim 3 \
  --tasks 4 \
  --long-time 32
```

## 6. Interpretation

The key diagnostic pattern is not simply which MSE is lowest.

The strongest evidence for a circulant expressiveness limit would be:

1. circulant succeeds on cyclic tasks;
2. circulant fails or degrades strongly on arbitrary tasks;
3. circulant analytic projection error is near zero for cyclic tasks and
   strictly positive for arbitrary tasks;
4. parallel/sequential agreement remains intact.

If that pattern appears, the next operator family should increase spatial
expressiveness rather than changing optimization.


## 7. CI reference result

GitHub Actions run 37797719671 executed the reference experiment with:

```text
steps=100
seeds=0,1,2
modules=6
local_dim=2
tasks=4
train_time=4
long_time=16
frozen substrate
```

Three-seed means:

| family | backend | MSE | relative MSE | slot accuracy | analytic projection error |
| --- | --- | ---: | ---: | ---: | ---: |
| cyclic | block | 0.96907 | 1.00231 | 0.1662 | 1.00000 |
| cyclic | spectral-Kronecker | 0.93165 | 0.96364 | 0.1660 | 0.97731 |
| cyclic | circulant-Kronecker | **0.00575** | **0.00594** | **0.9909** | **0.00000** |
| arbitrary | block | 0.87707 | 0.90733 | 0.2318 | 0.95644 |
| arbitrary | spectral-Kronecker | 0.88564 | 0.91593 | 0.1845 | 0.95723 |
| arbitrary | circulant-Kronecker | **0.65722** | **0.68003** | **0.2747** | **0.81384** |

The cyclic positive control is decisive: circulant routing nearly recovers the
entire permutation, and the analytic projection error is exactly zero.

On the arbitrary family, circulant remains somewhat better than the other
current families but collapses from 99.09% slot accuracy to 27.47%.  More
importantly, its best circulant matrix approximation has normalized Frobenius
error 0.81384.  The failure therefore reflects a representation ceiling, not
only an optimizer failure.

Block and positive-gain shared-basis spectral routing also have very high
analytic errors on the arbitrary family.

All tests passed on Python 3.10, 3.11 and 3.12, and the v0.6 test count is 27.
The benchmark artifact is stored as `v06-ci-reference`.

## 8. Consequence for the next operator family

The next family needs to represent arbitrary one-to-one module routing without
falling back to dense matrix composition.

A natural exact candidate is a monomial matrix:

\[
R_t=P_t\operatorname{diag}(g_t),
\]

or an equivalent permutation/gain convention.

Monomial matrices are closed under multiplication.  They can represent every
permutation exactly, and composition needs only permutation composition plus
gain gathering, both O(M).

The main unresolved issue is control: permutations are discrete, so an exact
monomial substrate is easy to scan but not directly compatible with the current
fully differentiable continuous Route head.  v0.7 should therefore separate
the algebraic substrate question from the discrete-controller question.
