# FlowCore v0.8 — learned discrete permutation controller

v0.7 showed that exact arbitrary one-to-one routing is algebraically compatible
with associative parallel scan, but the permutation was supplied by an oracle.

v0.8 attacks the next boundary: learn the permutation from task loss, then
execute the learned route with the exact monomial substrate.

## 1. Training/execution separation

The controller stores task-conditioned square logits

\[
S_k \in \mathbb{R}^{M\times M}.
\]

During training, the logits are converted to an approximately doubly-stochastic
matrix with Sinkhorn normalization:

\[
\tilde P_k = \operatorname{Sinkhorn}(S_k / \tau).
\]

Random payloads are routed by

\[
\tilde H = \tilde P_k H
\]

and optimized only from payload reconstruction loss.  The target permutation is
not used as a classification label.

Temperature is geometrically annealed from a soft matrix toward a sharp
assignment.

## 2. Exact execution projection

At execution/evaluation, the soft matrix is projected to a genuine one-to-one
assignment by exact maximum-weight matching.

For the current small-M research implementation, matching uses bitmask dynamic
programming:

\[
O(M^2 2^M).
\]

This is intentionally not claimed as the final large-scale routing algorithm.
It gives an exact reference projection for small module counts.

The resulting discrete permutation is then executed through v0.7's monomial
parallel scan.

## 3. Why this preserves the architectural distinction

Training may use a dense differentiable relaxation, but execution does not.

The deployed operator remains:

\[
\text{permutation} + \text{gain} + \text{local transform},
\]

with exact sparse gather semantics and exact associative composition.

This creates a clean separation between:

- differentiable route discovery;
- discrete sparse route execution.

## 4. Parameter cost

The simple task-table controller uses

\[
T M^2
\]

logits for T tasks and M modules.

The old v0.6 continuous route-vector table uses only

\[
T M.
\]

So v0.8 increases controller bandwidth by a factor of M.  This is acceptable for
a proof of learning, but not yet a scalable controller design.

Later work should test low-rank/factorized permutation policies, codebooks or
hierarchical routing.

## 5. Metrics

`experiments/learn_discrete_route.py` reports:

- soft Sinkhorn reconstruction MSE;
- hard monomial execution MSE;
- hard slot-routing accuracy;
- permutation-entry accuracy;
- exact whole-task permutation accuracy;
- parallel/sequential monomial error;
- controller parameter count vs the v0.6 route-vector baseline.

## 6. Reproducible command

```bash
python experiments/learn_discrete_route.py \
  --steps 300 \
  --seeds 0,1,2 \
  --modules 6 \
  --local-dim 3 \
  --tasks 4 \
  --time 16
```

## 7. Interpretation boundary

Success on this benchmark would show that a differentiable relaxation can learn
the discrete route required by the exact monomial substrate from behavioral
loss alone.

It would not yet show:

- scaling to hundreds/thousands of modules;
- route discovery without explicit task identity;
- online state-conditioned permutation routing;
- compatibility with a continuous Hold interpolation;
- efficient GPU assignment projection.

Those are separate next-stage problems.


## 8. CI reference result

GitHub Actions run 37799900510 executed the learned-controller reference with:

```text
steps=250
seeds=0,1,2
modules=6
local_dim=3
tasks=4
time=16
CPU
```

Every seed converged to the same result:

```text
soft_mse          0.000000
hard_mse          0.000000
slot_accuracy     1.0000
entry_accuracy    1.0000
task_exact        1.0000
parallel_error    0.000e+00
```

The 35-test suite passed on the Python 3.11 reference job; Python 3.10 also
passed in the same workflow.

The controller contains 144 learned logits:

\[
T M^2 = 4 \times 6^2 = 144.
\]

The v0.6 route-vector baseline contains only 24:

\[
T M = 4 \times 6 = 24.
\]

So the learned exact routing result currently costs a 6x controller-parameter
increase.

## 9. Revised research question

The v0.8 result removes one uncertainty: a differentiable Sinkhorn relaxation can
discover the correct discrete permutations from behavioral reconstruction loss
alone in this controlled setting.

The next question is no longer whether discrete routing can be learned. It is:

> How much controller bandwidth is actually required to learn the routing
> family?

A rank/factorization sweep should be preferred over immediately designing a
larger controller.
