# FlowCore v0.9 — factorized permutation-controller bandwidth sweep

v0.8 demonstrated exact recovery of four arbitrary 6-module permutations from
behavioral reconstruction loss.  The full controller used one learned
`M x M` logit matrix per task, costing 144 parameters for T=4, M=6.

v0.9 asks a narrower question: how much of that controller bandwidth is actually
needed?

## 1. Factorized score model

For task t, output slot i and candidate source slot j, the controller uses

\[
S_{tij}
=
\frac{\langle q_{ti}, k_j\rangle}{\sqrt r},
\]

where

- \(q_{ti}\in\mathbb R^r\) is a task/output-specific query;
- \(k_j\in\mathbb R^r\) is a source key shared across tasks;
- \(r\) is the factorization rank.

The score matrix is passed through the same Sinkhorn relaxation as v0.8 during
training and the same exact maximum-weight assignment at execution.

## 2. Parameter count

The factorized controller contains

\[
TMr + Mr = Mr(T+1)
\]

parameters.

For the reference setting T=4, M=6:

| rank | parameters | fraction of full 144 |
| ---: | ---: | ---: |
| 1 | 30 | 20.8% |
| 2 | 60 | 41.7% |
| 3 | 90 | 62.5% |
| 4 | 120 | 83.3% |

The old continuous route-vector baseline contains 24 values.

## 3. Experimental control

Nothing else changes relative to v0.8:

- same arbitrary non-cyclic permutation bank;
- same behavioral payload-reconstruction objective;
- same Sinkhorn annealing;
- same exact hard assignment;
- same monomial parallel execution;
- same parallel/sequential verification.

This isolates controller rank as the manipulated variable.

## 4. Primary metric

Soft MSE is secondary.

A compressed controller is considered successful only if the hard projected
permutation is correct.  The strongest metric is

```text
task_exact_accuracy
```

which requires every entry of a task's permutation to match.

The sweep also reports:

- hard MSE;
- slot accuracy;
- permutation-entry accuracy;
- parallel/sequential error;
- controller parameter count.

## 5. Reproducible command

```bash
python experiments/factorized_controller_sweep.py \
  --ranks 1,2,3,4 \
  --steps 350 \
  --seeds 0,1,2 \
  --modules 6 \
  --local-dim 3 \
  --tasks 4 \
  --time 16
```

## 6. Interpretation boundary

This experiment still uses explicit task IDs and a memorized task-conditioned
controller.  A low-rank success would show that the required route family can be
encoded more compactly, not that FlowCore can infer new routes from unseen
content or context.

That generalization question should be separated into a later experiment.


## 7. CI reference result

GitHub Actions run 37800868426 executed ranks 1,2,3,4 across seeds 0,1,2 with
300 optimization steps per run.

Every rank and every seed recovered all four arbitrary permutations exactly:

| rank | params | fraction of full | hard MSE | slot accuracy | entry accuracy | task exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 30 | 20.8% | 0 | 100% | 100% | 100% |
| 2 | 60 | 41.7% | 0 | 100% | 100% | 100% |
| 3 | 90 | 62.5% | 0 | 100% | 100% | 100% |
| 4 | 120 | 83.3% | 0 | 100% | 100% | 100% |

The Python 3.11 job passed 37 tests.  Python 3.10 and 3.12 also passed.

The rank-1 result is the important one.  In this particular four-task bank, a
30-parameter query/key factorization is already sufficient, only slightly above
the 24-value continuous route-vector baseline and far below the 144-parameter
full permutation table.

## 8. Consequence

There is no evidence from this benchmark that the full M x M task table is
needed.  More rank scanning would not answer a useful question because the
minimum tested rank already saturates hard accuracy.

The next bottleneck is explicit task identity.  v0.10 should remove task-ID
lookup and require a shared controller to infer the permutation from current
routing content/context, including permutations not stored as task-specific
parameters.
