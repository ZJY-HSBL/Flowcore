# FlowCore v0.14 — Global structured assignment-margin loss

## Research question

v0.13 improved learned high-noise routing using local target/source margin,
but failed to consistently beat zero-parameter geometric matching. A local
hardest-negative constraint need not express **competition between complete
one-to-one mappings**.

v0.14 therefore tests whether whole-permutation structured supervision yields
a better eight-parameter metric.

## Objective

For `M` modules and score matrix `S`, the weight of a permutation `pi` is

```text
score(pi) = (1/M) * sum_i S[i, pi[i]]
```

Given the payload-derived correct permutation `p`, select:

```text
p_bad = argmax_(pi != p) score(pi)
L_global = mean_b relu(0.1 + score(p_bad) - score(p))
```

The argmax is discrete and detached from autograd. Gradient flows through
the selected correct/incorrect score entries.

Exact small-M competitor discovery enumerates `M!` candidates and excludes
the known correct permutation. Complexity: O(B M! M), with a hard limit of
eight modules in the current implementation. Training uses M=6 (720
permutations per sample); the evaluation path uses the existing exact
assignment routine and includes unseen M=8.

This is a deliberately **non-scalable research oracle**, not a production
GPU kernel or a claim that global structured margins improve scan speed.

## Experiment

Keep the v0.13 training protocol fixed: M=6 training modules, M=6/M=8
evaluation, seeds 0/1/2, 350 updates, batch 64, evaluation batch 64,
descriptor dim 6, payload dim 3, eight learned positive metric weights,
fresh orthogonal views, independent source/target translations and scales
and random address/permutation sets per sample.

Five equally parameterized objectives:

- reconstruction (soft Sinkhorn payload MSE);
- local margin (bidirectional hardest-negative rows and columns);
- local hybrid (reconstruction + local margin);
- global margin (correct whole mapping vs strongest wrong whole mapping);
- global hybrid (reconstruction + global margin).

Two strict zero-parameter baselines:

- uniform quantile fingerprint matching;
- full sorted-distance matching.

All objectives use identical generated training minibatches in the same
order and are evaluated on identical held-out instances.

## Supervision limitation

The local and global margins derive permutation indices from paired
input/target payloads. There are **no additional observations**, but
assignments are made explicitly available to the objective, strengthening
supervision compared with the original reconstruction-only method.
A positive result would not show unsupervised semantic correspondence.

## Reproduce

```bash
python experiments/global_assignment_stress.py \
  --steps 350 --seeds 0,1,2 --batch 64 --eval-batch 64 \
  --train-modules 6 --eval-modules 6,8 \
  --eval-noises 0.03,0.12,0.20,0.30 \
  --max-exact-modules 8 --margin 0.1 --margin-weight 1.0
```

Primary metrics: hard slot accuracy, full-permutation exact accuracy,
hard payload MSE, plus per-seed standard deviation. The zero-param
baselines are mandatory comparators. Result tables will be updated after
a confirmed CI reference run.
