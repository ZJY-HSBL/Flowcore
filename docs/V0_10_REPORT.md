# FlowCore v0.10 — content-conditioned routing without task IDs

v0.8 and v0.9 learned arbitrary discrete permutations, but both used explicit
task identity.  v0.10 removes the lookup-table assumption.

## 1. Fresh route on every sample

For every sample, generate M independent latent routing addresses

\[
u_j \in \mathbb{R}^{d_c}.
\]

The source-side controller observes

\[
s_j=u_j A_s + \epsilon_s,
\]

while the target-side context observes a newly permuted address through another
view:

\[
t_i=u_{p_i}A_t + \epsilon_t.
\]

The matrices \(A_s\) and \(A_t\) are fixed random orthogonal views for one
training run.  The permutation \(p\) and latent addresses \(u\) are regenerated
for every batch.

There is therefore no finite task/permutation table to memorize.

## 2. Shared matching controller

A shared controller learns two linear projections:

\[
k_j = f_s(s_j),
\]

\[
q_i = f_t(t_i).
\]

After normalization,

\[
S_{ij}=q_i^T k_j.
\]

Training uses Sinkhorn normalization of S to route payloads softly.  The only
supervision is payload reconstruction.

At evaluation, the score matrix is projected to an exact one-to-one assignment
and executed by the v0.7 Monomial parallel scan.

## 3. Generalization target

Evaluation uses:

- unseen latent addresses;
- unseen random permutations;
- the same source/target observation relationship;
- a second evaluation at 2x context noise.

The central metric is exact/per-slot hard assignment on these newly generated
routes.

## 4. Parameter scaling

With context dimension d and embedding rank r, the current controller uses two
linear maps:

\[
2dr
\]

parameters.

Unlike the v0.8/v0.9 task tables, this count does not grow with the number of
stored tasks or permutations.

For d=r=6 the reference controller has only 72 parameters.

The score computation is still dense O(M^2 r), so this is a memory/generalization
advance, not yet a large-M compute solution.

## 5. Reproducible command

```bash
python experiments/context_route_generalization.py \
  --steps 500 \
  --seeds 0,1,2 \
  --modules 6 \
  --local-dim 3 \
  --context-dim 6 \
  --rank 6 \
  --noise 0.03 \
  --noise-multiplier 2
```

## 6. Interpretation boundary

Success would establish a stronger result than the task-table experiments:
FlowCore could infer previously unseen discrete module permutations from routing
content and then execute them through an exact associative substrate.

It would still not establish:

- semantic route discovery from natural language or raw sensory input;
- state-conditioned online routing;
- large-module-count assignment efficiency;
- continuous Hold compatibility with Monomial routing;
- generalization to a different source/target observation transformation.

Those remain separate experiments.

## 7. Verified CI result (October 8, 2026)

The [completed v0.10 reference CI run](https://github.com/ZJY-HSBL/Flowcore/actions/runs/37801859799)
passed all jobs (Python 3.10, 3.11 and 3.12, 39 tests).

Reference setting: seeds 0/1/2; 450 training steps per seed; 6 modules; payload
dimension 3; descriptor dimension and embedding rank 6; 512 evaluation samples
per seed; context noise 0.03, doubled-noise evaluation 0.06.

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Soft reconstruction MSE | 0.001702 | 0.002464 | 0.001807 | **0.001991** |
| Hard Monomial MSE | 0 | 0 | 0 | **0** |
| Unseen slot accuracy | 100% | 100% | 100% | **100%** |
| Whole-permutation exact | 100% | 100% | 100% | **100%** |
| 2x noise slot accuracy | 100% | 100% | 100% | **100%** |
| Parallel/serial error | 0 | 0 | 0 | **0** |

The controller contains 72 parameters. The result establishes exact learned
content matching *within the fixed pair of observation views* under these
conditions; it does not establish transfer to new observation transformations.

See the README visual summary and the full [results protocol](RESULTS.md).
