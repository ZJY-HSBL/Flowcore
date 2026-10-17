# FlowCore v0.12 — Learned view-invariant structural metric

## Motivation

v0.10 correctly matched unseen permutations only when source/target observation
views had been fixed during training. In v0.11, changing those views reduced
hard slot accuracy from almost 100% to roughly 23%, while a deterministic
distance-fingerprint baseline retained view invariance.

v0.12 asks if a controller can *learn which invariant geometric features to
trust* without memorizing a specific observation coordinate system.

## Operator and controller

For each observed point set X containing M module descriptors, form normalized
pairwise distances. For each module, sort its distances to the full set.
Interpolate each sorted M-vector to a fixed K=8 quantile feature vector f_i.

Source and target features are comparable even when each observation uses its
own orthogonal transform, global translation and positive global scale.

The trainable score is

```text
score[target_i, source_j] = -sum_k w[k] * (f_target[i,k]-f_source[j,k])^2
w[k] = softplus(raw[k]) / mean(softplus(raw))
```

Thus only **eight** real-valued weights are trainable. All scores are
transformed into a soft doubly-stochastic matrix with Sinkhorn during training,
and into a strict permutation by maximum-weight assignment at evaluation. Hard
execution remains compatible with the exact Monomial operator of v0.7.

The model's invariance is built into its feature definition. The claim to be
tested is noise-robust metric learning and limited module-count extrapolation,
*not* emergence of invariance from unconstrained optimization.

## Matched controls and genuine holdouts

Three matchers receive exactly the same generated evaluation batches:

1. learned quantile metric, K=8 weights;
2. uniform quantile metric, K=8, **zero trainable parameters**;
3. original v0.11 full sorted-distance metric, **zero parameters**.

Every synthetic training batch generates new Gaussian latent module addresses,
fresh permutations, independent per-sample orthogonal source and target
observation matrices, independent shifts and scales, and additive Gaussian
measurement noise.

Training uses **M=6** modules only, noise sampled uniformly in [0.03, 0.25],
and only soft payload-reconstruction loss—never permutation-label supervision.
Evaluation uses held-out random seeds/data for M=6 and **M=8**, with noise
0.03, 0.12, 0.20 and 0.30.

We report hard slot accuracy, exact whole-permutation accuracy and hard payload
MSE, with three training seeds. Full-distance baseline is especially important:
the learned eight-bin metric may be worse due to interpolation loss.

## Runtime limitations

Hard matching still uses the small-M exponential-time dynamic-programming
assignment reference, and score construction is quadratic in module count.
The experiment does not benchmark end-to-end GPU efficiency.

Sorted distance sets are only invariant under rigid transformations and
uniform scale, not arbitrary nonlinear encoders. Near-symmetric point clouds
can have ambiguous signatures. M=8 extrapolation is an empirical challenge,
not a mathematical guarantee.

## Reproduce

```bash
python experiments/structural_metric_stress.py \
  --steps 350 --seeds 0,1,2 --batch 64 --eval-batch 64 \
  --train-modules 6 --eval-modules 6,8 \
  --eval-noises 0.03,0.12,0.20,0.30 \
  --device cpu
```

Only completed CI measurements should be appended to this report.
