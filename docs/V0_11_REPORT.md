# FlowCore v0.11 — unseen observation-view routing stress

The v0.10 content controller learns to match source and target descriptions
from two fixed random orthogonal observation views. Exact matching on unseen
addresses does **not** imply invariance to previously unseen view transforms.

## Hypothesis and design

Train `MatchingPermutationController` on source/target contexts generated
under fixed, different orthogonal matrices `A_s` and `A_t`. Evaluate it under
both the trained view pair and a newly generated held-out view pair.

Compare with a zero-parameter analytical baseline:

1. Calculate pairwise Euclidean distances within each context set.
2. Normalize the distance matrix by the set's RMS distance.
3. Sort each node's vector of distances to all other nodes.
4. Match these sorted distance fingerprints across source and target sets using
   exact one-to-one assignment.

Orthogonal transformations and translations preserve pairwise distances, and
the optional scale normalization also removes a common global scale factor.
This gives an interpretable, rigid-transform-invariant matching baseline, but
not one guaranteed to distinguish symmetric or degenerate configurations.

## Conditions

- All tasks are synthetic and use six independent random latent descriptors.
- The source and target address sets contain the same points in a random order,
  observed under different orthogonal projections.
- The target permutation and the latent point set are regenerated per sample.
- The trained controller uses only reconstruction loss on soft Sinkhorn routing.
- Noise levels: 0.03, 0.06, 0.12, 0.20.
- Two view conditions: trained pair and held-out pair (new for each seed/run).

This baseline has **zero learned parameters**, but still uses dense O(M²)
distance construction and the small-M exponential-time hard assignment solver.

## Metrics

Report slot accuracy, exact full-permutation accuracy, and hard payload MSE for
each controller and view condition, with three random seeds.

The most informative comparison is the learned controller's decline on unseen
observation views versus any cross-view robustness retained by the invariant
distance-fingerprint method. A failure of the learned controller here would
identify a generalization limitation, not a failure of Monomial associative
composition.

## Reproduction

```bash
python experiments/unseen_view_stress.py \
  --steps 400 --seeds 0,1,2 \
  --noises 0.03,0.06,0.12,0.20 \
  --batch 128 --eval-batch 256 \
  --modules 6 --context-dim 6 --rank 6
```

An explicit future task is unseen nonlinear observation transforms and genuine
semantic routing, where pairwise Euclidean invariance need not apply.

## Verified CI results (October 17, 2026)

[Run 39002837720](https://github.com/ZJY-HSBL/Flowcore/actions/runs/39002837720)
completed successfully across Python 3.10/3.11/3.12 (44 tests).

Three-seed mean slot accuracy:

| Noise | Fixed-view learner / trained views | Fixed-view learner / unseen views | Invariant matcher / unseen views |
| ---: | ---: | ---: | ---: |
| 0.03 | 100.00% | 23.20% | 100.00% |
| 0.06 | 100.00% | 22.31% | 99.74% |
| 0.12 | 100.00% | 21.85% | 95.20% |
| 0.20 | 99.91% | 20.51% | 81.68% |

The unseen-view whole-permutation exact accuracy of the invariant matcher
falls from **100%** at noise 0.03 to **57.55%** at noise 0.20.

This experiment distinguishes two different notions of generalization:
fresh permutations under previously learned representations (v0.10), versus
fresh observation-view transforms (v0.11). The learned fixed-view linear
projections fail the second, while a distance-based analytical invariant
matches the same point set across different orthogonal views.

The analytical controller assumes a common complete latent point set and
a distance-preserving observation transform; its success cannot be extrapolated
to arbitrary nonlinear encoders or semantic matching.

Per-seed machine-readable output: the workflow artifact `v11-ci-reference`.
