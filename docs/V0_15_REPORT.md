# FlowCore v0.15 — Graph-consistent matching and selective confidence

## Motivation and hypothesis

v0.12–v0.14 kept a sorted-distance-quantile representation and changed training
losses. Their eight learned weights did not reliably outperform analytic
geometry at higher noise. This experiment instead changes the representation
available to the matching decision: a full pairwise distance graph.

We hypothesize that preserving relationships **between all matched pairs**
can correct some ambiguous node-fingerprint assignments. We also test whether
the distance to a neighboring competing full route can serve as a useful
ranking score for deciding when to trust the prediction.

## Algorithm

For each source/target point set, compute within-set Euclidean distances
and normalize by its pairwise RMS. This removes independent rigid transforms,
translation and positive global scale.

Compute the original v0.11 sorted-distance-fingerprint assignment. Then
search for an improving pair swap in the one-to-one mapping with the objective:

```text
E(p) = mean_{i != j} w(i,j) * (Ds[p[i],p[j]] - Dt[i,j])²
```

Unweighted graph refinement sets w=1. Short-edge-weighted graph refinement
sets w to the average of two deterministic reliability weights:

```text
reliability(d) = d² / (d² + 0.15²)
```

The 0.15 reliability floor is fixed across every test noise level and
is **not** calculated from the experiment's ground-truth noise parameter.
The heuristic downweights short edges that can be especially sensitive to
relative perturbations; its statistical optimality is unproven.

At each of at most five rounds, find the lowest-cost *single pair swap* and
apply it only if cost decreases. The method keeps permutations one-to-one.
It is a local search, not an exact graph-isomorphism or QAP solver.

## Confidence and selective routing

Define the nonnegative local-swap gap g as the cost of the lowest-cost
different one-swap assignment minus E(p), clamped to zero. Define

```text
confidence = g / (g + E(p) + 1e-8)
```

This is a deterministic score in [0,1] but **not a calibrated probability**.
We check its usefulness by retaining the top 50% of examples ranked by
confidence *without using correctness labels to choose a threshold*, then
measuring:

- overall hard slot and full-permutation accuracy;
- hard payload reconstruction MSE;
- mean confidence;
- whole-permutation exact accuracy and slot accuracy at 50% coverage.

This selective evaluation is not a proof of probabilistic calibration or a
production rejection system.

## Matched evaluation

Zero learnable parameters for all three methods. Exactly the same
fresh-generated source/target examples per seed/condition as in v0.12.
Two module sizes M=6 and held-out M=8; three seeds; additive descriptor
noise levels 0.03, 0.12, 0.20 and 0.30; 128 evaluation samples per
seed/condition. Sources and targets are independently transformed by random
orthogonal views, independent translation and positive scale.

The initial sorted-distance matching and its hard exact assignment solver
come from v0.11. Only the graph-refinement procedure and confidence ranking
change.

## Boundaries

- Full pairwise geometric evidence is O(M²), and local swap search is
  polynomial per iteration but not yet optimized for large GPU batches.
- Initial assignment remains the small-M bitmask solver.
- Pairwise geometry can be ambiguous for symmetric point sets; independent
  nonlinear observation encoders do not preserve these distances.
- This algorithm does not itself add nonlinear state-dependent Route/Hold to the
  parallel temporal scan; hard routes remain compatible with the Monomial
  primitive and could be supplied to it separately.
- Improvement on a small synthetic task does not establish generic semantic
  matching or real-world robustness.

## Reproduction

```bash
python experiments/graph_consistency_stress.py \
 --seeds 0,1,2 --modules 6,8 --noises 0.03,0.12,0.20,0.30 \
 --eval-batch 128 --rounds 5 --reliability-floor 0.15 --device cpu
```

Measured results will be appended from the completed GitHub Actions run.

## Verified CI measurements — October 20, 2026

The first CI reference failed due to a reporting-key typo, corrected in
`ec7692f`. The subsequently completed
[CI run 39359619263](https://github.com/ZJY-HSBL/Flowcore/actions/runs/39359619263)
passed across Python 3.10–3.12.

Three-seed hard slot accuracy (%):

| M | Noise | Fingerprint | Graph unweighted | Short-edge weighted |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 0.03 | 100.00% | 100.00% | 100.00% |
| 6 | 0.12 | 94.14% | 97.61% | 97.53% |
| 6 | 0.20 | 78.60% | 88.63% | 88.89% |
| 6 | 0.30 | 61.20% | 72.53% | 72.35% |
| 8 | 0.03 | 99.93% | 100.00% | 100.00% |
| 8 | 0.12 | 94.89% | 99.67% | 99.67% |
| 8 | 0.20 | 75.72% | 92.06% | 91.86% |
| 8 | 0.30 | 54.69% | 67.81% | 67.87% |

At unseen M8/noise 0.20, exact whole-permutation accuracy:
fingerprint 35.68%, unweighted graph 83.85%, shortedge graph 83.59%.
Unweighted graph exactness among top-confidence 50%: 98.44% at M8/noise 0.20,
versus 83.85% over all routes. The selection *rejects half the routes*.
The score is not a calibrated probability, nor is the 2-opt method a global
optimum guarantee. Test noise was not supplied to the controller.

[Per-seed JSON reference artifact](https://github.com/ZJY-HSBL/Flowcore/actions/runs/39359619263/artifacts/12237160797)
