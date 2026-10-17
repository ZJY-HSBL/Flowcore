# FlowCore v0.13 — Matching-margin loss under fixed model capacity

v0.12 found that an eight-parameter metric trained by soft Sinkhorn payload
reconstruction achieved worse high-noise accuracy than the zero-parameter
uniform/full-distance alternatives.

v0.13 changes the *training objective*, not the architecture.

## 1. Three equal-capacity learned models

All models use identical distance-quantile fingerprints, eight non-negative
normalized weights, model initialization, optimizer/hyperparameters, minibatch
data stream, training length and evaluation set.

### Reconstruction

```text
S = learned structural matching scores
P_soft = Sinkhorn(S / temperature)
L_rec = mean_squared_error(P_soft @ source_payload, target_payload)
```

This is the existing v0.12 objective.

### Assignment margin

Infer correspondence from the provided payload pair:

```text
positive_source[i] = argmin_j ||target_payload[i] - source_payload[j]||²
```

For each target row and source column, penalize when the positive
correspondence scores lower than the hardest incorrect candidate:

```text
L_row = mean_i relu(margin + max_j!=positive S[i,j] - S[i,positive])
L_col = mean_j relu(margin + max_i!=positive S[i,j] - S[positive,j])
L_margin = (L_row + L_col) / 2
```

The reference margin is 0.1.

### Hybrid

```text
L_hybrid = L_rec + 1.0 * L_margin
```

## 2. Supervision caveat

The margin method does *not* use separately supplied permutation labels;
it reconstructs the correspondence from the *same target/source payload*
training pairs available to reconstruction.

This is nonetheless **stronger explicit assignment supervision** than merely
computing payload-reconstruction MSE. Improvement would mean extracting and
using those implicit correspondences was helpful on this synthetic task,
not that hard matching can be learned from less information.

## 3. Matched experimental setting

Three seeds 0/1/2; 350 updates; batch 64; M=6 training and evaluation;
M=8 held-out evaluation; descriptor dim 6, payload dim 3; noise sampled
uniformly [0.03,0.25] in training and evaluated at
0.03/0.12/0.20/0.30.

Independent orthogonal view matrices, translations, scales, Gaussian module
addresses and permutations are regenerated per sample.

All five evaluation rows (three learned objectives and two zero-parameter
controls) use exactly the same generated test examples per seed/M/noise.

## 4. Metrics and decisive criterion

Primary: hard slot accuracy (target-to-source correct index) and hard
whole-permutation exact accuracy. Also store hard payload MSE, with a per-seed
record and mean/std summaries.

Even if margin beats reconstruction, the **uniform-quantile and full-distance
baselines remain the bar**. A gain over v0.12 alone is not enough.

## 5. Reproduction

```bash
python experiments/route_loss_comparison.py \
  --steps 350 --seeds 0,1,2 \
  --batch 64 --eval-batch 64 \
  --train-modules 6 --eval-modules 6,8 \
  --eval-noises 0.03,0.12,0.20,0.30 \
  --margin 0.1 --margin-weight 1.0 --device cpu
```

The v0.7 Monomial associative operator and the small-M assignment kernel are
unchanged. This test does not establish speedups, large-scale matching,
or semantic route discovery.

## Verified three-seed CI results (October 17, 2026)

[CI run 39018232214](https://github.com/ZJY-HSBL/Flowcore/actions/runs/39018232214)
passed 52 tests across Python 3.10/3.11/3.12.

| Modules | Noise | Reconstruction | Margin | Hybrid | Uniform (0p) | Full distance (0p) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.03 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| 6 | 0.12 | 93.23% | 94.18% | 93.75% | 94.53% | 94.27% |
| 6 | 0.20 | 74.22% | 77.08% | 74.57% | 77.86% | 77.95% |
| 6 | 0.30 | 59.46% | 60.42% | 59.81% | 60.94% | 61.63% |
| 8 | 0.03 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| 8 | 0.12 | 91.15% | 95.18% | 93.16% | 94.79% | 94.79% |
| 8 | 0.20 | 69.27% | 74.48% | 70.44% | 76.63% | 76.63% |
| 8 | 0.30 | 53.84% | 53.26% | 53.19% | 54.36% | 54.36% |

The main learned-objective improvement is at M=8/noise=0.20:
reconstruction 69.27% → margin 74.48%, but the uniform/full-distance
zero-parameter baselines remain at 76.63%.

At M=8/noise=0.12 margin reaches 95.18%, just 0.39 points over the
zero-parameter baselines (94.79%). This narrow conditional result is not
a reliable claim of general superiority given only three seeds, several
tested conditions and no pre-registered significance threshold.

Pure margin generally outperforms the fixed-coefficient hybrid here; a
different hybrid weight was not tested and is not ruled out.

**Scientific interpretation:** Explicitly extracting correspondences from
existing paired payload supervision improves some learned routing metrics
but does not resolve the analytic-baseline gap under more substantial noise.

The raw per-seed experiment is available as the workflow's
`v13-ci-reference` artifact.
