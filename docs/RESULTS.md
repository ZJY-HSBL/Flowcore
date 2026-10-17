# FlowCore CI reference results and provenance

This page documents the exact source and context for the plots in the project
README. The values are taken from completed GitHub Actions runs in this
repository. They are **not** a jointly trained benchmark suite: versions use
different tasks, state widths, seeds and hyperparameters.

Machine-readable curated values:
[`../results/ci_reference_summary.json`](../results/ci_reference_summary.json).

## v0.5 — Fixed-source physical transport

Source: [Actions run 37742036277](https://github.com/ZJY-HSBL/Flowcore/actions/runs/37742036277),
artifact `v05-ci-reference`.

Parameters: CPU, seeds 0/1/2, steps 120, batch 64, eval-batch 256, runtime-batch
16, 4 modules, local-dim 1, training horizon 4, evaluation horizon 16, frozen
substrates, 4×4 destination-to-route policy.

| Backend | Target-module MSE | Target energy | Parallel latency | Serial/parallel |
| --- | ---: | ---: | ---: | ---: |
| Block | 0.97018 | 0.00% | 0.340 ms | 0.608 |
| Spectral | 0.55690 | 12.56% | 0.710 ms | 0.836 |
| Circulant | 0.00320 | 99.88% | 0.609 ms | 0.749 |

The T=16 CPU result **does not** demonstrate a scan speedup: serial is faster
than parallel for all three tested backends under this short-run setup.

```bash
python experiments/compare_substrates.py \
  --steps 120 --seeds 0,1,2 --batch 64 --eval-batch 256 \
  --runtime-batch 16 --runtime-repeats 3 --modules 4 --local-dim 1 \
  --train-time 4 --long-time 16 --device cpu
```

## v0.6 — Cyclic vs arbitrary simultaneous permutations

Source: [Actions run 37797719671](https://github.com/ZJY-HSBL/Flowcore/actions/runs/37797719671),
artifact `v06-ci-reference`.

CPU, seeds 0/1/2, steps 100, 6 modules, local-dim 2, 4 task permutations,
train horizon 4, eval horizon 16.

| Backend | Cyclic slot accuracy | Arbitrary slot accuracy |
| --- | ---: | ---: |
| Block | 16.62% | 23.18% |
| Spectral | 16.60% | 18.45% |
| Circulant | 99.09% | 27.47% |

Circulant arbitrary-permutation analytic projection error: **0.81384**
(normalized Frobenius). This is a structural representational limit on that
family, rather than evidence that routing is inherently impossible.

```bash
python experiments/arbitrary_routing_stress.py \
  --steps 100 --seeds 0,1,2 --batch 64 --eval-batch 256 \
  --modules 6 --local-dim 2 --tasks 4 \
  --train-time 4 --long-time 16 --device cpu
```

## v0.7–v0.9 — Monomial representation and task-conditioned learning

- [v0.7 monomial oracle](https://github.com/ZJY-HSBL/Flowcore/actions/runs/37798579002):
  four non-cyclic 6-module permutations, hard MSE = 0, slot accuracy = 100%,
  parallel error = 0; permutation **supplied** externally.
- [v0.8 task-conditioned Sinkhorn training](https://github.com/ZJY-HSBL/Flowcore/actions/runs/37799900510):
  3 seeds, all four permutations exactly recovered after hard projection,
  144 task-table parameters.
- [v0.9 factorized controller](https://github.com/ZJY-HSBL/Flowcore/actions/runs/37800868426):
  ranks 1, 2, 3, 4 use 30, 60, 90, 120 parameters respectively, each reaching
  100% hard task exact accuracy for all three seeds on the same fixed task bank.

The factorized result is a capacity observation for the evaluated task bank,
**not** universal rank-1 sufficiency.

```bash
python experiments/factorized_controller_sweep.py \
  --ranks 1,2,3,4 --steps 300 --seeds 0,1,2 \
  --batch 128 --eval-batch 512 --modules 6 --local-dim 3 \
  --tasks 4 --time 16 --device cpu
```

## v0.10 — Content-conditioned matching, no task IDs

Source: [Actions run 37801859799](https://github.com/ZJY-HSBL/Flowcore/actions/runs/37801859799),
artifact `v10-ci-reference`.

CPU, seeds 0/1/2, 450 updates, batch 128, eval-batch 512, 6 modules,
local-dim 3, descriptor-dim 6, embedding rank 6, horizon 16, base context
noise 0.03 and higher noise 0.06.

| Seed | Soft MSE | Hard MSE | Slot accuracy | Permutation exact | Double-noise slot |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.001702 | 0 | 100% | 100% | 100% |
| 1 | 0.002464 | 0 | 100% | 100% | 100% |
| 2 | 0.001807 | 0 | 100% | 100% | 100% |
| Mean | **0.001991** | **0** | **100%** | **100%** | **100%** |

Parallel/sequential maximum absolute error: **0** in this reference experiment.
Controller size: **72 parameters**. CI: **39 tests passed** on Python 3.10,
3.11, 3.12.

The source and target descriptors are observed through two fixed but different
random orthogonal views within each seed/run. Evaluation generates unseen
addresses and permutations, but **not** unseen observation-view transforms.

```bash
python experiments/context_route_generalization.py \
  --steps 450 --seeds 0,1,2 --batch 128 --eval-batch 512 \
  --modules 6 --local-dim 3 --context-dim 6 --rank 6 \
  --time 16 --noise 0.03 --noise-multiplier 2 \
  --device cpu
```

## Reproducibility and limitations

All data shown above are CI reference results for small synthetic tasks.
Hardware, software versions, warmup strategy and seeds can affect performance;
CPU wall-clock timings are particularly environment-sensitive.

The README includes charts to make these controlled results interpretable.
This project has **not** yet demonstrated broad sequence modeling capabilities,
competitive large-scale efficiency, AGI-level behavior, or a production-ready
discrete routing kernel.

## v0.11 — Unseen orthogonal observation views

Source: [Actions run 39002837720](https://github.com/ZJY-HSBL/Flowcore/actions/runs/39002837720),
artifact `v11-ci-reference`.

CPU, seeds 0/1/2, 400 training steps, 6 modules, context dimension 6,
embedding rank 6, evaluation batch 256, training noise 0.03. The learned
v0.10 controller is trained only against its fixed source/target observation
pair. Evaluation compares that same pair against **new orthogonal view matrices**
for each seed, at noise 0.03 / 0.06 / 0.12 / 0.20.

| Noise | Learned/trained slot | Learned/unseen slot | Invariant/unseen slot | Invariant/unseen whole-permutation exact |
| ---: | ---: | ---: | ---: | ---: |
| 0.03 | 100.00% | 23.20% | 100.00% | 100.00% |
| 0.06 | 100.00% | 22.31% | 99.74% | 99.22% |
| 0.12 | 100.00% | 21.85% | 95.20% | 88.15% |
| 0.20 | 99.91% | 20.51% | 81.68% | 57.55% |

The invariant matcher is not trained; it compares sorted distances to all
other modules and assumes a common latent point set. Pairwise Euclidean
distances are preserved by orthogonal transforms. The method fails more often
as measurement noise changes those signatures.

The curves reflect rounded CI console means; the downloadable GitHub Actions
artifact contains the full per-seed JSON.

```bash
python experiments/unseen_view_stress.py \
  --steps 400 --seeds 0,1,2 \
  --noises 0.03,0.06,0.12,0.20 \
  --batch 128 --eval-batch 256 \
  --modules 6 --context-dim 6 --rank 6 --device cpu
```

CI passed **44 tests** across Python 3.10, 3.11 and 3.12.

## v0.12 — Learned distance-quantile metric vs geometric controls

Source: [Actions run 39017452897](https://github.com/ZJY-HSBL/Flowcore/actions/runs/39017452897).
Raw per-seed result artifact: `v12-ci-reference` (JSON). The checked-in
`results/ci_reference_summary.json` stores the four-decimal console
aggregates plus exact provenance.

CPU, 3 seeds, 350 updates/seed, batch=64, eval-batch=64, training module count
6, evaluation module counts 6 and 8, descriptor width=6, payload width=3,
training noise sampled uniformly in [0.03, 0.25], test noise 0.03/0.12/0.20/0.30.

Training and evaluation use new per-sample orthogonal views, independent
translations and scales, new latent Gaussian points and new random
permutations. The learned metric has 8 weights; both analytical baselines have
zero trainable parameters.

**Mean hard slot accuracy (percent):**

| Modules | Noise | Learned metric | Uniform quantile | Full-distance |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 0.03 | 100.00 | 100.00 | 100.00 |
| 6 | 0.12 | 93.23 | 94.53 | 94.27 |
| 6 | 0.20 | 74.22 | 77.86 | 77.95 |
| 6 | 0.30 | 59.46 | 60.94 | 61.63 |
| 8 | 0.03 | 100.00 | 100.00 | 100.00 |
| 8 | 0.12 | 91.15 | 94.79 | 94.79 |
| 8 | 0.20 | 69.27 | 76.63 | 76.63 |
| 8 | 0.30 | 53.84 | 54.36 | 54.36 |

For M=8, noise=0.20, whole-permutation accuracy is **32.81% learned**
versus **44.27%** for both analytical baselines.

This negative result isolates a weakness of the trained diagonal fingerprint
metric under the chosen Sinkhorn reconstruction objective. Do not interpret
it as proof that no learnable geometric controller can be more robust. The
representation's invariance was engineered rather than learned.

```bash
python experiments/structural_metric_stress.py \
  --steps 350 --seeds 0,1,2 --batch 64 --eval-batch 64 \
  --train-modules 6 --eval-modules 6,8 \
  --eval-noises 0.03,0.12,0.20,0.30 --device cpu
```

CI passed **48 tests** on Python 3.10, 3.11 and 3.12.
