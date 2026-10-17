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
