# Research roadmap

## Phase 0 — mathematical kernel (implemented in v0.1)

- exact affine operator composition
- sequential-vs-parallel equivalence tests
- differentiable block parallel scan
- multi-port candidate-basin injection
- Route/Hold controller
- predict-correct state-conditioned refinement
- policy-anchor memory primitive

## Phase 1 — validate the parallelization hypothesis

- CUDA benchmark over sequence length, block size, batch size
- `torch.compile` and custom associative-scan kernels
- numerical-stability tests for long horizons
- compare diagonal, block, dense-reference operator families

## Phase 2 — input topology experiments

- single random port
- central/high-controllability port placement
- disjoint multi-port injection
- overlapping multi-port injection
- fixed injection-energy budget
- measure controllability proxies and state participation ratio

## Phase 3 — structured cross-module routing

Preserve a closed/composable operator family while adding richer communication:

- low-rank global coupling
- block-sparse fixed graph with dynamic edge gains
- chunk-level sparse exchange between exact scan segments
- hierarchical module scan

## Phase 4 — policy memory and continual learning

- raw sample replay vs Route/Hold policy anchors
- equal-byte memory budgets
- controller drift and W drift analysis
- gradient-conflict measurements
- consolidation of repeated policy anchors into long-term W

## Phase 5 — true sparse execution

- active-module gather/scatter
- block sparse kernels
- capacity N much larger than active K
- report real FLOPs and wall-clock compute, not only state sparsity

## Phase 6 — structural plasticity

- usage/utility scores for modules and edges
- prune low-value structure
- grow capacity under persistent load
- fixed average resource-budget comparisons

## Phase 7 — real high-dimensional tasks

- sequence modeling
- continuous sensor streams
- multimodal encoders
- robotics/control
- compare against GRU/LSTM, SSM/Mamba-like baselines, Transformers and MoE

## Phase 8 — cognitive-system integration

- global workspace summary
- episodic + policy + structural memory
- world model
- planner/action loop
- offline consolidation / sleep-like reorganization
