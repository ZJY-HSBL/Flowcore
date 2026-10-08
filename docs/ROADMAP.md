# Research roadmap

## Phase 0 — mathematical kernel (implemented in v0.1)

- exact affine operator composition
- sequential-vs-parallel equivalence tests
- differentiable block parallel scan
- multi-port candidate-basin injection
- Route/Hold controller
- predict-correct state-conditioned refinement
- policy-anchor memory primitive

## Phase 1 — validate the parallelization hypothesis (v0.2 in progress)

Implemented:

- work-efficient Blelloch affine scan with O(T) operator-composition work
- Hillis-Steele reference scan
- opt-in `torch.compile` scan wrapper
- configurable CPU/CUDA benchmark harness
- long-horizon numerical-stability sweep
- dense, block and diagonal equivalence tests

Still required:

- systematic CUDA sweep over sequence length, block size and batch size
- compiler/kernel profiling on modern GPUs
- custom/Triton associative-scan kernel if PyTorch graph overhead is material
- mixed-precision stability characterization
- end-to-end FlowCore throughput comparison, not just scan-kernel timing

## Phase 2 — input topology experiments

- single random port
- central/high-controllability port placement
- disjoint multi-port injection
- overlapping multi-port injection
- fixed injection-energy budget
- measure controllability proxies and state participation ratio

## Phase 3 — structured cross-module routing (v0.3 started)

Implemented:

- exact spectral-Kronecker cross-module operator family
- shared orthogonal mixing basis with dynamic modal Route/Hold gains
- work-efficient parallel scan for the closed cross-module family
- cross-module transfer and gradient-equivalence tests
- end-to-end spectral backend in FlowCoreModel

Next:

- circulant/FFT routing for directional but still closed global communication
- products of multiple closed factors for higher expressiveness
- hierarchical chunk-level sparse exchange
- compare routing expressiveness vs scan cost and numerical stability
- retain low-rank/block-sparse approximations as experimental baselines

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
