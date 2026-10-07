# FlowCore architecture

FlowCore is an experimental implementation of a **parallel dynamic cognitive
substrate**.  The current library is intentionally narrower than the long-term
vision: it implements the mathematical core that must work before sparse graph
execution, structural plasticity, world models, planners, or multimodal systems
are layered on top.

## 1. State transition as a composable operator

The scan-friendly core uses an affine discrete-time transition

\[
h_{t+1}=A_t h_t+b_t.
\]

An affine operator is represented by \(\mathcal T_t=(A_t,b_t)\).  Composition is

\[
(A_2,b_2)\circ(A_1,b_1)
=(A_2A_1, A_2b_1+b_2).
\]

Composition is associative, so a time sequence can be evaluated by an inclusive
parallel prefix scan rather than a strictly serial Python recurrence.

## 2. Structured operator family

A fully dense \(A_t\) is exact but expensive to compose.  FlowCore therefore uses
small block-diagonal local operators as the default scalable representation.
Each block is a microcircuit; the operator family stays closed under composition.

The current exact scan pass does **not** include arbitrary cross-block edges.
Cross-module communication is planned at macro boundaries or through a future
structured low-rank/sparse operator family.  Preserving closure is a deliberate
constraint, not an accidental omission.

## 3. Inject / Propagate / Persist / Read

The architecture replaces the earlier Write/Route/Hold terminology with four
more primitive operations:

- **Inject**: multi-port matrices \(B_k\) place encoded content into candidate
  state basins.
- **Propagate**: long-term block dynamics \(W\), modulated by Route gates, define
  the local transition operator.
- **Persist**: Hold controls each block's update rate.  Small update rates retain
  old state; large rates accept new dynamics quickly.
- **Read**: a decoder maps persistent state to task outputs.

The model does not require an entrance gate to decide whether information is
"useful" before it is processed.  Candidate information enters bounded basins;
subsequent dynamics determine whether it propagates or persists.

## 4. Multi-port input topology

For encoded input \(z_t\),

\[
I_t=\sum_k s_{k,t}B_k z_t.
\]

The selector can activate only top-k ports.  Port weights are normalized to sum
to one so that using more ports does not simply inject more energy.  A support
mask can make ports disjoint, overlapping, modality-specific, or hand-designed
for controllability studies.

## 5. Predict-correct control

Exact one-pass scan requires controls to be known without recursively depending
on the current state.  `RefinedFlowCoreModel` introduces state-conditioned
control through a fixed number of global refinement rounds:

1. predict Route/Hold/port controls in parallel,
2. scan the full sequence,
3. summarize the resulting state trajectory,
4. update controls in parallel,
5. rescan.

This keeps controller depth proportional to the number of refinement rounds,
not sequence length.

## 6. Long-term target hierarchy

The intended long-term system is layered:

1. encoders + multi-port injection,
2. simple state cells,
3. local microcircuits/modules,
4. parallel dynamic substrate,
5. nonlinear operator controller,
6. global workspace / compressed global state,
7. planner, world model, tools, and action loop.

Different parameters should eventually evolve at different time scales:

\[
\tau_{state}<\tau_{policy}<\tau_W<\tau_{topology}.
\]

## 7. What this repository does **not** claim

- It is not an AGI implementation.
- It has not demonstrated emergent brain-like regions.
- Block sparsity here is an operator representation; it is not yet a custom
  sparse hardware kernel.
- Parallel scan is mathematically parallelizable, but wall-clock speedups depend
  on sequence length, block size, compiler/kernel quality, and hardware.
- The affine/weakly nonlinear substrate is a hypothesis to test, not a claim that
  general intelligence is linear.
