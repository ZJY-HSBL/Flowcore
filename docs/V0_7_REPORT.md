# FlowCore v0.7 — exact monomial routing primitive

v0.6 established a concrete representational gap: circulant routing is exact for
cyclic shifts but cannot represent generic simultaneous permutations.  v0.7
implements the smallest exact family that closes that gap without reverting to
dense module matrices.

## 1. Monomial module routing

A monomial matrix has exactly one non-zero entry in every row and column.  It can
be represented by a permutation p and a gain vector g.

FlowCore uses the convention

\[
y_i = g_i\,x_{p_i}.
\]

With a shared local transform L and affine bias B:

\[
H' = M(p,g) H L^T + B.
\]

Every permutation matrix is a special case with g=1.

## 2. Exact closure

Suppose the earlier operator is (p1,g1) and the later operator is (p2,g2).

Then

\[
p_{21}[i] = p_1[p_2[i]],
\]

and

\[
g_{21}[i] = g_2[i] g_1[p_2[i]].
\]

The local and bias terms compose as before:

\[
L_{21}=L_2L_1,
\]

\[
B_{21}=M(p_2,g_2)B_1L_2^T+B_2.
\]

The family is therefore associative and exactly closed.  Module-factor
composition is O(M): permutation gather plus gain gather/multiply.

## 3. Parallel scan

v0.7 implements a work-efficient Blelloch scan over:

- discrete permutation;
- continuous gain;
- shared local transform;
- affine bias.

The permutation itself has no gradient, while gain/local/bias remain fully
differentiable.

Tests compare sequential and parallel state trajectories and gradients.

## 4. Arbitrary-routing oracle

`experiments/monomial_oracle.py` reuses the non-cyclic random permutations from
v0.6.

The experiment directly supplies the target discrete permutation at the routing
step, with identity permutations before/after.  This is intentionally an oracle:
it tests the substrate's algebraic capacity, not whether a neural controller can
discover the permutation.

Expected result:

\[
MSE \approx 0,
\]

\[
slot\ accuracy = 1.
\]

If this succeeds while v0.6's continuous structured families fail, the
representational question and the control question become cleanly separated.

## 5. Important limitation: discrete control

v0.7 is not yet a `FlowCoreModel` backend.

The current Route head emits continuous values.  A monomial operator needs a
discrete permutation.  Replacing that with an arbitrary soft matrix would destroy
the exact monomial representation and, in general, its O(M) composition rule.

Potential controller directions include:

- discrete permutation codebooks;
- Gumbel/Sinkhorn training with hard projection at execution;
- slow structural permutation plus fast continuous gains;
- hierarchical small permutation groups;
- reinforcement/straight-through routing.

Each choice changes the learning problem and must be tested separately.

## 6. Hold interaction

Continuous residual Hold of the form

\[
(1-a)I + aM
\]

is generally a sum of two monomial matrices, not one monomial matrix.  Therefore
v0.7 does not silently claim that the old continuous Hold rule remains closed.

Exact persistence is instead represented by the identity permutation with unit
gain.  This exposes a real design trade-off between continuous time-scale
interpolation and exact arbitrary discrete routing.

That trade-off is now an explicit research target rather than an implementation
detail.


## 7. CI oracle result

GitHub Actions run 37798579002 executed:

```text
modules=6
local_dim=3
tasks=4 arbitrary non-cyclic permutations
batch=256
time=16
CPU
```

The monomial oracle produced:

```text
mse=0.00000000
relative_mse=0.00000000
slot_accuracy=1.000000
parallel_error=0.000e+00
```

The Python 3.11 test job passed 31 tests, including:

- direct monomial apply vs explicit gather;
- two-operator exact composition;
- sequential vs parallel state equivalence;
- sequential vs parallel gradient equivalence for gain/local/bias;
- exact arbitrary-permutation oracle routing.

This establishes the algebraic point cleanly: arbitrary one-to-one routing is
compatible with exact associative temporal scan and does not require dense
module-matrix composition.

It does not establish an end-to-end learning result because the discrete
permutation is supplied by the experiment.

## 8. New research boundary

After v0.6 and v0.7, the main bottleneck has moved.

Before v0.7 the question was:

> Can an exact scan-friendly operator family represent arbitrary module routing?

For one-to-one routing, the answer is now yes.

The new question is:

> Can a controller learn and select the right discrete permutation efficiently,
> while retaining stable gradients, sparse execution and useful Hold semantics?

That controller problem should be attacked before expanding the operator family
again.
