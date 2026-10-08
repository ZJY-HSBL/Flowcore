# FlowCore v0.3 — exact cross-module routing

v0.3 addresses the main structural limitation of the v0.1/v0.2 substrate:
block-local affine scans could propagate information through time, but modules
could not communicate inside an exact scan pass.

## 1. Closed cross-module operator family

Represent state as a matrix

\[
H_t \in \mathbb{R}^{M \times d},
\]

where \(M\) is the number of modules and \(d\) is the local state width.

v0.3 uses

\[
H_{t+1}=R_t H_t L_t^\top + B_t,
\]

with

\[
R_t = Q\,\mathrm{diag}(g_t)\,Q^\top.
\]

\(Q\) is a shared orthogonal mixing basis.  The controller changes the modal
gain vector \(g_t\) at every time step.

The critical property is

\[
R(g_2)R(g_1)
=
Q\,\mathrm{diag}(g_2 \odot g_1)\,Q^\top.
\]

Therefore two affine transitions compose as

\[
g_{21}=g_2\odot g_1,
\]

\[
L_{21}=L_2L_1,
\]

\[
B_{21}=R(g_2)B_1L_2^\top+B_2.
\]

The representation is exactly closed under composition, so the work-efficient
Blelloch scan remains valid.

## 2. Why this creates cross-module communication

When modal gains are not all equal, the physical routing matrix

\[
Q\,\mathrm{diag}(g_t)\,Q^\top
\]

is generally dense.  A state injected into one physical module can therefore
appear in other modules at the next transition.

Unlike a naive dense route matrix, the scan does not need to multiply two
\(M\times M\) routing matrices during every tree composition.  Module-factor
composition is elementwise in modal space.

## 3. Route/Hold interpretation

For the spectral backend, the existing controller outputs are interpreted as
modal controls rather than physical per-module gates.

Let \(r_t\) be Route, \(a_t\) be Hold/update-rate, and \(s\) be a slow
long-term modal strength.  The effective modal factor is

\[
g_t=(1-a_t)+a_t(r_t\odot s).
\]

Thus low update-rate tends toward identity/persistence, while high update-rate
allows the routed long-term dynamics to act.

The local factor is shared across modules:

\[
L_t=(1-\bar a_t)I+\bar a_tW_{local},
\]

where \(\bar a_t\) is the mean update-rate.  This shared local factor is a
deliberate closure constraint.

## 4. Verification

The v0.3 tests cover:

- structured apply vs explicitly materialized physical routing matrix;
- exact two-operator composition;
- parallel vs sequential state equivalence;
- parallel vs sequential gradient equivalence;
- cross-module signal transfer from a single injected module;
- end-to-end FlowCoreModel forward/backward with the new backend.

A standalone mathematical prototype of the new operator family was also checked
before the GitHub implementation: state and gradient agreement were within
float32 tolerance.

## 5. What v0.3 does not solve

This is not arbitrary graph routing.

All module matrices share the same basis \(Q\), so the current family is
simultaneously diagonalizable.  In particular, the real orthogonal construction
produces a constrained, effectively symmetric form of global communication.

That restriction is the price paid for:

- exact closure;
- compact operator composition;
- logarithmic scan dependency depth;
- no dense \(M\times M\) matrix-matrix composition in the scan tree.

The next research problem is to increase routing expressiveness without losing
those properties.  Candidate families include circulant/FFT routing, products of
multiple closed factors, and hierarchical chunk-level sparse exchange.
