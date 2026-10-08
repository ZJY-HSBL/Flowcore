# FlowCore v0.4 — directional circulant/FFT routing

v0.3 introduced exact cross-module communication through a shared orthogonal
basis.  That family is compact and closed, but its real symmetric routing form
cannot express arbitrary directionality.

v0.4 adds a second exact family based on circulant routing.

## 1. Circulant module operator

Let a real kernel be

\[
k_t \in \mathbb{R}^{M}.
\]

It defines a circular convolution across the module axis.  In the frequency
domain,

\[
\widehat H = \mathrm{FFT}(H),
\]

\[
R(k_t)H = \mathrm{IFFT}(\mathrm{FFT}(k_t)\odot\widehat H).
\]

The full state transition remains separable:

\[
H_{t+1}=R(k_t)H_tL_t^\top+B_t.
\]

## 2. Exact closure

Circulant matrices are closed under multiplication.  In frequency space this is
especially simple:

\[
s_{21}=s_2\odot s_1,
\]

where \(s_t=\mathrm{FFT}(k_t)\).

The local factor and bias compose as

\[
L_{21}=L_2L_1,
\]

\[
B_{21}=R(s_2)B_1L_2^\top+B_2.
\]

Therefore the same work-efficient Blelloch tree can compute the full temporal
prefix exactly.

## 3. Directional flow

A kernel concentrated at offset +1 implements a cyclic shift:

\[
0\rightarrow1,\quad1\rightarrow2,\quad\ldots
\]

so the family can represent directional transport, unlike the symmetric
shared-orthogonal-basis backend.

The controller supplies a non-negative route vector.  FlowCore combines it with
a slow learned kernel prior, normalizes it, and blends the result with the
identity delta kernel according to mean Hold/update-rate.

The normalized non-negative kernel also gives a useful stability property:
the module routing operator has spectral radius no larger than one.

## 4. Complexity

No dense module-by-module routing matrix is materialized.

Per application, module communication costs FFT/IFFT work, approximately

\[
O(M\log M)
\]

per local channel.  During scan-tree composition, routing spectra compose by
elementwise complex multiplication.

This does not prove a wall-clock advantage over dense or block routing; it gives
a structured family with a favorable asymptotic representation.

## 5. Verification

Tests cover:

- FFT routing vs explicit circular shift/convolution;
- sequential vs parallel state equivalence;
- sequential vs parallel gradient equivalence;
- directional module transfer;
- end-to-end FlowCoreModel forward/backward.

## 6. Remaining restriction

Circulant routing is directional but translation structured: the same relative
offset rule is reused at every module.

The next expressiveness step should therefore avoid jumping immediately to an
arbitrary dense graph.  More promising candidates are:

- learnable permutations/module layouts around circulant factors;
- products of two or more closed routing factors;
- hierarchical circulant groups;
- occasional sparse exchange between exact scan segments.

The research objective remains the same: maximize routing expressiveness while
preserving exact or controlled-complexity temporal composition.
