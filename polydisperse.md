# Poly-disperse support for `locate()` / `batch()`

Implementation plan for allowing `trackpy.locate` (and `trackpy.batch`) to detect
particles spanning a **range** of sizes, by passing a `Polydisperse` config object
as the `diameter` argument instead of a single number/tuple.

> **Shorthand:** throughout this doc, `mindiameter` / `maxdiameter` refer to the
> `Polydisperse.min_diameter` / `max_diameter` field *values* (the ends of the size
> range), not to standalone function parameters. See §5 for the actual API.

---

## 1. Goal & design premise

Today `locate(raw_image, diameter, ...)` assumes every feature is ~the same size.
`diameter` sets a single `radius = diameter // 2` that flows through the whole
Crocker–Grier pipeline. We want to support **poly-disperse** samples: a continuum
of sizes, up to ~10× range (e.g. `mindiameter=5`, `maxdiameter=51`).

Design premise:

- **A number/tuple `diameter` is untouched.** Behaviour is byte-for-byte identical
  to today. Zero regression risk for existing users.
- Passing a **`Polydisperse` object** as `diameter` activates "poly mode" (see §5).
  - Bundles `min_diameter`, `max_diameter`, `edge_frac`.
  - Mutual exclusion with monodisperse is **type-encoded** — one `diameter` slot, so
    the invalid combination is inexpressible (no cross-parameter check needed).
  - `min_diameter`/`max_diameter` odd integers, `max >= min`; validated in
    `Polydisperse.__init__`.
  - **Isotropic only** initially (scalar, not per-axis tuple) — see §7.
- Derived quantities in poly mode: `r_min = mindiameter // 2`,
  `r_max = maxdiameter // 2`.

### Why this isn't one parameter

The core insight from tracing the code: `diameter` silently configures **four
physically different things**, plus feeds characterization and uncertainty. In
monodisperse mode they happen to be the same number. Poly mode forces each to be
handled on its own terms:

| Role of `diameter` | Set via | Physical meaning |
|---|---|---|
| Background kernel | `smoothing_size` | boxcar window for background subtraction |
| Peak spacing | `separation` | min distance between distinct maxima |
| Refinement window | `radius` | COM integration + mass/size/ecc/signal |
| Edge margin | `margin` | exclusion zone at image borders |

The plan below addresses each **separately**.

---

## 2. Current code map (where `diameter` is used in `locate`)

File: `trackpy/feature.py`, function `locate` (starts line 204).

| Lines | What happens | Depends on |
|---|---|---|
| 324–328 | `diameter = validate_tuple(...)`, odd check, `radius = diameter//2` | diameter |
| 330–333 | `isotropic` check; forbids `maxsize` when anisotropic | radius |
| 337–338 | default `separation = diameter + 1` | diameter |
| 342–345 | default `smoothing_size = diameter` | diameter |
| 371 | `image = bandpass(raw_image, noise_size, smoothing_size, threshold)` | smoothing_size |
| 391–392 | `margin = max(radius, separation//2 - 1, smoothing_size//2)` | radius, sep, smoothing |
| 396 | `coords = grey_dilation(image, separation, percentile, margin, precise=False)` | separation |
| 399–401 | `refine_com(raw_image, image, radius, coords, ...)` | radius |
| 406–409 | `where_close(refined_coords[pos_columns], separation, mass)` dedup | separation |
| 419–421 | filter `mass > minmass`, `size < maxsize` | — |
| 445–455 | uncertainty `ep`: `measure_noise(image, raw_image, radius)`, `N_binary_mask(radius)`, `_static_error(mass, noise, radius, noise_size)` | radius |

Supporting code:

- `trackpy/preprocessing.py:82` `bandpass(image, lshort, llong, threshold, truncate)`
  = `lowpass(noise_size)` − `boxcar(smoothing_size)`. **Global op — cannot be
  per-feature.** boxcar window must exceed the particle or it subtracts the
  particle's own signal.
- `trackpy/find.py:72` `grey_dilation(image, separation, percentile, margin, precise)`
  — one dilation kernel `size = 2*separation/sqrt(ndim)` (line 113); one maximum
  per kernel-neighborhood; global `percentile` brightness gate (line 107).
- `trackpy/find.py:16` `where_close(pos, separation, intensity)` — rescales
  positions by a **fixed** `separation`, KD-tree pairs within distance 1, drops
  dimmer of each pair. `drop_close` (line 55) wraps it.
- `trackpy/refine/center_of_mass.py:27` `refine_com(...)` →
  `:102` `refine_com_arr(...)`. **One `radius` for all features.** Masks are
  memo-cached by a single `(radius, ndim)` (`trackpy/masks.py:8`); numba/python
  inner loops (`center_of_mass.py:146+`) reuse one fixed-shape mask.
- `trackpy/feature.py:175-188` `estimate_mass` / `estimate_size` — **single-shot**
  (no COM iteration) mass and radius-of-gyration over a window. (Considered for the
  Stage C size estimate but superseded by the curve-of-growth approach.)
- `trackpy/masks.py`: `binary_mask` (8), `N_binary_mask` (22), `r_squared_mask`
  (27) — all `@memo`-cached per `(radius, ndim)`. Building masks for a new radius
  is cheap and cached; no change needed here.

Scope: **`locate` + `batch` only.** `batch` (`feature.py:464`) just forwards
`**kwargs` to `locate`, so it only needs the two new params in its signature +
docstring. `find_link` (`trackpy/linking/find_link.py`) and nD/anisotropic poly
mode are **out of scope**.

---

## 3. Key architectural decision: bucketing is lossless, not a shortcut

The refinement stage (Stage C) uses **bucketing by odd diameter** rather than a
rewritten per-feature-radius COM kernel. This is deliberate and is the crux of the
design, so the reasoning is recorded here:

- A COM mask has an **integer** radius (`binary_mask` only takes integer tuples).
  There is no radius-4.5 mask. So *any* per-feature-radius method must quantize
  each feature's radius to an integer.
- If buckets = **every distinct odd diameter** in `[mindiameter, maxdiameter]`
  (~24 values for a 10× range), each feature lands in the bucket whose radius is
  exactly the integer it would have gotten anyway. **Same masks, same
  neighborhoods, same output numbers as a "true" per-feature method.** No accuracy
  loss.
- A hand-rolled per-feature loop would be **slower** (can't hoist the fixed-shape
  mask out of the loop → breaks the vectorized/numba kernel at
  `center_of_mass.py:146+`; variable neighborhoods defeat numba's contiguous
  assumptions) and **riskier** (rewrites tuned numba code instead of reusing
  `refine_com_arr` per bucket).
- Within a bucket, `refine_com_arr` stays fully vectorized/numba. Across buckets we
  pay ~24 fixed Python call-overheads — negligible vs. per-feature work.

**Sizing (historical note).** The chicken-and-egg (need a size estimate to pick a
window; need a window to estimate size) originally used a cheap **moment bootstrap**
— a single-shot `Rg` in the `r_max` window, mapped to a diameter via `rg_to_diameter`.
That was **superseded**: in dense fields the `r_max` moment window engulfs neighbours,
so small particles measured a huge `Rg` and got the max window (see §7 "Resolved").
Stage C now sizes each feature by a **curve of growth** that reads its own extent
(no `Rg → diameter` mapping). The bucketing rationale above is unchanged.

---

## 4. Per-stage plan

Each stage: *what it does today → why it breaks under polydispersity → the fix.*
Stages A, B, D, E, F, G are single-pass and simple; Stage C is the spine.

### Stage A — Background subtraction (`bandpass` / `smoothing_size`)

- **Today:** `smoothing_size = diameter`; boxcar background window must exceed the
  particle or it subtracts the particle's own signal.
- **Breaks:** `bandpass` is one global convolution — cannot be per-feature.
- **Fix:** `smoothing_size = maxdiameter`. Largest particle dictates the
  constraint; `max` guarantees no particle self-subtracts. Small particles get a
  slightly-too-large window → marginally less background rejection (safe
  direction). `noise_size` (Gaussian lowpass) is size-independent → unchanged.
- **Effort:** trivial. **Risk:** low.

### Stage B — Peak finding (`grey_dilation` / `separation`)

- **Today:** `separation = diameter + 1`; `grey_dilation` flags every pixel where
  `image == dilation` (`find.py:117`). `locate` calls it with `precise=False`
  (`feature.py:396`), so dedup is deferred to `where_close` after refine. The large
  monodisperse `separation` means the post-refine `where_close` merges any plateau.
- **Breaks:** to resolve closely-spaced *small* particles we must shrink
  `separation` to `mindiameter` — which removes the large-`separation` safety net
  monodisperse relied on. Two distinct failure modes appear on large particles:
  1. **Flat-top explosion (the expensive one).** `convert_to_int`
     (`feature.py:376-381`) quantizes to uint8, so a broad smooth top becomes a
     *flat integer plateau* where `image == dilation` across the **whole top** →
     O(diameter²) ≈ thousands of touching maxima for a d≈50 particle, all fed into
     refinement. This is a per-frame blowup, not "a few duplicates".
  2. **Non-touching bumps (the rare one).** Separate noise bumps on a broad top,
     spaced farther than `mindiameter` apart, so *not* a connected component.
- **Fix — split by cost:**
  - Detect at the fine scale: `separation = mindiameter + 1`.
  - **Immediate, size-free — connected-component collapse (= regional-maxima
    seeding).** Label connected components of the maxima mask
    (`scipy.ndimage.label`) and reduce each blob to a single seed (component
    centroid, or its brightest pixel in the float bandpassed image). This is not an
    ad-hoc dedup: `image == dilation` + `label` + one-seed-per-blob is the textbook
    *regional maximum* operator (the plateau-aware upgrade of the naive `==`
    maximum test; same thing skimage's `peak_local_max`/`local_maxima` do
    internally). Kills mode (1) at O(N_pixels), *before* any expensive stage.
    **Safe** against merging distinct particles in the normal case: adjacent maxima
    pixels only occur within one particle's flat top — two real particles are
    separated by a sub-maximum valley (valley pixels aren't maxima) *and* by the
    dilation kernel, so CC-collapse never eats a real neighbor.
  - **Alternatives considered and rejected for this sub-problem:** multi-scale /
    coarse-then-fine detection (assumes a big/small split a continuum lacks; fiddly
    mask-out); detecting on the float image instead of uint8 (attacks the
    quantization cause but doesn't eliminate saturation/noise plateaus, and
    `grey_dilation` coerces to uint8 internally anyway — `find.py:99`); thinning
    (skeleton ≠ point); watershed (heavyweight, unjustified before COM refine).
  - **Deferred, size-aware (Stage E).** Mode (2) — the residual handful of
    non-touching duplicates — needs a measured size to distinguish from two genuine
    close particles, so it stays in Stage E, now operating on a tiny count.
- **Implementation note:** `grey_dilation` returns positions, discarding the mask,
  so the CC-collapse must happen *inside* it (where `maxima` exists). Add an opt-in
  option (e.g. `precise='cc'` or a new bool) so monodisperse behaviour is untouched.
- **Effort:** low–medium. **Risk:** low.
- **Documented limitation:** the brightness gate is a single global `percentile`
  (`find.py:107`). A few big bright particles raise the threshold and can suppress
  faint small ones. Inherent to a one-pass detector; not solved here.

### Stage C — Refinement + characterization — the spine

Replace the single `refine_com` call (poly mode only) with:

1. **Size each detected peak by a curve of growth.** Accumulate the annular (ring)
   mass outward from the peak; the feature's edge is the first radius past the peak
   ring where the ring mass falls below `edge_frac` of the peak-so-far. This reads
   each particle's *own* extent from its radial intensity profile.
   - `assigned_diameter = clamp(2·edge + 1, [min, max])`; always odd.
   - **`edge_frac`** is a `Polydisperse` field (default **0.1**); larger → tighter
     windows. It replaces the earlier `rg_to_diameter` mapping (retired), because
     the growth curve measures the extent directly — no `Rg → diameter` heuristic.
   - **Robust by construction:** because it stops at the first decay of the
     feature's *own* profile, it is not inflated by (a) neighbours farther out
     (the ring only climbs again *past* the boundary) nor (b) duplicate peaks on
     the same particle (both give the same extent). A fixed `r_max` *moment* is
     inflated by both — that was the dense-field failure this fixes.
   - Implementation note: bin patch pixels by integer radius via `np.bincount`;
     **drop pixels beyond `r_max`** (patch corners) rather than folding them into
     the last bin, and scan outward with a running max — otherwise the corners
     pile neighbour signal into `ring[r_max]` and defeat the edge detection.
2. **Bucket** features by `assigned_diameter`:
   - **2D (and any case with ≤ 10 distinct odd diameters in range):** one bucket per
     distinct odd diameter — lossless (mask radii are integers anyway).
   - **3D with a wide range (> 10 distinct odd diameters):** cap at **10 buckets**,
     spaced **geometrically** across `[min, max]`, rounded to odd, snapping each
     feature to the nearest. Bounds the persistent mask cache to ≤ 10 radii; the
     give-up is a small window quantization on 3D features only.
3. **Refine (real pass):** for each bucket, call `refine_com` **once** on that
   subset at the bucket's radius, starting from the detected coordinates.
   Concatenate, **preserving the original coordinate order**.
4. **Emit `diameter` column** = per-feature `assigned_diameter`, alongside `size`.

- **No change to `refine_com*` or `masks.py`.** New logic is `Polydisperse.refine`
  plus `_growth_diameters` / `_bucketed_refine` in `polydisperse.py`.
- **Effort:** medium (bulk of the work). **Risk:** low, now that the sizing is
  content-based (validated: recall & precision 1.00 across the grid — see §9).

#### Stage C — performance vs. monodisperse

Cost model. `N` = feature count, `I` = `max_iterations` (default 10), `m(r)` = mask
pixels (≈ `π r²` 2D / `(4/3)π r³` 3D), `m̄` = population-average mask size.

- **Monodisperse:** `T_mono(r) = N·I·c·m(r)`.
- **Poly:** `T_poly ≈ N·c·[ m(r_max) + I·m̄ ]` — an O(N·m(r_max)) curve-of-growth
  sizing pass (one `bincount` over an `r_max` patch per feature) plus one real
  refine with each feature at its own (mostly smaller) radius.

Comparisons:

- **vs. ideal single-radius pass at average radius:**
  `T_poly/T_mono(r̄) = 1 + m(r_max)/(I·m̄) ≈ 1.2` (I=10, `m(r_max)≈2·m̄`) → **~20%
  overhead**, from the sizing pass. Not 2×: sizing is a single pass and the refine
  uses per-feature masks.
- **vs. naive workaround `diameter=maxdiameter`** (`T_mono(r_max)`):
  `T_poly/T_mono(r_max) = 1/I + m̄/m(r_max) ≈ 0.6` → poly is **~40% faster** *and*
  correct, because small particles no longer pay for a big mask.

**Contingent on Stage B.** These numbers assume `N` ≈ true particle count. Detecting
at `separation = mindiameter` surfaces more maxima, and the flat-top explosion would
inflate `N` by O(d²) per large particle. Stage B's CC-collapse is what bounds `N`, so
it is a *performance* prerequisite for Stage C, not just correctness.

**Memory.** Working set ≈ `O(m(r_max) + N·cols)`, same order as monodisperse (refine
runs one bucket at a time; `diameter` column is negligible). The real delta is the
`@memo` mask cache holding **K distinct radii** instead of 1 (never evicted):

- 2D: sub-MB for `maxdiameter=51`, a few MB at `d=101` — trivial.
- 3D: grows as `d³`; large ranges would reach tens of MB — bounded by the **10-bucket
  cap in 3D** (step 3), so the persistent cache holds at most 10 radii's masks.

**`batch()` amortization:** the cache is process-global and persistent, so masks are
built once and reused across all frames — poly does not rebuild masks per frame.

### Stage D — Edge margin

- **Today:** `margin = max(radius, separation//2 - 1, smoothing_size//2)`.
- **Breaks:** margin is applied inside `grey_dilation`, before sizes are known —
  can't be per-feature.
- **Fix:** conservative global margin from the largest scale:
  `max(r_max, (mindiameter+1)//2 - 1, maxdiameter//2)`. In the `preprocess=True`
  case (the only case in scope) this is pinned at `r_max` by the bandpass
  edge-artifact term (`smoothing_size//2 = maxdiameter//2`) regardless — the
  preprocessed border ring is unreliable for features of all sizes, so a per-feature
  margin would buy nothing. Excludes a slightly wider border (loses a few small
  particles near edges) rather than characterizing on clipped/corrupted data.
- **Effort:** trivial. **Risk:** low.
- **Out of scope:** `preprocess=False` (where the artifact term vanishes and a
  per-feature post-refine edge check would recover small near-edge particles).

### Stage E — Deduplication (`where_close`) — second real change

- **Today:** `where_close` rescales by a **fixed** `separation`, KD-tree pairs
  within distance 1, drops dimmer.
- **Breaks:** fixed separation can't remove big-particle duplicates from Stage B
  without also deleting legitimately close small particles.
- **Scope after Stage B's CC-collapse:** the flat-top explosion is already gone;
  this stage only handles the residual mode (2) — a *handful* of non-touching
  duplicate bumps on broad tops — so N here is small.
- **Fix:** add `where_close_variable(pos, separations, intensity)` to `find.py`: a
  KD-tree pairs features, and the dimmer of a pair is dropped when their distance is
  below the **larger** feature's `separation`. `locate` calls it for poly instead of
  `where_close`.
- **Critical: `separations` must be per-feature RADIUS (`diameter // 2`), not
  `diameter + 1`.** A feature is a duplicate of another only when it sits inside the
  other's *body* (a secondary maximum on the same particle). Keying on the full
  `diameter + 1` separation was catastrophic at wide size ranges: a large particle's
  separation (~40–52 px at x10) deleted every genuinely-separate small neighbour
  within it (measured: x10/medium recall collapsed to ~0.17; the radius rule
  restores it to ~0.31, with the remainder lost to bootstrap size mis-assignment —
  see the neighbour-limited-windows future work). Radius keying never merges
  non-overlapping particles (their distance ≥ sum of radii ≥ the larger radius) and
  still catches same-particle duplicates.
- **Effort:** medium. **Risk:** medium (verified: `test_small_near_large_not_
  deduplicated`).

### Stage F — Uncertainty `ep` (characterize path only)

- **Today:** `measure_noise(image, raw_image, radius)`, `N_binary_mask(radius)`,
  `_static_error(mass, noise, radius, noise_size)` — all with one radius.
- **Fix:** compute `ep` **per bucket** using each bucket's radius, alongside the
  Stage C refine, then concatenate. `measure_noise`'s black-level/noise depend only
  weakly on radius — computing once at `r_max` is an acceptable simplification if
  per-bucket is awkward. Gated by `characterize`.
- **Effort:** low–medium. **Risk:** low.

### Stage G — `minmass` / `maxsize` filters

- **`maxsize`:** honored as-is (same as monodisperse).
- **`minmass`:** kept as a **global floor**; a single threshold under-serves
  polydispersity (small particles carry less mass than big ones), so document it.
  Size-normalized minmass is **out of scope**.
- **Size-band filter — considered and REMOVED.** An earlier version dropped
  features whose measured size implied a diameter outside `[min, max]`. It was
  removed because (a) it dropped *legitimate* edge-sized particles whose measured
  size scattered just past the band, costing recall at narrow ranges (a regression
  vs. baseline — see the accuracy study), and (b) it was unreliable at its intended
  job: an oversized/merged blob is refined in the clamped `max` window, which
  truncates its measured size back toward `max`, so it can't be told apart from a
  legitimate max-sized particle. **Current behaviour:** out-of-range particles are
  simply clamped to the nearest bucket (an oversized feature is reported at
  `max_diameter`, not dropped); users wanting explicit size rejection use `maxsize`.
- **Effort:** low.

### Stage H — SPIFF pixel-locking correction (`spiff.py`)

- **Today:** `apply_spiff` (`spiff.py`) removes sub-pixel (pixel-locking) bias by
  histogram-equalization: for each position column it pools the fractional parts of
  **every row**, folds them around the pixel center (`spiff.py:82`), builds **one**
  empirical distribution `spiff_sorted` (`spiff.py:85`), and remaps all positions
  through that single CDF (`spiff.py:88-90`) so the pooled sub-pixel histogram
  becomes uniform. Wired in via `locate(..., spiff=...)` per-frame
  (`feature.py:466`) and `batch(..., spiff=...)`, which **pools across all frames**
  then corrects once (`feature.py:559, 620-622`). `MIN_FEATURES = 50`.
- **Breaks:** the single pooled `spiff_sorted` encodes "all features share one bias
  signature." In poly mode the pooled histogram is a **mixture** of per-size-class
  bias curves (small particles are undersampled and pixel-lock hard; large ones
  barely lock). The correction is *empirical* — it flattens whatever histogram it is
  given — so flattening the mixture flattens *neither* component: it over-corrects
  large particles and under-corrects small ones. (Stage C's per-feature windows make
  the mask coverage roughly size-independent, but the intrinsic *undersampling* of
  small particles remains — the dominant, size-dependent effect — so the mixture
  problem stands.)
- **Fix — size-class-aware `apply_spiff`:**
  1. Add a **`groupby` parameter**; build a separate `spiff_sorted` per group (the
     existing per-column logic runs unchanged within each group, reassembled on the
     original index).
  2. **Default `groupby='auto'`: use the `diameter` column if present, else pool**
     (today's behaviour). Since `diameter` is the poly-only output column, this makes
     both paths correct with **zero new wiring** — the pooled `batch` correction
     (`feature.py:620`) auto-stratifies whenever the input is polydisperse, and a
     direct `apply_spiff(batch_result)` call also does the right thing. Monodisperse
     output has no `diameter` column → pools exactly as now.
  3. **Adaptive merging (essential).** A naive per-`diameter` group would leave most
     classes below `MIN_FEATURES=50` and silently uncorrected. Because `diameter` is
     *ordinal*, merge adjacent classes until each bin clears the threshold (fall back
     to fully pooled only if the whole set is borderline). This decouples the
     correction binning (wants fat bins for a stable CDF) from the refine buckets
     (want fine bins for accuracy).
- **Caveats:**
  - Needs even more features than pooled SPIFF (≈ `n_bins × 50`) → really a
    batch-over-a-movie operation; single-frame `locate(spiff=...)` mostly no-ops as
    today. Existing `warn_if_insufficient` / `MIN_FEATURES` logic carries over
    per-bin.
  - **Never biased when the bias is actually uniform** — grouping only costs some
    per-bin statistical noise (guarded by the merge + threshold), converging to the
    pooled result when size doesn't matter and beating it when it does. Safe default.
  - `size` (Rg) is the finer bias driver for sub-bucket stratification, but
    `diameter` is discrete, already emitted, and ordinal → the natural default key.
- **Effort:** low (local change in `spiff.py`; no `locate`/`batch` plumbing change).
  **Risk:** low.

---

## 5. Public API changes

**No new parameters.** Poly mode is selected by the *type* of the existing
`diameter` argument: a number/tuple (monodisperse, unchanged) **or** a
`Polydisperse` config object. This encodes the mutual exclusion in the type system
(the invalid combination is inexpressible), keeps poly-only options off the
`locate`/`batch` signatures, and makes `batch` work with **zero** signature changes
(it already forwards `diameter` first-positional).

```python
tp.locate(img, 11)                                   # monodisperse, unchanged
tp.locate(img, tp.Polydisperse(5, 51))               # poly, all defaults
tp.locate(img, tp.Polydisperse(5, 51, edge_frac=0.2))
tp.batch(frames, tp.Polydisperse(5, 51))             # free — no batch changes
```

### The `Polydisperse` config object

Public class (exported from `trackpy`) that **validates in `__init__`** so errors
surface at construction. It also carries the poly-specific behaviour as **methods**
(`resolve`, `refine`, `static_error`), so `feature.py` imports only the class:

```python
class Polydisperse:
    def __init__(self, min_diameter, max_diameter, edge_frac=0.1):
        # validation (fail fast): odd positive min/max diameters (scalar or
        # equal-entry tuple -> isotropic only); max >= min; 0 < edge_frac < 1.
        ...
```

- Positional `min_diameter`, `max_diameter` (named to avoid shadowing the `min`/
  `max` builtins). `Polydisperse(5, 51)` reads cleanly.
- `edge_frac` (default 0.1) is the single sizing knob (see Stage C).

### Dispatch in `locate` / `batch`

`diameter` becomes polymorphic (number | tuple | `Polydisperse`). **`locate` stays a
single function** — it is *not* forked into a parallel clone (which would duplicate
the shared ~70%: preprocessing, scale correction, `minmass`/`maxsize` filter,
`topn`, `ep`, `spiff`, frame tagging). The mode is resolved up front and the
*divergent stages* are the only branches:

```python
poly = diameter if isinstance(diameter, Polydisperse) else None
if poly is not None:
    resolved = poly.resolve(ndim)                  # min/max diameter, r_max
# ...shared preprocessing...
# detection:  grey_dilation(..., collapse_flat=poly is not None)
# refine:     refined = poly.refine(...) if poly else refine_com(...)
# dedup:      where_close_variable(...) if poly else where_close(...)
# ep:         poly.static_error(...) if poly else <inline>
# ...shared scale correction / filter / topn / spiff / frame tag...
```

The poly logic lives in `Polydisperse` methods that **return into** `locate`'s
shared tail rather than re-implementing it — keeping the monodisperse path
byte-for-byte unchanged and the shared pipeline single-copy. `batch` needs no
signature change — a `Polydisperse` flows through its existing `diameter` argument
into `locate`.

### Output DataFrame

Existing columns unchanged: `x, y[, z], mass, size, ecc, signal, raw_mass, ep`.
**New column `diameter`** = per-feature assigned diameter (poly mode only).
Document `size` (Rg) vs. `diameter` (assigned window) distinction in both
`locate` and `batch` Returns docstrings.

### Validation (in `Polydisperse.__init__`)

- `min_diameter`, `max_diameter` odd integers (or equal-entry tuples of odd ints);
  `max_diameter >= min_diameter`.
- Isotropic only → raise a clear error for anisotropic (unequal-tuple) diameters,
  mirroring the existing anisotropic-`maxsize` restriction (`feature.py:331-333`).
- `0 < edge_frac < 1`.
- Mutual exclusion with monodisperse `diameter` is automatic — there is only one
  `diameter` slot, so no cross-parameter check is needed.

---

## 6. Implementation phases

| Phase | Work | Files |
|---|---|---|
| 0 | `Polydisperse` class (export from `trackpy`) with `__init__` validation (odd, min≤max, isotropy guard, `0 < edge_frac < 1`); `isinstance` dispatch in `locate`/`batch`; `resolve(ndim)` → min/max diameter, r_max | `polydisperse.py`, `feature.py`, `trackpy/__init__.py` |
| 1 | Stage A + D: `smoothing_size = maxdiameter`, conservative margin | `feature.py` |
| 2 | Stage B: peak-find at `mindiameter` separation + connected-component collapse of the maxima mask (opt-in in `grey_dilation`) | `feature.py`, `find.py` |
| 3 | Stage C: curve-of-growth sizing → bucketed refine → emit `diameter` column | `polydisperse.py` (reuses `refine_com`) |
| 4 | Stage E: `where_close_variable` size-aware dedup | `find.py`, `feature.py` |
| 5 | Stage F + G: per-bucket `ep`; `minmass`/`maxsize` filters (no size-band filter — removed, see §7/§8) | `feature.py` |
| 6 | Stage H: size-class-aware `apply_spiff` (`groupby='auto'` → `diameter`, adaptive-merge to `MIN_FEATURES`); no `locate`/`batch` plumbing change | `spiff.py` |
| 7 | Tests (synthetic polydisperse via `trackpy.artificial`, incl. per-class SPIFF) + docstrings (`locate`, `batch`, `apply_spiff`) | `trackpy/tests/test_feature.py`, `trackpy/tests/test_spiff.py`, `feature.py`, `spiff.py` |

Phases 1–2 and 4 are largely independent; Phase 3 is the spine everything attaches
to. Do 0 → 1 → 2 → 3 first for an end-to-end working path, then 4/5, then 6 (SPIFF),
then 7 (tests/docs). Phase 6 depends only on Phase 3 emitting the `diameter` column.

---

## 7. Scope, limitations & open items

**In scope:** `locate`, `batch`; isotropic 2D/3D; size-class-aware SPIFF
(`apply_spiff`, Stage H).

**Out of scope (documented, not implemented):**
- `find_link` / linking-based finder (parallel code path, ~doubles surface).
- Anisotropic (per-axis tuple) poly mode — needs a scalar size to bucket on.
- Size-normalized `minmass`.

**Known limitations to document:**
- Global `percentile` brightness gate can let bright large particles suppress faint
  small ones (Stage B).
- Single `minmass` floor is imperfect across a size range (Stage G).
- Size-class-aware SPIFF needs ≈ `n_bins × MIN_FEATURES` features to correct all
  classes; sparse size classes are merged or fall back to pooled correction, and a
  single frame mostly no-ops (Stage H).
- Two equally-bright particles whose maxima regions touch with no intensity valley
  between them (e.g. adjacent saturated particles) collapse to a single detection in
  Stage B's CC step. Not a regression (monodisperse `where_close` merges them too)
  and fundamentally unresolvable by Crocker–Grier without an intensity valley.
- **`topn` ranks by integrated `mass`** (`feature.py:442-449`), which grows with
  particle size, so in poly mode "N brightest" ≈ "N largest" — small particles are
  systematically discarded. No `diameter` reference, but a quiet size assumption
  (same category as SPIFF). Documented only; a size-stratified `topn` (top-N per
  `diameter` class) is a possible future opt-in, not implemented.
- **Bandpass `threshold` clip** (`preprocessing.py:139`, default `1` / `1/255`) is a
  single global intensity floor; it removes a size-dependent fraction of each
  particle's tail and bites faint/small particles hardest (same family as
  `percentile`/`minmass`). No code change.
- **Stage E occlusion (residual).** With radius-keyed dedup (Stage E), a small
  particle is dropped only if its centre lies *inside* a large particle's body — an
  unresolvable overlap. That residual is inherent; the catastrophic version (below)
  is fixed.
- **Isolated-particle position accuracy (minor).** For well-separated particles on
  a clean background a large window is optimal (least pixel-locking); poly's
  size-matched windows lock slightly more, so its sub-pixel position is marginally
  worse than the `diameter=max` baseline there (worst at high noise). SPIFF narrows
  the gap. Recall is unaffected (perfect); this is a small position-precision gap
  only.

**Resolved (kept for history):** the dense mixed-size *collapse* (poly recall < 0.2
at wide range + high density) is **fixed**. It was misdiagnosed three times before
tracing pinned it down: (1) boxcar over-subtraction — a white top-hat background did
not recover it and is noise-fragile (reverted); (2) the global `percentile` gate — a
noise-referenced threshold did not recover it (reverted); (3) the Stage E dedup
keying on `diameter+1` — fixed to key on radius, which helped but left recall ~0.31.
The true root was **bootstrap size mis-assignment**: an `r_max` *moment* window
engulfs neighbours in a crowd, so small particles measured a huge Rg and were
assigned the max diameter. Replacing the moment bootstrap with the **curve-of-growth**
sizing (Stage C) — which reads each feature's own extent — restored **recall and
precision to 1.00 across the whole grid**.

### Future work

- **Neighbour-limited windows for isolated-particle position.** The one remaining
  gap above is the isolated-particle pixel-locking difference. Refining the
  *centroid* in a larger (neighbour-limited) window would tie the baseline there
  while keeping the size-matched window for characterization. Minor; SPIFF already
  narrows it, and recall is perfect — revisit only if sub-pixel precision on
  isolated particles matters.
- **Size-aware brightness threshold** — a *minor* improvement (the global
  `percentile` gate can slightly clip faint particles). A noise-referenced threshold
  was prototyped and gave only marginal grid gains before being reverted. (A white
  top-hat background was also tried and rejected — noise-fragile.)
- **Size-stratified `topn`** (top-N per `diameter` class) so "N brightest" stops
  meaning "N largest".

**Testing notes:** build synthetic polydisperse fields with `trackpy.artificial`
(draw features at several radii in one frame); assert (a) counts, (b) positions
within tolerance, (c) `diameter`/`size` correlate with ground truth, (d) closely
spaced small particles are both found, (e) a large particle yields exactly one
detection (Stage E dedup), (f) monodisperse `diameter` path is unchanged
(regression), (g) size-class-aware SPIFF flattens each class's sub-pixel histogram
where a single pooled correction does not, and pooled behaviour is unchanged for
monodisperse output (no `diameter` column).

---

## 8. Decisions locked in

1. **API:** no new parameters — overload `diameter` to accept a number/tuple
   (monodisperse, unchanged) or a `Polydisperse(min_diameter, max_diameter,
   edge_frac=0.1)` object. Mutual exclusion is type-encoded; `batch` needs no
   signature change; validation lives in `Polydisperse.__init__`.
2. **Sizing:** per-feature **curve of growth** — each feature's diameter is where
   its annular (ring) mass first decays below `edge_frac` of its peak. Reads the
   feature's own extent, robust to neighbours and to duplicate peaks; **retired the
   `Rg → diameter` (`rg_to_diameter`) mapping and the moment bootstrap** it needed.
2b. **Peak dedup split:** flat-top explosion killed immediately by
   connected-component collapse at detection (Stage B); only non-touching residual
   duplicates deferred to size-aware dedup (Stage E, keyed on **radius**).
2c. **Bucketing:** one bucket per odd diameter in 2D (lossless); **cap at 10
   geometrically-spaced buckets in 3D** to bound the mask cache.
3. **`edge_frac`:** the single sizing knob (default **0.1**); larger → tighter
   windows. Replaces `rg_to_diameter` — the growth curve measures extent directly,
   so no `Rg`-scaling constant is needed.
4. **Output:** add `diameter` column (assigned per-feature diameter). Doubles as the
   grouping key for size-class-aware SPIFF.
5. **Scope:** `locate` + `batch`, isotropic only.
6. **SPIFF (Stage H):** make `apply_spiff` size-class-aware via `groupby='auto'`
   (uses the `diameter` column when present, else pools), with adaptive-merge of
   adjacent classes to meet `MIN_FEATURES`. No `locate`/`batch` plumbing change.
   Kept after ablation: a modest position-accuracy gain over pooled at narrow
   ranges / high noise, and it degenerates exactly to pooled when per-class features
   are too few (wide ranges).
7. **No automatic size-band filter** (see Stage G): it regressed recall at narrow
   ranges and was unreliable; out-of-range features are clamped to the nearest
   bucket, and `maxsize` remains for explicit rejection.

---

## 9. Implementation status

Implemented and tested (`trackpy/polydisperse.py`, `feature.py`, `find.py`,
`spiff.py`; tests in `trackpy/tests/test_polydisperse.py`):

- `Polydisperse` config class with `resolve`, `refine`, `static_error` **methods**
  (not free functions); `feature.py` imports only the class and calls
  `poly.resolve(ndim)` / `poly.refine(...)` / `poly.static_error(...)`. The single
  `locate` pipeline branches with `if poly` at the divergent stages; monodisperse is
  byte-for-byte unchanged.
- Stages A (max smoothing), B (min-scale separation + `grey_dilation(collapse_flat)`
  connected-component collapse), C (**curve-of-growth sizing** → bucketed refine →
  `diameter` column), D (conservative margin), E (`where_close_variable`, radius-
  keyed), F (per-bucket `ep`), H (size-class-aware `apply_spiff`).
- Stage G reduced to `minmass`/`maxsize` only (band filter removed, see above).

**Accuracy study (summary).** Across density × noise × size-range grids (SPIFF on
both methods), with the curve-of-growth sizing and radius-keyed dedup: **poly recall
≥ the `diameter=max` baseline in every grid cell**, and in the crowded / wide-range
cells where the baseline merges particles, poly reaches **recall and precision 1.00
with exactly the ground-truth count** (e.g. x10/high: baseline recall ~0.05 → poly
1.00; x5/medium: 0.55 → 1.00). The earlier dense-field collapse is resolved (§7).
The only residual is a minor isolated-particle *position*-precision gap (pixel
locking of small windows), which SPIFF narrows and which does not affect recall.