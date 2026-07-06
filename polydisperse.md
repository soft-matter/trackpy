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
  - Bundles `min_diameter`, `max_diameter`, `rg_to_diameter`, `max_radius_iterations`.
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
  (no COM iteration) mass and radius-of-gyration over an `r_max` window. These are
  the cheap bootstrap primitives we reuse in Stage C.
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

**Passes.** The chicken-and-egg (need a size estimate to pick a window; need a
window to estimate size) makes *a* bootstrap unavoidable. But it must be **cheap**:

- Bootstrap = single-shot `estimate_mass` + `estimate_size` at `r_max`
  (no COM iteration), **not** a full iterative refine pass.
- Then **one** real bucketed refine pass at each feature's assigned radius.
- Total ≈ monodisperse refine cost + O(N) bootstrap, instead of 2× full refine.

| Approach | Accuracy | Cost | Code risk |
|---|---|---|---|
| Two full refine passes | optimal final; slightly better *first-guess* bucket | ~2× refine | low |
| **Cheap bootstrap + 1 bucketed refine** (chosen) | identical final; first-guess bucket may differ only for smallest borderline particles | ~1× refine + O(N) | low |
| Rewritten per-feature-radius kernel | optimal (same!) | ≥1× refine, no numba hoist | high |

**Why the cheap bootstrap is not less accurate.** Both variants end with the *same*
full refine at the assigned radius (Pass 2), which re-converges to the same
center-of-brightness regardless of starting point — so final position/mass/size/ecc
are identical *given the same bucket*. The only channel for divergence is the size
estimate that picks the bucket: the bootstrap measures Rg from a 1-iteration
(~1 px off-center) window, inflating Rg by ≈`δ²` (parallel-axis effect). Through the
assignment (bucket width = 2 in diameter) this shifts diameter by ~0.14 for a big
particle (never changes the bucket) and up to ~0.45 for the smallest particles (can
occasionally cross a bucket edge). That residual is *shared contamination* — both
variants estimate at `r_max`, so a small particle's Rg is dominated by neighbor/
background contamination that full convergence does **not** fix. A full Pass-1 at
`r_max` can even *mis-center* a small particle next to a big bright one (large window
pulls COM toward the neighbor), making its Pass-2 start worse than the raw detected
pixel. The correct cure for the borderline-bucket sensitivity is
`max_radius_iterations > 1` (re-estimate at the assigned, uncontaminated radius,
re-bucket, re-refine) — a fixed-point iteration to which **both variants converge
identically**. So the extra full pass buys only a marginally better first guess that
the iteration would fix anyway.

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

### Stage C — Refinement + characterization (`refine_com` / `radius`) — the spine

Replace the single `refine_com` call (poly mode only) with:

1. **Bootstrap (cheap, no iteration):** for each detected peak, compute mass then
   Rg over an `r_max` window using `estimate_mass` / `estimate_size`
   (`feature.py:175-188`), or `refine_com_arr(..., max_iterations=1)`. O(N).
2. **Assign per-feature diameter (tunable):**
   `assigned_diameter = round_to_odd(rg_to_diameter * size)`, clamped to
   `[mindiameter, maxdiameter]`.
   - `rg_to_diameter` is a **`Polydisperse` field**, default derived from a **Gaussian**
     profile: for an isotropic Gaussian `Rg = σ·√n` (n = ndim), and a window
     half-width `R = k·σ` gives `rg_to_diameter = 2R/Rg = 2k/√n`. Default **k = 2.5**
     (captures ~95.6% of the mass in 2D, ~89.9% in 3D):
     - **2D:** `5/√2 ≈ 3.536`
     - **3D:** `5/√3 ≈ 2.887`
   - Because the default is dimension-aware, the kwarg defaults to `None` and is
     resolved to `5/√2` or `5/√3` from the image's `ndim` inside `locate`.
   - Power users retune via `k`: `2k/√n` (e.g. `k=2` tighter for crowded fields,
     `k=3` for maximal mass fidelity), or pass `rg_to_diameter` directly.
   - Caveat: the derivation assumes an untruncated ideal Gaussian; the bootstrap
     measures `Rg` in the `r_max` window, so far-field noise (surviving the bandpass
     `threshold`) can slightly inflate `Rg` for very small particles — mitigated by
     the `threshold` clip and `max_radius_iterations > 1`.
3. **Bucket** features by `assigned_diameter`:
   - **2D (and any case with ≤ 10 distinct odd diameters in range):** one bucket per
     distinct odd diameter — lossless (mask radii are integers anyway).
   - **3D with a wide range (> 10 distinct odd diameters):** cap at **10 buckets**.
     Choose 10 bucket diameters spaced **geometrically** (log) across
     `[mindiameter, maxdiameter]`, rounded to the nearest odd and deduplicated, then
     snap each feature to the nearest bucket diameter. Geometric spacing keeps the
     *relative* window error roughly uniform across a 10× range (a 2 px error matters
     far more at d=5 than d=51), and bounds the persistent mask cache to ≤ 10 radii.
     The give-up is a small window-size quantization on 3D features only.
4. **Refine (real pass):** for each bucket, call `refine_com_arr` **once** on that
   subset at the bucket's radius, starting from bootstrap positions. Concatenate
   results, **preserving the original coordinate order/index**.
5. **Optional convergence:** `max_radius_iterations` kwarg (**default 1** = off).
   When >1, re-estimate size from the refined result, re-bucket, re-refine until
   stable. Only matters in dense mixed fields (a small particle next to a big
   bright one can bootstrap into a too-large bucket).
6. **Emit `diameter` column** = per-feature `assigned_diameter`, alongside the
   existing `size` (Rg).

- **No change to `refine_com*` or `masks.py`.** All new logic is orchestration in
  a `feature.py` helper (e.g. `_refine_polydisperse`).
- **Effort:** medium (bulk of the work). **Risk:** medium — the `rg_to_diameter`
  default and bootstrap contamination are the sensitivities; both mitigated above.

#### Stage C — performance vs. monodisperse

Cost model. `N` = feature count, `I` = `max_iterations` (default 10), `m(r)` = mask
pixels (≈ `π r²` 2D / `(4/3)π r³` 3D), `m̄` = population-average mask size.

- **Monodisperse:** `T_mono(r) = N·I·c·m(r)`.
- **Poly:** `T_poly ≈ N·c·[ m(r_max) + I·m̄ ]` — a single-shot bootstrap at `r_max`
  (I=1) plus one real refine with each feature at its own (mostly smaller) radius.
  (×1 unless `max_radius_iterations > 1`.)

Comparisons:

- **vs. ideal single-radius pass at average radius:**
  `T_poly/T_mono(r̄) = 1 + m(r_max)/(I·m̄) ≈ 1.2` (I=10, `m(r_max)≈2·m̄`) → **~20%
  overhead**, all from the bootstrap. Not 2×: bootstrap is single-iteration and the
  refine uses per-feature masks.
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
- **Fix:** add `where_close_variable(pos, separations, intensity)` to `find.py`,
  where `separations` is an `(N, ndim)` array derived from each feature's measured
  size (mapped via the same `rg_to_diameter`, clamped to `[min, max]`). Query the
  KD-tree at the global `r_max` radius, then keep only pairs whose distance is below
  the **larger** feature's separation; drop the dimmer. `locate` calls this instead
  of `where_close` in poly mode (replacing lines 406–409).
- **Effort:** medium. **Risk:** medium (correctness of the pair-filter rule — needs
  a unit test).

### Stage F — Uncertainty `ep` (characterize path only)

- **Today:** `measure_noise(image, raw_image, radius)`, `N_binary_mask(radius)`,
  `_static_error(mass, noise, radius, noise_size)` — all with one radius.
- **Fix:** compute `ep` **per bucket** using each bucket's radius, alongside the
  Stage C refine, then concatenate. `measure_noise`'s black-level/noise depend only
  weakly on radius — computing once at `r_max` is an acceptable simplification if
  per-bucket is awkward. Gated by `characterize`.
- **Effort:** low–medium. **Risk:** low.

### Stage G — `minmass` / `maxsize` filters

- **`maxsize`:** in poly mode, `[mindiameter, maxdiameter]` implies a natural size
  band — drop features whose measured size maps outside the band (merged/garbage
  detections). Keep `maxsize` honored if also supplied.
- **`minmass`:** keep as a **global floor**, but **document** that a single
  threshold under-serves polydispersity (small particles carry less mass than big
  ones; one floor either passes big-particle fragments or culls faint small ones).
  Size-normalized minmass is **out of scope**.
- **Effort:** low. **Risk:** low (documented sharp edge).

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
  large particles and under-corrects small ones. (Stage C's `rg_to_diameter` scaling
  removes the *mask-coverage* axis of variation but not the intrinsic *undersampling*
  axis, which is the dominant, size-dependent one — so the mixture problem stands.)
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
tp.locate(img, tp.Polydisperse(5, 51, rg_to_diameter=3.2))
tp.batch(frames, tp.Polydisperse(5, 51))             # free — no batch changes
```

### The `Polydisperse` config object

New public class (exported from `trackpy`), a small dataclass-like container that
**validates in `__init__`** so errors surface at construction, not deep inside
`locate`:

```python
class Polydisperse:
    def __init__(self, min_diameter, max_diameter,
                 rg_to_diameter=None, max_radius_iterations=1):
        # __init__ validation (fail fast):
        #   - min_diameter, max_diameter are odd integers (or tuples of odd ints)
        #   - max_diameter >= min_diameter (elementwise)
        #   - isotropic only for now: scalar, or tuple with equal entries →
        #     else raise (mirrors the anisotropic-maxsize restriction,
        #     feature.py:331-333)
        #   - rg_to_diameter is None (→ resolved from ndim in locate) or > 0
        #   - max_radius_iterations >= 1
        ...
```

- Positional `min_diameter`, `max_diameter` (named to avoid shadowing the `min`/
  `max` builtins). `Polydisperse(5, 51)` reads cleanly.
- `rg_to_diameter` stays `None` on the object and is resolved to the dimension-aware
  Gaussian default `2k/√n` (k=2.5 → 2D `5/√2 ≈ 3.54`, 3D `5/√3 ≈ 2.89`) inside
  `locate`, once `ndim` is known (the object can't know `ndim` at construction).
- `max_radius_iterations` default 1 (off).

### Dispatch in `locate` / `batch`

`diameter` becomes polymorphic (number | tuple | `Polydisperse`). **`locate` stays a
single function** — it is *not* forked into a parallel `_locate_polydisperse` clone,
which would duplicate the shared ~70% (preprocessing, scale correction, `minmass`/
`maxsize` filter, `topn`, `ep`, `spiff`, frame tagging). Instead, the mode is resolved
up front and the *divergent stages* are the only branches:

```python
poly = diameter if isinstance(diameter, Polydisperse) else None
if poly is not None:
    resolved = _resolve_polydisperse(poly, ndim)   # r_min/r_max/rg_to_diameter/...
# ...shared preprocessing...
# detection:  grey_dilation(..., cc=poly is not None)
# refine:     refined = _refine_polydisperse(...) if poly else refine_com(...)
# dedup:      where_close_variable(...) if poly else where_close(...)
# ep:         _static_error_per_bucket(...) if poly else <inline>
# ...shared scale correction / filter / topn / spiff / frame tag...
```

The heavy poly logic lives in stage helpers (e.g. `_refine_polydisperse`) that
**return into** `locate`'s shared tail rather than re-implementing it — keeping the
monodisperse path byte-for-byte unchanged and the shared pipeline single-copy.
`batch` needs no signature change — a `Polydisperse` flows through its existing
`diameter` argument into `locate`. Only its docstring gains a mention.

### Output DataFrame

Existing columns unchanged: `x, y[, z], mass, size, ecc, signal, raw_mass, ep`.
**New column `diameter`** = per-feature assigned diameter (poly mode only).
Document `size` (Rg) vs. `diameter` (assigned window) distinction in both
`locate` and `batch` Returns docstrings.

### Validation (in `Polydisperse.__init__`, Phase 0)

- `min_diameter`, `max_diameter` odd integers (or tuples of odd ints);
  `max_diameter >= min_diameter`.
- Isotropic only → raise a clear error for anisotropic (unequal-tuple) diameters,
  mirroring the existing anisotropic-`maxsize` restriction (`feature.py:331-333`).
- `rg_to_diameter` is `None` or positive; `max_radius_iterations >= 1`.
- Mutual exclusion with monodisperse `diameter` is automatic — there is only one
  `diameter` slot, so no cross-parameter check is needed.

---

## 6. Implementation phases

| Phase | Work | Files |
|---|---|---|
| 0 | `Polydisperse` class (export from `trackpy`) with `__init__` validation (odd, min≤max, isotropy guard, positive `rg_to_diameter`, `max_radius_iterations>=1`); `isinstance` dispatch in `locate`/`batch`; derive `r_min`/`r_max` | `feature.py` (or new module), `trackpy/__init__.py` |
| 1 | Stage A + D: `smoothing_size = maxdiameter`, conservative margin | `feature.py` |
| 2 | Stage B: peak-find at `mindiameter` separation + connected-component collapse of the maxima mask (opt-in in `grey_dilation`) | `feature.py`, `find.py` |
| 3 | Stage C: cheap bootstrap → tunable radius assignment → bucketed refine (optional iterate) → emit `diameter` column | `feature.py` (reuses `refine_com_arr`, `estimate_mass/size`) |
| 4 | Stage E: `where_close_variable` size-aware dedup | `find.py`, `feature.py` |
| 5 | Stage F + G: per-bucket `ep`, size-band filter | `feature.py` |
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
- Bootstrap size estimate can be contaminated for a small particle adjacent to a
  large bright one; mitigated by `max_radius_iterations > 1` (Stage C).
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
- **Stage E occlusion:** the size-aware dedup rule "drop the dimmer if within the
  *larger* feature's separation" can drop a real small particle that legitimately
  sits inside a large particle's exclusion zone. Inherent ambiguity.

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
   rg_to_diameter=None, max_radius_iterations=1)` object. Mutual exclusion is
   type-encoded; `batch` needs no signature change; validation lives in
   `Polydisperse.__init__`.
2. **Refinement:** cheap non-iterative bootstrap + single **bucketed** refine pass
   (bucketing is lossless vs. per-feature radius; reuses `refine_com_arr`).
2b. **Peak dedup split:** flat-top explosion killed immediately by
   connected-component collapse at detection (Stage B); only non-touching residual
   duplicates deferred to size-aware dedup (Stage E).
2c. **Bucketing:** one bucket per odd diameter in 2D (lossless); **cap at 10
   geometrically-spaced buckets in 3D** to bound the mask cache.
3. **`rg_to_diameter`:** tunable kwarg, dimension-aware Gaussian default
   `2k/√n` with `k=2.5` (2D `5/√2 ≈ 3.54`, 3D `5/√3 ≈ 2.89`); defaults to `None`,
   resolved from `ndim`.
4. **Output:** add `diameter` column (assigned per-feature diameter). Doubles as the
   grouping key for size-class-aware SPIFF.
5. **Optional convergence:** `max_radius_iterations`, default 1 (off).
6. **Scope:** `locate` + `batch`, isotropic only.
7. **SPIFF (Stage H):** make `apply_spiff` size-class-aware via `groupby='auto'`
   (uses the `diameter` column when present, else pools), with adaptive-merge of
   adjacent classes to meet `MIN_FEATURES`. No `locate`/`batch` plumbing change.