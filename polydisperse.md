# Poly-disperse support for `locate()` / `batch()`

Implementation plan for allowing `trackpy.locate` (and `trackpy.batch`) to detect
particles spanning a **range** of sizes, by supplying `mindiameter` + `maxdiameter`
instead of a single `diameter`.

---

## 1. Goal & design premise

Today `locate(raw_image, diameter, ...)` assumes every feature is ~the same size.
`diameter` sets a single `radius = diameter // 2` that flows through the whole
Crocker–Grier pipeline. We want to support **poly-disperse** samples: a continuum
of sizes, up to ~10× range (e.g. `mindiameter=5`, `maxdiameter=51`).

Design premise:

- **`diameter` is untouched.** If the user supplies `diameter`, behaviour is
  byte-for-byte identical to today. Zero regression risk for existing users.
- A **new optional pair `mindiameter` + `maxdiameter`** activates "poly mode".
  - Both required together.
  - Mutually exclusive with `diameter` (error if `diameter` also given).
  - Each must be an odd integer; `maxdiameter >= mindiameter`.
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
| Two full refine passes | optimal | ~2× refine | low |
| **Cheap bootstrap + 1 bucketed refine** (chosen) | optimal | ~1× refine + O(N) | low |
| Rewritten per-feature-radius kernel | optimal (same!) | ≥1× refine, no numba hoist | high |

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
   - `rg_to_diameter` is a **new kwarg**, default from the uniform-disk relation
     `Rg = R / sqrt(2)` ⇒ `diameter = 2R = 2*sqrt(2)*Rg ≈ 2.83`.
   - Power users retune for Gaussian-ish vs. disk-ish profiles.
3. **Bucket** features by `assigned_diameter` (distinct odd diameters in range).
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
- 3D: grows as `d³`; large ranges reach tens of MB → the concrete argument for
  **coarser bucket granularity in 3D**.

**`batch()` amortization:** the cache is process-global and persistent, so masks are
built once and reused across all frames — poly does not rebuild masks per frame.

### Stage D — Edge margin

- **Today:** `margin = max(radius, separation//2 - 1, smoothing_size//2)`.
- **Breaks:** margin is applied inside `grey_dilation`, before sizes are known —
  can't be per-feature.
- **Fix:** conservative global margin from the largest scale:
  `max(r_max, (mindiameter+1)//2 - 1, maxdiameter//2)`. Excludes a slightly wider
  border (loses a few small particles near edges) rather than characterizing a big
  particle with clipped data.
- **Effort:** trivial. **Risk:** low.

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

---

## 5. Public API changes

### `locate` signature (new kwargs, all optional / backward-compatible)

```python
def locate(raw_image, diameter=None, minmass=None, maxsize=None, separation=None,
           noise_size=1, smoothing_size=None, threshold=None, invert=False,
           percentile=64, topn=None, preprocess=True, max_iterations=10,
           filter_before=None, filter_after=None,
           characterize=True, engine='auto',
           # --- new ---
           mindiameter=None, maxdiameter=None,
           rg_to_diameter=2 * 2 ** 0.5,      # ~2.83, uniform-disk Rg->diameter
           max_radius_iterations=1):
```

Note: `diameter` becomes `diameter=None` so it can be omitted in poly mode.
Validation must enforce exactly one of {`diameter`} / {`mindiameter` &
`maxdiameter`} is provided.

`batch` (`feature.py:464`) forwards `**kwargs`; add the same params to its
signature/docstring and pass through.

### Output DataFrame

Existing columns unchanged: `x, y[, z], mass, size, ecc, signal, raw_mass, ep`.
**New column `diameter`** = per-feature assigned diameter (poly mode only).
Document `size` (Rg) vs. `diameter` (assigned window) distinction in both
`locate` and `batch` Returns docstrings.

### Validation rules (Phase 0)

- `diameter` and (`mindiameter`/`maxdiameter`) are mutually exclusive.
- `mindiameter` and `maxdiameter` must be supplied together.
- Both odd integers; `maxdiameter >= mindiameter`.
- Poly mode requires isotropic (scalar) diameters → raise a clear error for
  tuples, mirroring the existing anisotropic-`maxsize` restriction
  (`feature.py:331-333`).

---

## 6. Implementation phases

| Phase | Work | Files |
|---|---|---|
| 0 | Param plumbing + validation (mutual exclusion, odd, min≤max, isotropy guard); derive `r_min`/`r_max`; branch poly vs mono | `feature.py`, `utils.py` |
| 1 | Stage A + D: `smoothing_size = maxdiameter`, conservative margin | `feature.py` |
| 2 | Stage B: peak-find at `mindiameter` separation + connected-component collapse of the maxima mask (opt-in in `grey_dilation`) | `feature.py`, `find.py` |
| 3 | Stage C: cheap bootstrap → tunable radius assignment → bucketed refine (optional iterate) → emit `diameter` column | `feature.py` (reuses `refine_com_arr`, `estimate_mass/size`) |
| 4 | Stage E: `where_close_variable` size-aware dedup | `find.py`, `feature.py` |
| 5 | Stage F + G: per-bucket `ep`, size-band filter | `feature.py` |
| 6 | Tests (synthetic polydisperse via `trackpy.artificial`) + docstrings (`locate`, `batch`) | `trackpy/tests/test_feature.py`, `feature.py` |

Phases 1–2 and 4 are largely independent; Phase 3 is the spine everything attaches
to. Do 0 → 1 → 2 → 3 first for an end-to-end working path, then 4/5, then 6.

---

## 7. Scope, limitations & open items

**In scope:** `locate`, `batch`; isotropic 2D/3D.

**Out of scope (documented, not implemented):**
- `find_link` / linking-based finder (parallel code path, ~doubles surface).
- Anisotropic (per-axis tuple) poly mode — needs a scalar size to bucket on.
- Size-normalized `minmass`.

**Known limitations to document:**
- Global `percentile` brightness gate can let bright large particles suppress faint
  small ones (Stage B).
- Single `minmass` floor is imperfect across a size range (Stage G).
- Bootstrap size estimate can be contaminated for a small particle adjacent to a
  large bright one; mitigated by `max_radius_iterations > 1` (Stage C).
- Two equally-bright particles whose maxima regions touch with no intensity valley
  between them (e.g. adjacent saturated particles) collapse to a single detection in
  Stage B's CC step. Not a regression (monodisperse `where_close` merges them too)
  and fundamentally unresolvable by Crocker–Grier without an intensity valley.

**Testing notes:** build synthetic polydisperse fields with `trackpy.artificial`
(draw features at several radii in one frame); assert (a) counts, (b) positions
within tolerance, (c) `diameter`/`size` correlate with ground truth, (d) closely
spaced small particles are both found, (e) a large particle yields exactly one
detection (Stage E dedup), (f) monodisperse `diameter` path is unchanged
(regression).

---

## 8. Decisions locked in

1. **API:** keep `diameter`; add `mindiameter`/`maxdiameter` (mutually exclusive).
2. **Refinement:** cheap non-iterative bootstrap + single **bucketed** refine pass
   (bucketing is lossless vs. per-feature radius; reuses `refine_com_arr`).
2b. **Peak dedup split:** flat-top explosion killed immediately by
   connected-component collapse at detection (Stage B); only non-touching residual
   duplicates deferred to size-aware dedup (Stage E).
3. **`rg_to_diameter`:** tunable kwarg, default `2*sqrt(2) ≈ 2.83` (uniform-disk).
4. **Output:** add `diameter` column (assigned per-feature diameter).
5. **Optional convergence:** `max_radius_iterations`, default 1 (off).
6. **Scope:** `locate` + `batch`, isotropic only.