"""Configuration object for polydisperse (variable-size) feature finding.

Pass a :class:`Polydisperse` instance as the ``diameter`` argument of
:func:`trackpy.locate` or :func:`trackpy.batch` to detect particles that span a
range of sizes, instead of supplying a single ``diameter``.
"""
import numbers
from collections import namedtuple

import numpy as np

from .refine import refine_com
from .masks import N_binary_mask
from .uncertainty import measure_noise, _static_error
from .utils import validate_tuple, pandas_concat, default_pos_columns


def _check_odd_int(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(
            "{} must be an odd integer, got {!r}.".format(name, value))
    value = int(value)
    if value < 1 or value % 2 == 0:
        raise ValueError(
            "{} must be a positive odd integer, got {}.".format(name, value))
    return value


def _validate_diameter(value, name):
    """Return the diameter as an int (scalar) or tuple of ints, or raise.

    Requires positive odd integers and -- for tuples -- isotropy (all entries
    equal), since polydisperse bucketing needs a single scalar size per feature.
    """
    if hasattr(value, '__iter__'):
        items = list(value)
        if len(items) == 0:
            raise ValueError("{} must not be empty.".format(name))
        validated = tuple(_check_odd_int(v, name) for v in items)
        if any(v != validated[0] for v in validated):
            raise ValueError(
                "{} entries must all be equal, got {}. Express anisotropic "
                "(fixed-shape) features with the `aspect` argument instead of "
                "unequal diameters.".format(name, validated))
        return validated
    return _check_odd_int(value, name)


def _scalar(value):
    """Reduce a validated (isotropic) diameter to its scalar magnitude."""
    return value[0] if isinstance(value, tuple) else value


def _nearest_odd(value):
    """Nearest positive odd integer to ``value`` (at least 1)."""
    odd = 2 * int(round((float(value) - 1) / 2.0)) + 1
    return max(odd, 1)


def _validate_aspect(value, name='aspect'):
    """Return the aspect ratio as a positive float scalar or tuple of floats.

    ``aspect`` is a per-axis shape multiplier: the physical diameter along axis
    ``i`` is the (scalar) reference diameter times ``aspect[i]``. A scalar is
    broadcast to every axis (isotropic). The tuple length is checked against
    ``ndim`` later, in :meth:`Polydisperse.for_ndim`.
    """
    if hasattr(value, '__iter__'):
        items = tuple(value)
        if len(items) == 0:
            raise ValueError("{} must not be empty.".format(name))
        out = []
        for v in items:
            v = float(v)
            if not v > 0:
                raise ValueError("{} entries must be positive, got {}.".format(
                    name, items))
            out.append(v)
        return tuple(out)
    v = float(value)
    if not v > 0:
        raise ValueError("{} must be positive, got {!r}.".format(name, value))
    return v


class Polydisperse:
    """Configuration for polydisperse feature finding.

    Parameters
    ----------
    min_diameter : odd int or tuple of equal odd ints
        Smallest feature diameter to detect. When in doubt, round up.
    max_diameter : odd int or tuple of equal odd ints
        Largest feature diameter to detect. Must be ``>= min_diameter``.
    aspect : number or tuple of numbers, optional
        Fixed feature shape as a per-axis multiplier, default 1 (isotropic). The
        physical diameter along axis ``i`` is ``<reference diameter> * aspect[i]``,
        where the reference diameter is the scale swept by
        ``min_diameter``/``max_diameter``; every feature is thus the same
        axis-aligned ellipse and only its overall scale varies. ``aspect`` is used
        as given (not normalised); a scalar broadcasts to all axes. Set the
        well-resolved axes to 1 and the others to their ratio -- e.g. confocal
        data (trackpy axis order ``z, y, x``) whose z extent is one third of
        ``x``/``y`` uses ``aspect=(1/3, 1, 1)``, and ``min_diameter``/
        ``max_diameter`` then range over the ``x``/``y`` diameter. Only
        axis-aligned shapes are supported (no per-feature rotation).
    edge_frac : float, optional
        Curve-of-growth boundary threshold in (0, 1), default 0.1. Each feature's
        refinement diameter is placed where its annular (ring) mass first falls
        below ``edge_frac`` of the peak ring -- i.e. where its radial intensity
        profile has decayed. Larger values give tighter windows.

    Notes
    -----
    A single scalar *scale* per feature is required for bucketing, so a tuple
    ``diameter`` must have equal entries. Anisotropy is expressed through the
    fixed-shape ``aspect`` argument rather than through unequal diameters.
    """

    def __init__(self, min_diameter, max_diameter, aspect=1, edge_frac=0.1):
        self.min_diameter = _validate_diameter(min_diameter, 'min_diameter')
        self.max_diameter = _validate_diameter(max_diameter, 'max_diameter')
        self.aspect = _validate_aspect(aspect)

        if (isinstance(self.min_diameter, tuple)
                and isinstance(self.max_diameter, tuple)
                and len(self.min_diameter) != len(self.max_diameter)):
            raise ValueError("min_diameter and max_diameter must have the same "
                             "number of dimensions.")
        if _scalar(self.max_diameter) < _scalar(self.min_diameter):
            raise ValueError(
                "max_diameter ({}) must be >= min_diameter ({}).".format(
                    _scalar(self.max_diameter), _scalar(self.min_diameter)))

        if (isinstance(edge_frac, bool)
                or not isinstance(edge_frac, numbers.Real)
                or not 0 < edge_frac < 1):
            raise ValueError("edge_frac must be a number in (0, 1), got "
                             "{!r}.".format(edge_frac))
        self.edge_frac = float(edge_frac)

    def for_ndim(self, ndim):
        """Broadcast/validate the size range and aspect to `ndim`.

        The per-axis diameter along axis ``i`` is the reference scale
        (``min_diameter``/``max_diameter``) times ``aspect[i]``. Returns the
        per-axis diameters and radii, the (raw) aspect tuple, and the scalar
        reference size range used for curve-of-growth sizing and bucketing.
        """
        aspect = validate_tuple(self.aspect, ndim)
        ref_min = _scalar(self.min_diameter)
        ref_max = _scalar(self.max_diameter)
        min_diameter = tuple(_nearest_odd(ref_min * a) for a in aspect)
        max_diameter = tuple(_nearest_odd(ref_max * a) for a in aspect)
        return SizeParams(
            min_diameter=min_diameter,
            max_diameter=max_diameter,
            r_max=tuple(d // 2 for d in max_diameter),
            aspect=aspect,
            ref_min=ref_min,
            ref_max=ref_max)

    def refine(self, raw_image, image, coords, max_iterations=10,
               engine='auto', characterize=True, pos_columns=None):
        """Refine and characterize features spanning a range of sizes.

        Each detected coordinate is assigned an odd refinement diameter from a
        curve-of-growth measurement of its own extent (robust to neighbours and
        to duplicate peaks on the same particle), then refined in buckets that
        share a diameter. Returns the refined features with an assigned
        ``diameter`` column, in the original coordinate order.
        """
        ndim = image.ndim
        sizes = self.for_ndim(ndim)
        if pos_columns is None:
            pos_columns = default_pos_columns(ndim)

        coords = np.round(np.asarray(coords)).astype(int)
        if len(coords) > 0:
            assigned, keep = _growth_diameters(image, coords, sizes, ndim,
                                               edge_frac=self.edge_frac)
            # Drop detections whose measured extent is below the declared minimum
            # diameter -- sub-minimum peaks (mostly noise) that would otherwise be
            # promoted to the smallest window, inflating false positives and the
            # refinement cost.
            coords, assigned = coords[keep], assigned[keep]

        if len(coords) == 0:
            empty = refine_com(raw_image, image, sizes.r_max, coords,
                               max_iterations=1, engine=engine,
                               characterize=characterize,
                               pos_columns=pos_columns)
            empty['diameter'] = np.empty(0, dtype=int)
            return empty

        return _bucketed_refine(raw_image, image, coords, assigned, sizes.aspect,
                                max_iterations, engine, characterize,
                                pos_columns)

    def static_error(self, features, image, raw_image, noise_size):
        """Per-bucket static (position) error for polydisperse features.

        The background level and noise are measured once at the largest window
        (they depend only weakly on radius); each feature's mass is then
        background-corrected and its error computed with its own refinement
        radius. Returns an array aligned with ``features`` -- 1-D when isotropic,
        else one column per dimension.
        """
        ndim = image.ndim
        sizes = self.for_ndim(ndim)
        black_level, noise = measure_noise(image, raw_image, sizes.r_max)
        raw_mass = features['raw_mass'].values
        diameters = features['diameter'].values
        order = np.arange(len(features))
        ep_parts, idx_parts = [], []
        for d in np.unique(diameters):
            in_bucket = diameters == d
            radius = _axis_radius(d, sizes.aspect)
            npx = N_binary_mask(radius, ndim)
            mass = raw_mass[in_bucket] - npx * black_level
            ep_parts.append(_static_error(mass, noise, radius, noise_size))
            idx_parts.append(order[in_bucket])
        ep = np.concatenate(ep_parts, axis=0)
        return ep[np.argsort(np.concatenate(idx_parts))]

    def __repr__(self):
        return ("Polydisperse(min_diameter={!r}, max_diameter={!r}, "
                "aspect={!r}, edge_frac={!r})".format(
            self.min_diameter, self.max_diameter, self.aspect,
            self.edge_frac))


# Size parameters broadcast to a given image dimensionality
# (see Polydisperse.for_ndim). `min_diameter`/`max_diameter`/`r_max` are per-axis;
# `aspect` is the (raw) per-axis shape multiplier; `ref_min`/`ref_max` are the
# scalar reference size range (the scale of the aspect==1 axes) that the
# curve-of-growth sizing and bucketing work in.
SizeParams = namedtuple(
    'SizeParams', ['min_diameter', 'max_diameter', 'r_max', 'aspect',
                   'ref_min', 'ref_max'])

# Minimum peak-above-local-floor, in units of the robust noise scale, for a
# curve-of-growth detection to be treated as a real feature rather than a noise
# maximum. Detections below this are flagged for dropping in `_growth_diameters`.
_PEAK_SIGMA = 3.0


def _geometric_odd_buckets(min_diameter, max_diameter, n):
    """`n` geometrically-spaced odd diameters spanning [min, max] (deduped)."""
    values = np.geomspace(min_diameter, max_diameter, n)
    odd = 2 * np.rint((values - 1) / 2.0).astype(int) + 1
    return np.unique(np.clip(odd, min_diameter, max_diameter))


def _axis_radius(diameter, aspect):
    """Per-axis integer refinement radius for a reference-axis ``diameter``.

    The reference radius ``diameter // 2`` is scaled by each axis' ``aspect``
    multiplier and rounded, giving the semi-axes of the elliptical mask. Clamped
    to at least 1 so a small axis never collapses to a degenerate zero radius.
    For isotropic ``aspect`` this reduces to ``(diameter // 2,) * ndim``.
    """
    r_ref = int(diameter) // 2
    return tuple(max(int(round(r_ref * a)), 1) for a in aspect)


def _growth_diameters(image, coords, sizes, ndim, edge_frac=0.1,
                      max_buckets=10):
    """Assign each feature an odd refinement diameter by a curve of growth.

    Accumulate the annular (ring) *mean intensity* outward from each coordinate;
    the feature's edge is the first radius past the peak ring where that mean,
    above the patch's own local floor, falls below ``edge_frac`` of the peak.
    This reads each particle's own extent from the image, so -- unlike a moment
    in a fixed large window -- it is not inflated by neighbours farther out, nor
    by duplicate peaks on the same particle (both give the same, correct extent).
    The returned diameter is the scalar reference scale, clamped at the top to
    ``max``; in 3D+ it is snapped to at most ``max_buckets`` geometrically-spaced
    values to bound the refinement-mask memory.

    A detection is dropped (flagged in the returned ``keep`` mask) when either its
    measured extent is below ``min`` (a sub-minimum peak, not promoted up to the
    smallest window) or its peak fails to rise a few noise sigma above its local
    floor (a noise maximum in an empty region). Without the latter test such
    detections size to the max window -- their profile never decays -- and, having
    a large exclusion zone and mass, delete genuine small neighbours during
    deduplication; this severely hurt recall in sparse, noisy images.

    For anisotropic (fixed-shape) features the radial index is measured in
    *normalised* coordinates -- each axis divided by its ``aspect`` ratio -- which
    maps the ellipse onto a circle, so the isotropic scan below applies unchanged
    and yields a single reference scale.

    Returns
    -------
    assigned : ndarray of int
        Per-coordinate reference diameter, clamped at the top to ``max``.
    keep : ndarray of bool
        False where the detection is sub-minimum or not significant above the
        noise floor (drop).
    """
    min_d = sizes.ref_min
    max_d = sizes.ref_max
    r_max = sizes.ref_max // 2  # reference radius (scale of the aspect==1 axes)
    aspect = sizes.aspect
    r_axis = sizes.r_max  # per-axis patch half-widths (elongated ellipse)
    # Integer radial index (in reference-axis units) of every pixel in the
    # per-axis patch. Dividing each axis by its aspect ratio normalises the
    # ellipse to a circle. Pixels beyond r_max (the patch corners) are dropped,
    # not folded into the last bin -- otherwise they pile neighbour signal into
    # ring[r_max].
    axes = [np.arange(-rr, rr + 1) for rr in r_axis]
    grids = np.meshgrid(*axes, indexing='ij')
    rint = np.rint(np.sqrt(sum((g.astype(float) / a) ** 2
                               for g, a in zip(grids, aspect)))
                   ).astype(int).ravel()
    inside = rint <= r_max
    rbin = rint[inside]
    ring_px = np.bincount(rbin, minlength=r_max + 1)  # pixel count per ring
    has_px = ring_px > 0
    # Robust background noise scale: the 84th percentile of the whole image.
    # After band-pass the background is clipped to a spike at zero, so this is ~0
    # for a clean image -- no detection is gated and low-noise sizing is unchanged
    # -- and rises to roughly the 1-sigma noise excursion when the background is
    # genuinely noisy. Unlike a median/MAD it does not collapse to zero once half
    # the pixels are zero (which happens for the larger band-pass kernels), and it
    # is not pulled up by the bright feature tail (a sparse minority of pixels).
    # Used below to reject detections whose peak never rises significantly above
    # their own local floor -- i.e. noise maxima in empty regions. These were
    # previously sized to the max window (their ring mass never decays) and, with a
    # large exclusion zone and mass, deleted genuine small neighbours during
    # deduplication.
    noise = float(np.percentile(image, 84))

    assigned = np.empty(len(coords), dtype=int)
    significant = np.empty(len(coords), dtype=bool)
    for i, c in enumerate(coords):
        window = tuple(slice(int(ci) - rr, int(ci) + rr + 1)
                       for ci, rr in zip(c, r_axis))
        vals = image[window].astype(float).ravel()[inside]
        ring = np.bincount(rbin, weights=vals, minlength=r_max + 1)
        # Per-pixel mean intensity at each radius. Using the mean rather than the
        # ring total keeps a flat, featureless noise patch flat in radius: the
        # total instead ramps up with the pixel count (~r^(ndim-1)) and is never
        # seen to decay, which sized noise to the max window.
        profile = ring / np.maximum(ring_px, 1)
        # Per-patch baseline: the lowest ring level in this patch -- the feature
        # skirt, or the local noise floor of a spurious detection. Subtracting it
        # (instead of one global mean) cancels the elevated floor of a locally
        # bright noise cluster, which a global mean underestimates.
        baseline = profile[has_px].min()
        prof = profile - baseline
        # Scan outward from the centre; the edge is the first radius where the
        # (baseline-subtracted) mean intensity drops below `edge_frac` of the peak
        # seen so far (the feature boundary), before any farther neighbour makes
        # it climb again.
        running_max = 0.0
        edge = r_max
        for r in range(r_max + 1):
            if not has_px[r]:
                continue
            if prof[r] > running_max:
                running_max = prof[r]
            elif prof[r] < edge_frac * running_max:
                edge = r
                break
        # Clamp only at the top; a sub-minimum extent is kept as-is and flagged
        # below (not promoted to min_d) so the caller can drop it.
        assigned[i] = min(2 * edge + 1, max_d)
        # A real feature towers over its skirt; a noise maximum barely clears its
        # local floor. Flag the latter for dropping rather than sizing it to max.
        significant[i] = running_max > _PEAK_SIGMA * noise

    keep = (assigned >= min_d) & significant

    if ndim >= 3:
        distinct = np.unique(assigned)
        if len(distinct) > max_buckets:
            centers = _geometric_odd_buckets(min_d, max_d, max_buckets)
            assigned = centers[np.argmin(
                np.abs(assigned[:, None] - centers[None, :]), axis=1)]
    return assigned.astype(int), keep


def _bucketed_refine(raw_image, image, positions, assigned, aspect,
                     max_iterations, engine, characterize, pos_columns):
    """Refine features grouped by their assigned diameter.

    Each group shares a single (per-axis) refinement radius, so it is refined in
    one fixed-radius ``refine_com`` call (reusing its cached masks). The radius is
    the bucket's reference diameter scaled by ``aspect`` per axis. Results are
    reassembled in the original coordinate order, with a ``diameter`` column
    reporting the reference-axis diameter.
    """
    order = np.arange(len(positions))
    parts = []
    for d in np.unique(assigned):
        in_bucket = assigned == d
        radius = _axis_radius(d, aspect)
        sub = refine_com(raw_image, image, radius, positions[in_bucket],
                         max_iterations=max_iterations, engine=engine,
                         characterize=characterize, pos_columns=pos_columns)
        sub['diameter'] = int(d)
        sub.index = order[in_bucket]
        parts.append(sub)
    refined = pandas_concat(parts).sort_index()
    refined.reset_index(drop=True, inplace=True)
    return refined
