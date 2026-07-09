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
                "Polydisperse supports isotropic features only: {} entries "
                "must all be equal, got {}.".format(name, validated))
        return validated
    return _check_odd_int(value, name)


def _scalar(value):
    """Reduce a validated (isotropic) diameter to its scalar magnitude."""
    return value[0] if isinstance(value, tuple) else value


class Polydisperse:
    """Configuration for polydisperse feature finding.

    Parameters
    ----------
    min_diameter : odd int or tuple of equal odd ints
        Smallest feature diameter to detect. When in doubt, round up.
    max_diameter : odd int or tuple of equal odd ints
        Largest feature diameter to detect. Must be ``>= min_diameter``.
    edge_frac : float, optional
        Curve-of-growth boundary threshold in (0, 1), default 0.1. Each feature's
        refinement diameter is placed where its annular (ring) mass first falls
        below ``edge_frac`` of the peak ring -- i.e. where its radial intensity
        profile has decayed. Larger values give tighter windows.

    Notes
    -----
    Only isotropic features are supported: a tuple ``diameter`` must have equal
    entries, because a single scalar size per feature is required for bucketing.
    """

    def __init__(self, min_diameter, max_diameter, edge_frac=0.1):
        self.min_diameter = _validate_diameter(min_diameter, 'min_diameter')
        self.max_diameter = _validate_diameter(max_diameter, 'max_diameter')

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

    def resolve(self, ndim):
        """Broadcast/validate the diameters to `ndim`; derive the max radius."""
        min_diameter = validate_tuple(self.min_diameter, ndim)
        max_diameter = validate_tuple(self.max_diameter, ndim)
        return ResolvedPolydisperse(
            min_diameter=min_diameter,
            max_diameter=max_diameter,
            r_max=tuple(d // 2 for d in max_diameter))

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
        resolved = self.resolve(ndim)
        if pos_columns is None:
            pos_columns = default_pos_columns(ndim)

        if len(coords) == 0:
            empty = refine_com(raw_image, image, resolved.r_max, coords,
                               max_iterations=1, engine=engine,
                               characterize=characterize,
                               pos_columns=pos_columns)
            empty['diameter'] = np.empty(0, dtype=int)
            return empty

        coords = np.round(np.asarray(coords)).astype(int)
        assigned = _growth_diameters(image, coords, resolved, ndim,
                                     edge_frac=self.edge_frac)
        return _bucketed_refine(raw_image, image, coords, assigned,
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
        black_level, noise = measure_noise(image, raw_image, self.resolve(ndim).r_max)
        raw_mass = features['raw_mass'].values
        diameters = features['diameter'].values
        order = np.arange(len(features))
        ep_parts, idx_parts = [], []
        for d in np.unique(diameters):
            in_bucket = diameters == d
            radius = (int(d) // 2,) * ndim
            npx = N_binary_mask(radius, ndim)
            mass = raw_mass[in_bucket] - npx * black_level
            ep_parts.append(_static_error(mass, noise, radius, noise_size))
            idx_parts.append(order[in_bucket])
        ep = np.concatenate(ep_parts, axis=0)
        return ep[np.argsort(np.concatenate(idx_parts))]

    def __repr__(self):
        return ("Polydisperse(min_diameter={!r}, max_diameter={!r}, "
                "edge_frac={!r})".format(
            self.min_diameter, self.max_diameter, self.edge_frac))


# Size-range parameters resolved against a given image dimensionality
# (see Polydisperse.resolve).
ResolvedPolydisperse = namedtuple(
    'ResolvedPolydisperse', ['min_diameter', 'max_diameter', 'r_max'])


def _geometric_odd_buckets(min_diameter, max_diameter, n):
    """`n` geometrically-spaced odd diameters spanning [min, max] (deduped)."""
    values = np.geomspace(min_diameter, max_diameter, n)
    odd = 2 * np.rint((values - 1) / 2.0).astype(int) + 1
    return np.unique(np.clip(odd, min_diameter, max_diameter))


def _growth_diameters(image, coords, resolved, ndim, edge_frac=0.1,
                      max_buckets=10):
    """Assign each feature an odd refinement diameter by a curve of growth.

    Accumulate the annular (ring) mass outward from each coordinate; the
    feature's edge is the first radius past the peak ring where the ring mass
    falls below ``edge_frac`` of that peak. This reads each particle's own
    extent from the image, so -- unlike a moment in a fixed large window -- it is
    not inflated by neighbours farther out, nor by duplicate peaks on the same
    particle (both give the same, correct extent). Diameters are clamped to
    ``[min, max]``; in 3D+ they are snapped to at most ``max_buckets``
    geometrically-spaced values to bound the refinement-mask memory.
    """
    min_d = resolved.min_diameter[0]
    max_d = resolved.max_diameter[0]
    r_max = resolved.r_max[0]
    # Integer radial index of every pixel in the (2*r_max+1)^ndim patch. Pixels
    # beyond r_max (the patch corners) are dropped, not folded into the last bin
    # -- otherwise they pile neighbour signal into ring[r_max].
    axes = [np.arange(-r_max, r_max + 1)] * ndim
    rint = np.rint(np.sqrt(sum(g.astype(float) ** 2
                               for g in np.meshgrid(*axes, indexing='ij')))
                   ).astype(int).ravel()
    inside = rint <= r_max
    rbin = rint[inside]

    assigned = np.empty(len(coords), dtype=int)
    for i, c in enumerate(coords):
        window = tuple(slice(int(ci) - r_max, int(ci) + r_max + 1) for ci in c)
        vals = image[window].astype(float).ravel()[inside]
        ring = np.bincount(rbin, weights=vals, minlength=r_max + 1)
        # Scan outward from the centre; the edge is the first radius where the
        # ring mass drops below `edge_frac` of the peak seen so far (the feature
        # boundary), before any farther neighbour makes it climb again.
        running_max = 0.0
        edge = r_max
        for r in range(r_max + 1):
            if ring[r] > running_max:
                running_max = ring[r]
            elif ring[r] < edge_frac * running_max:
                edge = r
                break
        assigned[i] = min(max(2 * edge + 1, min_d), max_d)

    if ndim >= 3:
        distinct = np.unique(assigned)
        if len(distinct) > max_buckets:
            centers = _geometric_odd_buckets(min_d, max_d, max_buckets)
            assigned = centers[np.argmin(
                np.abs(assigned[:, None] - centers[None, :]), axis=1)]
    return assigned.astype(int)


def _bucketed_refine(raw_image, image, positions, assigned, max_iterations,
                     engine, characterize, pos_columns):
    """Refine features grouped by their assigned diameter.

    Each group shares a single refinement radius, so it is refined in one
    fixed-radius ``refine_com`` call (reusing its cached masks). Results are
    reassembled in the original coordinate order, with a ``diameter`` column.
    """
    ndim = image.ndim
    order = np.arange(len(positions))
    parts = []
    for d in np.unique(assigned):
        in_bucket = assigned == d
        radius = (int(d) // 2,) * ndim
        sub = refine_com(raw_image, image, radius, positions[in_bucket],
                         max_iterations=max_iterations, engine=engine,
                         characterize=characterize, pos_columns=pos_columns)
        sub['diameter'] = int(d)
        sub.index = order[in_bucket]
        parts.append(sub)
    refined = pandas_concat(parts).sort_index()
    refined.reset_index(drop=True, inplace=True)
    return refined
