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

# Sigma multiple for the refinement-window half-width (R = k * sigma) used to
# derive the default ``rg_to_diameter`` for a Gaussian intensity profile.
# k = 2.5 captures ~95.6% of a 2D Gaussian's mass (~89.9% in 3D).
DEFAULT_RG_K = 2.5


def gaussian_rg_to_diameter(ndim, k=DEFAULT_RG_K):
    """Default factor converting radius of gyration to refinement diameter.

    For an isotropic Gaussian the radius of gyration is ``Rg = sigma *
    sqrt(ndim)``; choosing a window half-width ``R = k * sigma`` gives
    ``diameter / Rg = 2 * k / sqrt(ndim)``.
    """
    return float(2.0 * k / np.sqrt(ndim))


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
    rg_to_diameter : float, optional
        Factor converting a feature's radius of gyration (the ``size`` output)
        to the diameter of the refinement window used to characterize it. If
        None (default), a Gaussian-profile default ``2 * k / sqrt(ndim)`` with
        ``k = 2.5`` is resolved from the image dimensionality inside ``locate``
        (2D: ~3.54, 3D: ~2.89).
    max_radius_iterations : int, optional
        Number of times to (re-)estimate each feature's size and assign its
        refinement radius. Default 1 (single assignment, no iteration). Values
        > 1 help in dense mixed fields where a small particle adjacent to a
        large bright one can initially be assigned too large a radius.

    Notes
    -----
    Only isotropic features are supported: a tuple ``diameter`` must have equal
    entries, because a single scalar size per feature is required for bucketing.
    """

    def __init__(self, min_diameter, max_diameter, rg_to_diameter=None,
                 max_radius_iterations=1):
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

        if rg_to_diameter is not None:
            if (isinstance(rg_to_diameter, bool)
                    or not isinstance(rg_to_diameter, numbers.Real)
                    or rg_to_diameter <= 0):
                raise ValueError("rg_to_diameter must be a positive number or "
                                 "None, got {!r}.".format(rg_to_diameter))
            rg_to_diameter = float(rg_to_diameter)
        self.rg_to_diameter = rg_to_diameter

        if (isinstance(max_radius_iterations, bool)
                or not isinstance(max_radius_iterations, numbers.Integral)
                or max_radius_iterations < 1):
            raise ValueError("max_radius_iterations must be an integer >= 1, "
                             "got {!r}.".format(max_radius_iterations))
        self.max_radius_iterations = int(max_radius_iterations)

    def resolve_rg_to_diameter(self, ndim):
        """Explicit ``rg_to_diameter``, or the Gaussian default for ``ndim``."""
        if self.rg_to_diameter is not None:
            return self.rg_to_diameter
        return gaussian_rg_to_diameter(ndim)

    def resolve(self, ndim):
        """Resolve the size parameters against an image dimensionality.

        Broadcasts/validates the min and max diameters to `ndim`, derives the
        min/max refinement radii, and resolves the (possibly dimension-
        dependent) ``rg_to_diameter``.
        """
        min_diameter = validate_tuple(self.min_diameter, ndim)
        max_diameter = validate_tuple(self.max_diameter, ndim)
        return ResolvedPolydisperse(
            min_diameter=min_diameter,
            max_diameter=max_diameter,
            r_min=tuple(d // 2 for d in min_diameter),
            r_max=tuple(d // 2 for d in max_diameter),
            rg_to_diameter=self.resolve_rg_to_diameter(ndim),
            max_radius_iterations=self.max_radius_iterations)

    def refine(self, raw_image, image, coords, max_iterations=10,
               engine='auto', characterize=True, pos_columns=None):
        """Refine and characterize features spanning a range of sizes.

        A single-iteration pass at the largest window bootstraps a size estimate
        for each detected coordinate; each feature is then assigned an odd
        refinement diameter from its radius of gyration (via ``rg_to_diameter``)
        and refined in buckets that share a diameter. Returns the refined
        features with an assigned ``diameter`` column, in the original
        coordinate order.
        """
        ndim = image.ndim
        resolved = self.resolve(ndim)
        min_d, max_d = resolved.min_diameter[0], resolved.max_diameter[0]
        rg_to_diameter = resolved.rg_to_diameter
        if pos_columns is None:
            pos_columns = default_pos_columns(ndim)

        if len(coords) == 0:
            empty = refine_com(raw_image, image, resolved.r_max, coords,
                               max_iterations=1, engine=engine,
                               characterize=characterize,
                               pos_columns=pos_columns)
            empty['diameter'] = np.empty(0, dtype=int)
            return empty

        # Bootstrap: one iteration at the largest window gives a size (radius of
        # gyration) estimate and a starting position for every feature.
        boot = refine_com(raw_image, image, resolved.r_max, coords,
                          max_iterations=1, engine=engine, characterize=True,
                          pos_columns=pos_columns)
        positions = boot[pos_columns].values
        assigned = _assign_diameters(boot['size'].values, rg_to_diameter,
                                     min_d, max_d, ndim)
        refined = _bucketed_refine(raw_image, image, positions, assigned,
                                   max_iterations, engine, characterize,
                                   pos_columns)

        # Optionally re-estimate size at the assigned window and re-refine until
        # the diameter assignment stabilizes. Needs the size column
        # (characterize).
        if characterize:
            for _ in range(resolved.max_radius_iterations - 1):
                reassigned = _assign_diameters(refined['size'].values,
                                               rg_to_diameter, min_d, max_d,
                                               ndim)
                if np.array_equal(reassigned, refined['diameter'].values):
                    break
                refined = _bucketed_refine(raw_image, image,
                                           refined[pos_columns].values,
                                           reassigned, max_iterations, engine,
                                           characterize, pos_columns)
        return refined

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
                "rg_to_diameter={!r}, max_radius_iterations={!r})".format(
                    self.min_diameter, self.max_diameter, self.rg_to_diameter,
                    self.max_radius_iterations))


# Size-range parameters resolved against a given image dimensionality.
# See `_resolve_polydisperse`.
ResolvedPolydisperse = namedtuple(
    'ResolvedPolydisperse',
    ['min_diameter', 'max_diameter', 'r_min', 'r_max', 'rg_to_diameter',
     'max_radius_iterations'])


def _geometric_odd_buckets(min_diameter, max_diameter, n):
    """`n` geometrically-spaced odd diameters spanning [min, max] (deduped)."""
    values = np.geomspace(min_diameter, max_diameter, n)
    odd = 2 * np.rint((values - 1) / 2.0).astype(int) + 1
    return np.unique(np.clip(odd, min_diameter, max_diameter))


def _assign_diameters(sizes, rg_to_diameter, min_diameter, max_diameter, ndim,
                      max_buckets=10):
    """Map radius-of-gyration estimates to odd refinement diameters.

    Each size is scaled by ``rg_to_diameter``, rounded to the nearest odd
    integer and clamped to ``[min_diameter, max_diameter]``. In 3D and higher,
    if more than ``max_buckets`` distinct diameters result they are snapped to
    that many geometrically-spaced values, bounding the number of refinement
    masks (and their memory) that must be built.
    """
    diameters = rg_to_diameter * np.asarray(sizes, dtype=float)
    # Non-finite sizes (e.g. from zero mass) fall back to the smallest diameter.
    diameters = np.where(np.isfinite(diameters), diameters, min_diameter)
    assigned = 2 * np.rint((diameters - 1) / 2.0).astype(int) + 1
    assigned = np.clip(assigned, min_diameter, max_diameter)

    if ndim >= 3:
        distinct = np.unique(assigned)
        if len(distinct) > max_buckets:
            centers = _geometric_odd_buckets(min_diameter, max_diameter,
                                             max_buckets)
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
