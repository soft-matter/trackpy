import warnings
import logging

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from .utils import validate_tuple
from .masks import binary_mask
from .preprocessing import convert_to_int

logger = logging.getLogger(__name__)

# Exponent applied to pixel intensity when computing the collapse_flat centroid
# (poly mode only). >1 sharpens the weighting toward the bright peak, pulling the
# representative off dim shoulder maxima that skew a plain centroid.
_CENTROID_POWER = 2.0


def where_close(pos, separation, intensity=None):
    """ Returns indices of features that are closer than separation from other
    features. When intensity is given, the one with the lowest intensity is
    returned: else the most topleft is returned (to avoid randomness)"""
    if len(pos) == 0:
        return []
    separation = validate_tuple(separation, pos.shape[1])
    if any([s == 0 for s in separation]):
        return []
    # Rescale positions, so that pairs are identified below a distance
    # of 1.
    if isinstance(pos, pd.DataFrame):
        pos_rescaled = pos.values / separation
    else:
        pos_rescaled = pos / separation
    duplicates = cKDTree(pos_rescaled, 30).query_pairs(1 - 1e-7)
    if len(duplicates) == 0:
        return []
    index_0 = np.fromiter((x[0] for x in duplicates), dtype=int)
    index_1 = np.fromiter((x[1] for x in duplicates), dtype=int)
    if intensity is None:
        to_drop = np.where(np.sum(pos_rescaled[index_0], 1) >
                           np.sum(pos_rescaled[index_1], 1),
                           index_1, index_0)
    else:
        intensity = np.asarray(intensity)
        intensity_0 = intensity[index_0]
        intensity_1 = intensity[index_1]
        to_drop = np.where(intensity_0 > intensity_1, index_1, index_0)
        edge_cases = intensity_0 == intensity_1
        if np.any(edge_cases):
            index_0 = index_0[edge_cases]
            index_1 = index_1[edge_cases]
            to_drop[edge_cases] = np.where(np.sum(pos_rescaled[index_0], 1) >
                                           np.sum(pos_rescaled[index_1], 1),
                                           index_1, index_0)
    return np.unique(to_drop)


def drop_close(pos, separation, intensity=None):
    """ Removes features that are closer than separation from other features.
    When intensity is given, the one with the lowest intensity is dropped:
    else the most topleft is dropped (to avoid randomness)"""
    to_drop = where_close(pos, separation, intensity)
    return np.delete(pos, to_drop, axis=0)


def where_close_variable(pos, separations, intensity, aspect=None):
    """Indices of features closer than a per-feature separation.

    Like :func:`where_close`, but each feature carries its own minimum
    separation. A pair is a duplicate when its distance is below the *larger* of
    the two features' separations -- i.e. one feature falls inside the other's
    exclusion zone -- and the lower-intensity feature is returned. Intended for
    polydisperse features, where separation scales with size.

    Parameters
    ----------
    pos : ndarray (N, ndim)
        Feature positions.
    separations : ndarray (N,)
        Per-feature minimum separation, in reference-axis units.
    intensity : ndarray (N,)
        Per-feature intensity; the dimmer of each too-close pair is dropped.
    aspect : sequence of numbers, optional
        Per-axis shape multiplier for anisotropic (fixed-shape) features. When
        given, positions are divided by ``aspect`` before distances are computed,
        which maps each feature's elliptical exclusion zone onto a sphere of the
        (reference-unit) ``separations`` radius -- so the isotropic pairing logic
        applies unchanged. Defaults to isotropic.
    """
    pos = np.asarray(pos, dtype=float)
    if len(pos) == 0:
        return []

    if aspect is not None:
        pos = pos / np.asarray(aspect, dtype=float)

    separations = np.asarray(separations, dtype=float)
    max_sep = separations.max()
    if max_sep <= 0:
        return []

    # Any duplicate pair is within the largest separation; filter that superset
    # down to pairs closer than the larger feature's own separation.
    pairs = cKDTree(pos).query_pairs(max_sep)
    if len(pairs) == 0:
        return []

    index_0 = np.fromiter((p[0] for p in pairs), dtype=int)
    index_1 = np.fromiter((p[1] for p in pairs), dtype=int)
    dist = np.sqrt(np.sum((pos[index_0] - pos[index_1]) ** 2, axis=1))
    close = dist < np.maximum(separations[index_0], separations[index_1])
    index_0, index_1 = index_0[close], index_1[close]
    if len(index_0) == 0:
        return []

    intensity = np.asarray(intensity)
    to_drop = np.where(intensity[index_0] > intensity[index_1],
                       index_1, index_0)

    # Break intensity ties deterministically (drop the most bottom-right).
    ties = intensity[index_0] == intensity[index_1]
    if np.any(ties):
        i0, i1 = index_0[ties], index_1[ties]
        to_drop[ties] = np.where(np.sum(pos[i0], 1) > np.sum(pos[i1], 1),
                                 i0, i1)
    return np.unique(to_drop)


def percentile_threshold(image, percentile):
    """Find grayscale threshold based on distribution in image."""

    not_black = image[np.nonzero(image)]
    if len(not_black) == 0:
        return np.nan
    return np.percentile(not_black, percentile)


def grey_dilation(image, separation, percentile=64, margin=None, precise=True,
                  collapse_flat=False):
    """Find local maxima whose brightness is above a given percentile.

    Parameters
    ----------
    image : ndarray
        For best performance, provide an integer-type array. If the type is not
        of integer-type, the image will be normalized and coerced to uint8.
    separation : number or tuple of numbers
        Minimum separation between maxima. See precise for more information.
    percentile : float in range of [0,100], optional
        Features must have a peak brighter than pixels in this percentile.
        This helps eliminate spurious peaks. Default 64.
    margin : integer or tuple of integers, optional
        Zone of exclusion at edges of image. Default is ``separation / 2``.
    precise : boolean, optional
        Determines whether there will be an extra filtering step (``drop_close``)
        discarding features that are too close. Degrades performance.
        Because of the square kernel used, too many features are returned when
        precise=False. Default True.
    collapse_flat : boolean, optional
        When True, collapse each connected region of equal-valued maxima to a
        single representative (its centroid). A flat or plateaued peak -- common
        on large or saturated features, especially after integer coercion --
        otherwise reports every pixel of the plateau as a separate maximum. Two
        distinct features never merge, as they are separated by a lower-valued
        gap. Default False.

    See Also
    --------
    drop_close : removes features that are too close to brighter features
    grey_dilation_legacy : local maxima finding routine used until trackpy v0.3
    """
    # convert to integer. does nothing if image is already of integer type
    factor, image = convert_to_int(image, dtype=np.uint8)

    ndim = image.ndim
    separation = validate_tuple(separation, ndim)
    if margin is None:
        margin = tuple([int(s / 2) for s in separation])

    # Compute a threshold based on percentile.
    threshold = percentile_threshold(image, percentile)
    if np.isnan(threshold):
        warnings.warn("Image is completely black.", UserWarning)
        return np.empty((0, ndim))

    # Find the largest box that fits inside the ellipse given by separation
    size = [int(2 * s / np.sqrt(ndim)) for s in separation]

    # The intersection of the image with its dilation gives local maxima.
    dilation = ndimage.grey_dilation(image, size, mode='constant')
    maxima = (image == dilation) & (image > threshold)
    if np.sum(maxima) == 0:
        warnings.warn("Image contains no local maxima.", UserWarning)
        return np.empty((0, ndim))

    if collapse_flat:
        # Reduce each connected blob of maxima (e.g. a feature's flat top) to
        # one representative at the blob centroid, using full connectivity so a
        # plateau is never split into diagonally-adjacent fragments.
        structure = ndimage.generate_binary_structure(ndim, ndim)
        labels, count = ndimage.label(maxima, structure=structure)
        index = np.arange(1, count + 1)
        # Intensity-weighted centroid per blob, computed with bincount (far cheaper
        # than ndimage.center_of_mass, which loops over labels in Python). The
        # maxima pixels within a blob are not one flat plateau -- each is the max of
        # its own kernel, so their values differ -- and a maximum abutting a
        # brighter neighbour is clipped on that side, skewing an unweighted centroid
        # off the true peak (and thus the size measurement and refinement that start
        # from it). Weighting by image value pulls the representative back onto the
        # bright centre; for a genuinely flat top all weights are equal, so it
        # reduces to the plain centroid.
        blob_coords = np.argwhere(maxima)
        blob_label = labels[maxima]                   # label (1..count) per pixel
        blob_weight = image[maxima].astype(float) ** _CENTROID_POWER
        weight_sum = np.bincount(blob_label, weights=blob_weight,
                                 minlength=count + 1)[1:]
        centroid = np.column_stack([
            np.bincount(blob_label,
                        weights=blob_coords[:, d].astype(float) * blob_weight,
                        minlength=count + 1)[1:]
            for d in range(ndim)]) / weight_sum[:, None]
        pos = np.round(centroid).astype(int)
        intensity = np.asarray(ndimage.maximum(image, labels, index))
    else:
        pos = np.vstack(np.where(maxima)).T
        intensity = image[maxima]

    # Do not accept peaks near the edges.
    shape = np.array(image.shape)
    near_edge = np.any((pos < margin) | (pos > (shape - margin - 1)), 1)
    pos = pos[~near_edge]
    intensity = intensity[~near_edge]

    if len(pos) == 0:
        warnings.warn("All local maxima were in the margins.", UserWarning)
        return np.empty((0, ndim))

    # Remove local maxima that are too close to each other
    if precise:
        pos = drop_close(pos, separation, intensity)

    return pos


def grey_dilation_legacy(image, separation, percentile=64, margin=None):
    """Find local maxima whose brightness is above a given percentile.

    Parameters
    ----------
    separation : minimum separation between maxima
    percentile : chooses minimum greyscale value for a local maximum
    margin : zone of exclusion at edges of image. Defaults to radius.
            A smarter value is set by locate().

    See Also
    --------
    grey_dilation : faster local maxima finding routine
    """
    if margin is None:
        margin = separation

    ndim = image.ndim
    # Compute a threshold based on percentile.
    threshold = percentile_threshold(image, percentile)
    if np.isnan(threshold):
        warnings.warn("Image is completely black.", UserWarning)
        return np.empty((0, ndim))

    if not np.issubdtype(image.dtype, np.integer):
        factor = 255 / image.max()
        image = (factor * image.clip(min=0.)).astype(np.uint8)

    # The intersection of the image with its dilation gives local maxima
    footprint = binary_mask(separation, ndim)
    dilation = ndimage.grey_dilation(image, footprint=footprint,
                                     mode='constant')
    maxima = np.vstack(np.where((image == dilation) & (image > threshold))).T
    if not np.size(maxima) > 0:
        warnings.warn("Image contains no local maxima.", UserWarning)
        return np.empty((0, ndim))

    # Do not accept peaks near the edges.
    shape = np.array(image.shape)
    near_edge = np.any((maxima < margin) | (maxima > (shape - margin - 1)), 1)
    maxima = maxima[~near_edge]
    if not np.size(maxima) > 0:
        warnings.warn("All local maxima were in the margins.", UserWarning)

    # Return coords in as a numpy array shaped so it can be passed directly
    # to the DataFrame constructor.
    return maxima
