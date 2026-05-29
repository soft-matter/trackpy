import numpy as np
import pandas as pd
from ..try_numba import try_numba_jit

import warnings
import logging

from ..utils import (validate_tuple, guess_pos_columns, default_pos_columns,
                     default_size_columns)
from ..masks import (binary_mask, r_squared_mask,
                     x_squared_masks, cosmask, sinmask)

from ..try_numba import NUMBA_AVAILABLE, int, round


logger = logging.getLogger(__name__)


def _safe_center_of_mass(x, radius, grids):
    normalizer = x.sum()
    if normalizer == 0:  # avoid divide-by-zero errors
        return np.array(radius)
    return np.array([(x * grids[dim]).sum() / normalizer
                    for dim in range(x.ndim)])


def refine_com(raw_image, image, radius, coords, max_iterations=10,
               engine='auto', shift_thresh=0.6, characterize=True,
               pos_columns=None):
    """Find the center of mass of a bright feature starting from an estimate.

    Characterize the neighborhood of a local maximum, and iteratively
    hone in on its center-of-brightness.

    Parameters
    ----------
    raw_image : array (any dimensions)
        Image used for final characterization. Ideally, pixel values of
        this image are not rescaled, but it can also be identical to
        ``image``.
    image : array (same size as raw_image)
        Processed image used for centroid-finding and most particle
        measurements.
    coords : array or DataFrame
        estimated position
    max_iterations : integer
        max number of loops to refine the center of mass, default 10
    engine : {'python', 'numba'}
        Numba is faster if available, but it cannot do walkthrough.
    shift_thresh : float, optional
        Default 0.6 (unit is pixels).
        If the brightness centroid is more than this far off the mask center,
        shift mask to neighboring pixel. The new mask will be used for any
        remaining iterations.
    characterize : boolean, True by default
        Compute and return mass, size, eccentricity, signal.
    pos_columns: list of strings, optional
        Column names that contain the position coordinates.
        Defaults to ``['y', 'x']`` or ``['z', 'y', 'x']``, if ``'z'`` exists.

    Returns
    -------
    DataFrame([x, y, mass, size, ecc, signal, raw_mass])
        where "x, y" are appropriate to the dimensionality of the image,
        mass means total integrated brightness of the blob,
        size means the radius of gyration of its Gaussian-like profile,
        ecc is its eccentricity (0 is circular),
        and raw_mass is the total integrated brightness in raw_image.
    """
    if isinstance(coords, pd.DataFrame):
        if pos_columns is None:
            pos_columns = guess_pos_columns(coords)
        index = coords.index
        coords = coords[pos_columns].values
    else:
        index = None

    radius = validate_tuple(radius, image.ndim)

    if pos_columns is None:
        pos_columns = default_pos_columns(image.ndim)
    columns = pos_columns + ['mass']
    if characterize:
        isotropic = radius[1:] == radius[:-1]
        columns += default_size_columns(image.ndim, isotropic) + \
            ['ecc', 'signal', 'raw_mass']

    if len(coords) == 0:
        return pd.DataFrame(columns=columns)

    refined = refine_com_arr(raw_image, image, radius, coords,
                             max_iterations=max_iterations,
                             engine=engine, shift_thresh=shift_thresh,
                             characterize=characterize)

    return pd.DataFrame(refined, columns=columns, index=index)


def refine_com_arr(raw_image, image, radius, coords, max_iterations=10,
                   engine='auto', shift_thresh=0.6, characterize=True,
                   walkthrough=False):
    """Refine coordinates and return a numpy array instead of a DataFrame.

    See also
    --------
    refine_com
    """
    if max_iterations <= 0:
        warnings.warn("max_iterations has to be larger than 0. setting it to 1.")
        max_iterations = 1
    if raw_image.ndim != coords.shape[1]:
        raise ValueError("The image has a different number of dimensions than "
                         "the coordinate array.")

    # ensure that radius is tuple of integers, for direct calls to refine_com_arr()
    radius = validate_tuple(radius, image.ndim)
    # Main loop will be performed in separate function.
    if engine == 'auto':
        if NUMBA_AVAILABLE and image.ndim in [2, 3]:
            engine = 'numba'
        else:
            engine = 'python'

    # In here, coord is an integer. Make a copy, will not modify inplace.
    coords = np.round(coords).astype(int)

    if engine == 'python':
        results = _refine(raw_image, image, radius, coords, max_iterations,
                          shift_thresh, characterize, walkthrough)
    elif engine == 'numba':
        if not NUMBA_AVAILABLE:
            warnings.warn("numba could not be imported. Without it, the "
                          "'numba' engine runs very slow. Use the 'python' "
                          "engine or install numba.", UserWarning)
        if image.ndim not in [2, 3]:
            raise NotImplementedError("The numba engine only supports 2D or 3D "
                                      "images. You can extend it if you feel "
                                      "like a hero.")
        if walkthrough:
            raise ValueError("walkthrough is not availabe in the numba engine")
        # Do some extra prep in pure Python that can't be done in numba.
        N = coords.shape[0]
        mask = binary_mask(radius, image.ndim)
        if image.ndim == 3:
            if characterize:
                if np.all(radius[1:] == radius[:-1]):
                    results_columns = 8
                else:
                    results_columns = 10
            else:
                results_columns = 4
            r2_mask = r_squared_mask(radius, image.ndim)[mask]
            x2_masks = x_squared_masks(radius, image.ndim)
            z2_mask = image.ndim * x2_masks[0][mask]
            y2_mask = image.ndim * x2_masks[1][mask]
            x2_mask = image.ndim * x2_masks[2][mask]
            results = np.empty((N, results_columns), dtype=np.float64)
            maskZ, maskY, maskX = np.asarray(np.asarray(mask.nonzero()),
                                             dtype=np.int16)
            _numba_refine_3D(np.asarray(raw_image), np.asarray(image),
                             radius[0], radius[1], radius[2], coords, N,
                             int(max_iterations), shift_thresh,
                             characterize,
                             image.shape[0], image.shape[1], image.shape[2],
                             maskZ, maskY, maskX, maskX.shape[0],
                             r2_mask, z2_mask, y2_mask, x2_mask, results)
        elif not characterize:
            mask_coordsY, mask_coordsX = np.asarray(mask.nonzero(), np.int16)
            results = np.empty((N, 3), dtype=np.float64)
            _numba_refine_2D(np.asarray(image), radius[0], radius[1], coords, N,
                             int(max_iterations), shift_thresh,
                             image.shape[0], image.shape[1],
                             mask_coordsY, mask_coordsX, mask_coordsY.shape[0],
                             results)
        elif radius[0] == radius[1]:
            mask_coordsY, mask_coordsX = np.asarray(mask.nonzero(), np.int16)
            results = np.empty((N, 7), dtype=np.float64)
            r2_mask = r_squared_mask(radius, image.ndim)[mask]
            cmask = cosmask(radius)[mask]
            smask = sinmask(radius)[mask]
            _numba_refine_2D_c(np.asarray(raw_image), np.asarray(image),
                               radius[0], radius[1], coords, N,
                               int(max_iterations), shift_thresh,
                               image.shape[0], image.shape[1], mask_coordsY,
                               mask_coordsX, mask_coordsY.shape[0],
                               r2_mask, cmask, smask, results)
        else:
            mask_coordsY, mask_coordsX = np.asarray(mask.nonzero(), np.int16)
            results = np.empty((N, 8), dtype=np.float64)
            x2_masks = x_squared_masks(radius, image.ndim)
            y2_mask = image.ndim * x2_masks[0][mask]
            x2_mask = image.ndim * x2_masks[1][mask]
            cmask = cosmask(radius)[mask]
            smask = sinmask(radius)[mask]
            _numba_refine_2D_c_a(np.asarray(raw_image), np.asarray(image),
                                 radius[0], radius[1], coords, N,
                                 int(max_iterations), shift_thresh,
                                 image.shape[0], image.shape[1], mask_coordsY,
                                 mask_coordsX, mask_coordsY.shape[0],
                                 y2_mask, x2_mask, cmask, smask, results)
    else:
        raise ValueError("Available engines are 'python' and 'numba'")

    return results


# (This is pure Python. A numba variant follows below.)
def _refine(raw_image, image, radius, coords, max_iterations,
            shift_thresh, characterize, walkthrough):
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError('The coords array should be of integer datatype')
    ndim = image.ndim
    isotropic = np.all(radius[1:] == radius[:-1])
    mask = binary_mask(radius, ndim).astype(np.uint8)

    # Declare arrays that we will fill iteratively through loop.
    N = coords.shape[0]
    final_coords = np.empty_like(coords, dtype=np.float64)
    mass = np.empty(N, dtype=np.float64)
    raw_mass = np.empty(N, dtype=np.float64)
    if characterize:
        if isotropic:
            Rg = np.empty(N, dtype=np.float64)
        else:
            Rg = np.empty((N, len(radius)), dtype=np.float64)
        ecc = np.empty(N, dtype=np.float64)
        signal = np.empty(N, dtype=np.float64)

    ogrid = np.ogrid[[slice(0, i) for i in mask.shape]]  # for center of mass
    ogrid = [g.astype(float) for g in ogrid]

    for feat, coord in enumerate(coords):
        for iteration in range(max_iterations):
            # Define the circular neighborhood of (x, y).
            rect = tuple([slice(c - r, c + r + 1)
                          for c, r in zip(coord, radius)])
            neighborhood = mask * image[rect]
            cm_n = _safe_center_of_mass(neighborhood, radius, ogrid)
            cm_i = cm_n - radius + coord  # image coords

            off_center = cm_n - radius
            logger.debug('off_center: %f', off_center)
            if np.all(np.abs(off_center) < shift_thresh):
                break  # Accurate enough.
            # If we're off by more than half a pixel in any direction, move..
            coord[off_center > shift_thresh] += 1
            coord[off_center < -shift_thresh] -= 1
            # Don't move outside the image!
            upper_bound = np.array(image.shape) - 1 - radius
            coord = np.clip(coord, radius, upper_bound).astype(int)

        # stick to yx column order
        final_coords[feat] = cm_i

        if walkthrough:
            import matplotlib.pyplot as plt
            plt.imshow(neighborhood)

        # Characterize the neighborhood of our final centroid.
        mass[feat] = neighborhood.sum()
        if not characterize:
            continue  # short-circuit loop
        if isotropic:
            Rg[feat] = np.sqrt(np.sum(r_squared_mask(radius, ndim) *
                                      neighborhood) / mass[feat])
        else:
            Rg[feat] = np.sqrt(ndim * np.sum(x_squared_masks(radius, ndim) *
                                             neighborhood,
                                             axis=tuple(range(1, ndim + 1))) /
                               mass[feat])
        # I only know how to measure eccentricity in 2D.
        if ndim == 2:
            ecc[feat] = np.sqrt(np.sum(neighborhood*cosmask(radius))**2 +
                                np.sum(neighborhood*sinmask(radius))**2)
            ecc[feat] /= (mass[feat] - neighborhood[radius] + 1e-6)
        else:
            ecc[feat] = np.nan
        signal[feat] = neighborhood.max()  # based on bandpassed image
        raw_neighborhood = mask * raw_image[rect]
        raw_mass[feat] = raw_neighborhood.sum()  # based on raw image

    if not characterize:
        return np.column_stack([final_coords, mass])
    else:
        return np.column_stack([final_coords, mass, Rg, ecc, signal, raw_mass])


@try_numba_jit(nopython=True)
def _numba_refine_2D(image, radiusY, radiusX, coords, N, max_iterations,
                     shift_thresh, shapeY, shapeX, maskY, maskX, N_mask,
                     results):
    # Column indices into the 'results' array
    MASS_COL = 2

    upper_boundY = shapeY - radiusY - 1
    upper_boundX = shapeX - radiusX - 1

    for feat in range(N):
        # coord is an integer.
        coordY = coords[feat, 0]
        coordX = coords[feat, 1]
        for iteration in range(max_iterations):
            # Define the circular neighborhood of (x, y).
            cm_nY = 0.
            cm_nX = 0.
            squareY = coordY - radiusY
            squareX = coordX - radiusX
            mass_ = 0.0
            for i in range(N_mask):
                px = image[squareY + maskY[i],
                           squareX + maskX[i]]
                cm_nY += px*maskY[i]
                cm_nX += px*maskX[i]
                mass_ += px

            cm_nY /= mass_
            cm_nX /= mass_
            cm_iY = cm_nY - radiusY + coordY
            cm_iX = cm_nX - radiusX + coordX

            off_centerY = cm_nY - radiusY
            off_centerX = cm_nX - radiusX
            if (abs(off_centerY) < shift_thresh and
                abs(off_centerX) < shift_thresh):
                break  # Go to next feature

            # If we're off by more than half a pixel in any direction, move.
            oc = off_centerY
            if oc > shift_thresh:
                coordY += 1
            elif oc < - shift_thresh:
                coordY -= 1
            oc = off_centerX
            if oc > shift_thresh:
                coordX += 1
            elif oc < - shift_thresh:
                coordX -= 1
            # Don't move outside the image!
            if coordY < radiusY:
                coordY = radiusY
            if coordX < radiusX:
                coordX = radiusX
            if coordY > upper_boundY:
                coordY = upper_boundY
            if coordX > upper_boundX:
                coordX = upper_boundX

        # use yx order
        results[feat, 0] = cm_iY
        results[feat, 1] = cm_iX

        # Characterize the neighborhood of our final centroid.
        results[feat, MASS_COL] = mass_

    return 0  # Unused

@try_numba_jit(nopython=True)
def _numba_refine_2D_c(raw_image, image, radiusY, radiusX, coords, N,
                       max_iterations, shift_thresh, shapeY, shapeX, maskY,
                       maskX, N_mask, r2_mask, cmask, smask, results):
    # Column indices into the 'results' array
    MASS_COL = 2
    RG_COL = 3
    ECC_COL = 4
    SIGNAL_COL = 5
    RAW_MASS_COL = 6

    upper_boundY = shapeY - radiusY - 1
    upper_boundX = shapeX - radiusX - 1

    for feat in range(N):
        # coord is an integer.
        coordY = coords[feat, 0]
        coordX = coords[feat, 1]

        for iteration in range(max_iterations):
            # Define the circular neighborhood of (x, y).
            cm_nY = 0.
            cm_nX = 0.
            squareY = coordY - radiusY
            squareX = coordX - radiusX
            mass_ = 0.0
            for i in range(N_mask):
                px = image[squareY + maskY[i],
                           squareX + maskX[i]]
                cm_nY += px*maskY[i]
                cm_nX += px*maskX[i]
                mass_ += px

            cm_nY /= mass_
            cm_nX /= mass_
            cm_iY = cm_nY - radiusY + coordY
            cm_iX = cm_nX - radiusX + coordX

            off_centerY = cm_nY - radiusY
            off_centerX = cm_nX - radiusX
            if (abs(off_centerY) < shift_thresh and
                abs(off_centerX) < shift_thresh):
                break  # Go to next feature

            oc = off_centerY
            if oc > shift_thresh:
                coordY += 1
            elif oc < - shift_thresh:
                coordY -= 1
            oc = off_centerX
            if oc > shift_thresh:
                coordX += 1
            elif oc < - shift_thresh:
                coordX -= 1
            # Don't move outside the image!
            if coordY < radiusY:
                coordY = radiusY
            if coordX < radiusX:
                coordX = radiusX
            if coordY > upper_boundY:
                coordY = upper_boundY
            if coordX > upper_boundX:
                coordX = upper_boundX

        # use yx order
        results[feat, 0] = cm_iY
        results[feat, 1] = cm_iX

        # Characterize the neighborhood of our final centroid.
        raw_mass_ = 0.
        Rg_ = 0.
        ecc1 = 0.
        ecc2 = 0.
        signal_ = 0.

        for i in range(N_mask):
            px = image[squareY + maskY[i],
                       squareX + maskX[i]]
            Rg_ += r2_mask[i]*px
            ecc1 += cmask[i]*px
            ecc2 += smask[i]*px
            raw_mass_ += raw_image[squareY + maskY[i],
                                   squareX + maskX[i]]
            if px > signal_:
                signal_ = px
        results[feat, RG_COL] = np.sqrt(Rg_/mass_)
        results[feat, MASS_COL] = mass_
        center_px = image[squareY + radiusY, squareX + radiusX]
        results[feat, ECC_COL] = np.sqrt(ecc1**2 + ecc2**2) / (mass_ - center_px + 1e-6)
        results[feat, SIGNAL_COL] = signal_
        results[feat, RAW_MASS_COL] = raw_mass_

    return 0  # Unused


@try_numba_jit(nopython=True)
def _numba_refine_2D_c_a(raw_image, image, radiusY, radiusX, coords, N,
                         max_iterations, shift_thresh, shapeY, shapeX, maskY,
                         maskX, N_mask, y2_mask, x2_mask, cmask, smask,
                         results):
    # Column indices into the 'results' array
    MASS_COL = 2
    RGY_COL = 3
    RGX_COL = 4
    ECC_COL = 5
    SIGNAL_COL = 6
    RAW_MASS_COL = 7

    upper_boundY = shapeY - radiusY - 1
    upper_boundX = shapeX - radiusX - 1

    for feat in range(N):
        # coord is an integer.
        coordY = coords[feat, 0]
        coordX = coords[feat, 1]

        for iteration in range(max_iterations):
            # Define the circular neighborhood of (x, y).
            cm_nY = 0.
            cm_nX = 0.
            squareY = coordY - radiusY
            squareX = coordX - radiusX
            mass_ = 0.0
            for i in range(N_mask):
                px = image[squareY + maskY[i],
                           squareX + maskX[i]]
                cm_nY += px*maskY[i]
                cm_nX += px*maskX[i]
                mass_ += px

            cm_nY /= mass_
            cm_nX /= mass_
            cm_iY = cm_nY - radiusY + coordY
            cm_iX = cm_nX - radiusX + coordX

            off_centerY = cm_nY - radiusY
            off_centerX = cm_nX - radiusX
            if (abs(off_centerY) < shift_thresh and
                abs(off_centerX) < shift_thresh):
                break  # Go to next feature

            # If we're off by more than half a pixel in any direction, move.
            oc = off_centerY
            if oc > shift_thresh:
                coordY += 1
            elif oc < - shift_thresh:
                coordY -= 1
            oc = off_centerX
            if oc > shift_thresh:
                coordX += 1
            elif oc < - shift_thresh:
                coordX -= 1
            # Don't move outside the image!
            if coordY < radiusY:
                coordY = radiusY
            if coordX < radiusX:
                coordX = radiusX
            if coordY > upper_boundY:
                coordY = upper_boundY
            if coordX > upper_boundX:
                coordX = upper_boundX
            # Update slice to shifted position.

        # use yx order
        results[feat, 0] = cm_iY
        results[feat, 1] = cm_iX

        # Characterize the neighborhood of our final centroid.
        mass_ = 0.
        raw_mass_ = 0.
        RgY = 0.
        RgX = 0.
        ecc1 = 0.
        ecc2 = 0.
        signal_ = 0.

        for i in range(N_mask):
            px = image[squareY + maskY[i],
                       squareX + maskX[i]]
            mass_ += px

            RgY += y2_mask[i]*px
            RgX += x2_mask[i]*px
            ecc1 += cmask[i]*px
            ecc2 += smask[i]*px
            raw_mass_ += raw_image[squareY + maskY[i],
                                   squareX + maskX[i]]
            if px > signal_:
                signal_ = px
        results[feat, RGY_COL] = np.sqrt(RgY/mass_)
        results[feat, RGX_COL] = np.sqrt(RgX/mass_)
        results[feat, MASS_COL] = mass_
        center_px = image[squareY + radiusY, squareX + radiusX]
        results[feat, ECC_COL] = np.sqrt(ecc1**2 + ecc2**2) / (mass_ - center_px + 1e-6)
        results[feat, SIGNAL_COL] = signal_
        results[feat, RAW_MASS_COL] = raw_mass_

    return 0  # Unused


@try_numba_jit(nopython=True)
def _numba_refine_3D(raw_image, image, radiusZ, radiusY, radiusX, coords, N,
                     max_iterations, shift_thresh, characterize, shapeZ, shapeY,
                     shapeX, maskZ, maskY, maskX, N_mask, r2_mask, z2_mask,
                     y2_mask, x2_mask, results):
    # Column indices into the 'results' array
    MASS_COL = 3
    isotropic = (radiusX == radiusY and radiusX == radiusZ)
    if isotropic:
        RG_COL = 4
        ECC_COL = 5
        SIGNAL_COL = 6
        RAW_MASS_COL = 7
    else:
        RGZ_COL = 4
        RGY_COL = 5
        RGX_COL = 6
        ECC_COL = 7
        SIGNAL_COL = 8
        RAW_MASS_COL = 9

    upper_boundZ = shapeZ - radiusZ - 1
    upper_boundY = shapeY - radiusY - 1
    upper_boundX = shapeX - radiusX - 1

    for feat in range(N):
        # coord is an integer.
        coordZ = coords[feat, 0]
        coordY = coords[feat, 1]
        coordX = coords[feat, 2]

        for iteration in range(max_iterations):
            # Define the neighborhood of (x, y, z).
            cm_nZ = 0.
            cm_nY = 0.
            cm_nX = 0.
            squareZ = int(round(coordZ)) - radiusZ
            squareY = int(round(coordY)) - radiusY
            squareX = int(round(coordX)) - radiusX
            mass_ = 0.0
            for i in range(N_mask):
                px = image[squareZ + maskZ[i],
                           squareY + maskY[i],
                           squareX + maskX[i]]
                cm_nZ += px*maskZ[i]
                cm_nY += px*maskY[i]
                cm_nX += px*maskX[i]
                mass_ += px

            cm_nZ /= mass_
            cm_nY /= mass_
            cm_nX /= mass_
            cm_iZ = cm_nZ - radiusZ + coordZ
            cm_iY = cm_nY - radiusY + coordY
            cm_iX = cm_nX - radiusX + coordX

            off_centerZ = cm_nZ - radiusZ
            off_centerY = cm_nY - radiusY
            off_centerX = cm_nX - radiusX
            if (abs(off_centerZ) < shift_thresh and
                abs(off_centerY) < shift_thresh and
                abs(off_centerX) < shift_thresh):
                break  # Go to next feature

            if off_centerZ > shift_thresh:
                coordZ += 1
            elif off_centerZ < - shift_thresh:
                coordZ -= 1
            if off_centerY > shift_thresh:
                coordY += 1
            elif off_centerY < - shift_thresh:
                coordY -= 1
            if off_centerX > shift_thresh:
                coordX += 1
            elif off_centerX < - shift_thresh:
                coordX -= 1
            # Don't move outside the image!
            if coordZ < radiusZ:
                coordZ = radiusZ
            if coordY < radiusY:
                coordY = radiusY
            if coordX < radiusX:
                coordX = radiusX
            if coordZ > upper_boundZ:
                coordZ = upper_boundZ
            if coordY > upper_boundY:
                coordY = upper_boundY
            if coordX > upper_boundX:
                coordX = upper_boundX

        # use zyx order
        results[feat, 0] = cm_iZ
        results[feat, 1] = cm_iY
        results[feat, 2] = cm_iX

        # Characterize the neighborhood of our final centroid.
        raw_mass_ = 0.
        Rg_ = 0.
        RgZ = 0.
        RgY = 0.
        RgX = 0.
        signal_ = 0.

        if not characterize:
            pass
        elif isotropic:
            for i in range(N_mask):
                px = image[squareZ + maskZ[i],
                           squareY + maskY[i],
                           squareX + maskX[i]]

                Rg_ += r2_mask[i]*px
                raw_mass_ += raw_image[squareZ + maskZ[i],
                                       squareY + maskY[i],
                                       squareX + maskX[i]]
                if px > signal_:
                    signal_ = px
            results[feat, RG_COL] = np.sqrt(Rg_/mass_)
        else:
            for i in range(N_mask):
                px = image[squareZ + maskZ[i],
                           squareY + maskY[i],
                           squareX + maskX[i]]

                RgZ += z2_mask[i]*px
                RgY += y2_mask[i]*px
                RgX += x2_mask[i]*px

                raw_mass_ += raw_image[squareZ + maskZ[i],
                                       squareY + maskY[i],
                                       squareX + maskX[i]]
                if px > signal_:
                    signal_ = px
            results[feat, RGZ_COL] = np.sqrt(RgZ/mass_)
            results[feat, RGY_COL] = np.sqrt(RgY/mass_)
            results[feat, RGX_COL] = np.sqrt(RgX/mass_)

        results[feat, MASS_COL] = mass_
        if characterize:
            results[feat, SIGNAL_COL] = signal_
            results[feat, ECC_COL] = np.nan
            results[feat, RAW_MASS_COL] = raw_mass_

    return 0  # Unused
