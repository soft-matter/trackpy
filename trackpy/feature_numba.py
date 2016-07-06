from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
from .try_numba import try_numba_autojit


@try_numba_autojit(nopython=True)
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

        # matplotlib and ndimage have opposite conventions for xy <-> yx.
        results[feat, 0] = cm_iX
        results[feat, 1] = cm_iY

        # Characterize the neighborhood of our final centroid.
        results[feat, MASS_COL] = mass_

    return 0  # Unused

@try_numba_autojit(nopython=True)
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

        # matplotlib and ndimage have opposite conventions for xy <-> yx.
        results[feat, 0] = cm_iX
        results[feat, 1] = cm_iY

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
    

@try_numba_autojit(nopython=True)
def _numba_refine_2D_c_a(raw_image, image, radiusY, radiusX, coords, N,
                         max_iterations, shift_thresh, shapeY, shapeX, maskY,
                         maskX, N_mask, y2_mask, x2_mask, cmask, smask,
                         results):
    # Column indices into the 'results' array
    MASS_COL = 2
    RGX_COL = 3
    RGY_COL = 4
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

        # matplotlib and ndimage have opposite conventions for xy <-> yx.
        results[feat, 0] = cm_iX
        results[feat, 1] = cm_iY

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


@try_numba_autojit(nopython=True)
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
        RGX_COL = 4
        RGY_COL = 5
        RGZ_COL = 6
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

        # matplotlib and ndimage have opposite conventions for xy <-> yx.
        results[feat, 0] = cm_iX
        results[feat, 1] = cm_iY
        results[feat, 2] = cm_iZ

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
