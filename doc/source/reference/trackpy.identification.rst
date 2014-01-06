=============================
 :mod:`identification` Module
=============================

.. warning:: This module is not in use by the author.

A minimal implementation of the Crocker-Grier algorithm for locating
round features in images.  This is done in three passes:

  1. band pass (:func:`trackpy.identification.band_pass`)
  2. locating the pixels which are local maximum (:func:`trackpy.identification.find_local_max`)
  3. sub-pixel refinement (:func:`trackpy.identification.subpixel_centroid`)

as such:

.. code-block:: python

    import trackpy.identification as tid
    # load a single frame
    img = load_your_image()
    # band pass the image
    bp_img = tid.band_pass(img, p_rad, hwhm)
    # locate the local maximum (more-or-less the center of the particles)
    res_lm = tid.find_local_max(bp_img, d_rad)
    # refine to get sub-pixel resolution
    locs, mass, r2 = tid.subpixel_centroid(bp_img, res_lm, mask_rad)




.. automodule:: trackpy.identification
   :members:
   :show-inheritance:
