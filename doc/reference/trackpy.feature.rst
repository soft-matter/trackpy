=============================
 :mod:`feature` Module
=============================

A full-featured implementation of the Crocker-Grier algorithm for locating
round features in images.


.. code-block:: python

    import trackpy as tp
    diameter = 5  # estimated size of features
    tp.locate(image, diameter)

`locate` returns a DataFrame (a spreadsheet-like object) listing the
position, mass (total brightness), size (radius-of-gyration of brightness),
and eccentricity. It also lists the signal, a measure related the contrast,
and ep for epsilon, the estimated uncertainty in the position of the feature.

`locate` prepares the image by performing a band pass using sensible defaults
derived from the diameter you specified. You choose your settings or
override this preprocessing all together; see the API documentation below.

Then, following the Crocker-Grier procedure, it locates all local maxima,
filters very dim maxima away, and refines the remainder to subpixel
accuracy by iteratively honing in on their center of brightness.


.. autofunction:: trackpy.locate
.. autofunction:: trackpy.batch

These locate doesn't do exactly what you want, you can dig into the lower-
level functions and develop something of your own.

.. autofunction:: trackpy.local_maxima
.. autofunction:: trackpy.refine
.. autofunction:: trackpy.binary_mask
.. autofunction:: trackpy.r_squared_mask
.. autofunction:: trackpy.theta_mask
.. autofunction:: trackpy.sinmask
.. autofunction:: trackpy.cosmask
