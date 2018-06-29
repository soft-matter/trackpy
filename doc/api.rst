.. _api_ref:

.. currentmodule:: trackpy

API reference
=============
The core functionality of trackpy is grouped into three separate steps:

1. Locating features in an image
2. Refining feature coordinates to obtain subpixel precision
3. Identifying features through time, linking them into trajectories.

Convenience functions for feature finding, refinement, and linking are readily available:

.. autosummary::
    :toctree: generated/

    locate
    batch
    link

For more control on your tracking "pipeline", the following core functions are provided:


Feature finding
---------------
.. autosummary::
    :toctree: generated/

    grey_dilation
    find_link


Coordinate refinement
---------------------
.. autosummary::
    :toctree: generated/

    refine_com
    refine_leastsq

Linking
-------
.. autosummary::
    :toctree: generated/

    link
    link_iter
    link_df_iter


:func:`~trackpy.linking.link` and :func:`~trackpy.linking.link_df_iter` run
the same underlying code. :func:`~trackpy.linking.link` operates on a single
DataFrame containing data for an entire movie.
:func:`~trackpy.linking.link_df_iter` streams through larger data sets,
in the form of one DataFrame for each video frame.
:func:`~trackpy.linking.link_iter` streams through a series of numpy
ndarrays.

See the tutorial on large data sets for more.

Static Analysis
---------------

.. autosummary::
    :toctree: generated/

    static.proximity
    static.pair_correlation_2d
    static.pair_correlation_3d
    static.cluster

Motion Analysis
---------------

.. autosummary::
    :toctree: generated/

    motion.msd
    motion.imsd
    motion.emsd
    motion.compute_drift
    motion.subtract_drift
    motion.vanhove
    motion.relate_frames
    motion.velocity_corr
    motion.direction_corr
    motion.is_typical
    motion.diagonal_size
    motion.theta_entropy
    motion.min_rolling_theta_entropy
    filtering.filter_stubs
    filtering.filter_clusters

Prediction Framework
--------------------

Trackpy extends the Crocker--Grier algoritm using a prediction framework, described in the prediction tutorial.

.. autosummary::
   :toctree: generated/

   predict.NullPredict     
   predict.ChannelPredict
   predict.DriftPredict
   predict.NearestVelocityPredict
   predict.predictor
   predict.instrumented

Plotting Tools
--------------

Trackpy includes functions for plotting the data in ways that are commonly useful. If you don't find what you need here, you can plot the data any way you like using matplotlib, seaborn, or any other plotting library.

.. autosummary::
    :toctree: generated/

    annotate
    scatter
    plot_traj
    annotate3d
    scatter3d
    plot_traj3d
    plot_displacements
    subpx_bias
    plot_density_profile

These two are almost too simple to justify their existence -- just a convenient shorthand for a common plotting task.

.. autosummary::
    :toctree: generated/

    mass_ecc
    mass_size

Image Conversion
----------------

By default, :func:`~trackpy.feature.locate` applies a bandpass and a percentile-based
threshold to the image(s) before finding features. You can turn off this functionality
using ``preprocess=False, percentile=0``.) In many cases, the default bandpass, which
guesses good length scales from the ``diameter`` parameter, "just works." But if you want
to executre these steps manually, you can.

.. autosummary::
    :toctree: generated/

    find.percentile_threshold
    preprocessing.bandpass
    preprocessing.lowpass
    preprocessing.scale_to_gamut
    preprocessing.invert_image
    preprocessing.convert_to_int

Framewise Data Storage & Retrieval Interface
--------------------------------------------

Trackpy implements a generic interface that could be used to store and
retrieve particle tracking data in any file format. We hope that it can
make it easier for researchers who use different file formats to exchange data. Any in-house format could be accessed using the same simple interface in trackpy.

At present, the interface is implemented only for HDF5 files. There are
several different implementations, each with different performance
optimizations. :class:`~trackpy.framewise_data.PandasHDFStoreBig` is a good general-purpose choice.

.. autosummary::
    :toctree: generated/

    PandasHDFStore
    PandasHDFStoreBig
    PandasHDFStoreSingleNode
    FramewiseData

That last class cannot be used directly; it is meant to be subclassed
to support other formats. See *Writing Your Own Interface* in the streaming tutorial for
more.

Logging
-------

Trackpy issues log messages. This functionality is mainly used to report the
progress of lengthy jobs, but it may be used in the future to report details of
feature-finding and linking for debugging purposes.

When trackpy is imported, it automatically calls `handle_logging()`, which sets
the logging level and attaches a logging handler that plays nicely with
IPython notebooks. You can override this by calling `ignore_logging()` and
configuring the logger however you like.

.. autosummary::
    :toctree: generated/

    quiet
    handle_logging
    ignore_logging

Utility functions
-----------------

.. autosummary::
    :toctree: generated/

    minmass_v03_change
    minmass_v04_change
    utils.fit_powerlaw

Diagnostic functions
--------------------

.. autosummary::
   :toctree: generated/

   diag.performance_report
   diag.dependencies

Low-Level API (Advanced)
------------------------

Switching Between Numba and Pure Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trackpy implements the most intensive (read: slowest) parts of the core feature-finding and linking algorithm in pure Python (with numpy) and also in `numba <http://numba.pydata.org/>`_, which accelerates Python code. Numba can offer a major performance boost, but it is still relatively new, and it can be challenging to use. If numba is available, trackpy will use the numba implementation by default; otherwise, it will use pure Python. The following functions allow sophisticated users to manually switch between numba and pure-Python modes. This may be used, for example, to measure the performance of these two implementations on your data.

.. autosummary::
   :toctree: generated/

   enable_numba
   disable_numba


Low-Level Linking API
^^^^^^^^^^^^^^^^^^^^^

All of the linking functions in trackpy provide the same level of control over the linking algorithm itself. For almost all users, the functions above will be sufficient. But :func:`~trackpy.linking.link_df` and :func:`~trackpy.linking.link_df_iter` above do assume that the data is stored in a pandas DataFrame. For users who want to use some other iterable data structure, the functions below provide direct access to the linking code.

.. autosummary::
    :toctree: generated/

    link_iter
    link

And the following classes can be subclassed to implement a customized linking procedure.

.. autosummary::
    :toctree: generated/

    Point
    PointND
    Track
    TrackUnstored 
    HashTable
    SubnetOversizeException


Masks
^^^^^

These functions may also be useful for rolling your own algorithms:

.. autosummary::
    :toctree: generated/

    masks.binary_mask
    masks.r_squared_mask
    masks.x_squared_masks
    masks.cosmask
    masks.sinmask
    masks.theta_mask
    masks.gaussian_kernel
    masks.mask_image
    masks.slice_image

