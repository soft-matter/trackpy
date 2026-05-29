.. _api_ref:

API reference
=============
The core functionality of trackpy is grouped into three separate steps:

1. Locating features in an image
2. Refining feature coordinates to obtain subpixel precision
3. Identifying features through time, linking them into trajectories.

Convenience functions for feature finding, refinement, and linking are readily available:

.. autosummary::
    :toctree: generated/

    trackpy.locate
    trackpy.batch
    trackpy.link

For more control on your tracking "pipeline", the following core functions are provided:


Feature finding
---------------
.. autosummary::
    :toctree: generated/

    trackpy.grey_dilation
    trackpy.find_link


Coordinate refinement
---------------------
.. autosummary::
    :toctree: generated/

    trackpy.refine_com
    trackpy.refine_leastsq

Linking
-------
.. autosummary::
    :toctree: generated/

    trackpy.link
    trackpy.link_iter
    trackpy.link_df_iter
    trackpy.link_partial
    trackpy.reconnect_traj_patch


:func:`~trackpy.linking.link` and :func:`~trackpy.linking.link_df_iter` run
the same underlying code. :func:`~trackpy.linking.link` operates on a single
DataFrame containing data for an entire movie.
:func:`~trackpy.linking.link_df_iter` streams through larger data sets,
in the form of one DataFrame for each video frame.
:func:`~trackpy.linking.link_iter` streams through a series of numpy
ndarrays.
:func:`~trackpy.linking.link_partial` can patch a region of trajectories in
an already linked dataset.


See the tutorial on large data sets for more.

Static Analysis
---------------

.. autosummary::
    :toctree: generated/

    trackpy.static.proximity
    trackpy.static.pair_correlation_2d
    trackpy.static.pair_correlation_3d
    trackpy.static.cluster

Motion Analysis
---------------

.. autosummary::
    :toctree: generated/

    trackpy.motion.msd
    trackpy.motion.imsd
    trackpy.motion.emsd
    trackpy.motion.compute_drift
    trackpy.motion.subtract_drift
    trackpy.motion.vanhove
    trackpy.motion.relate_frames
    trackpy.motion.velocity_corr
    trackpy.motion.direction_corr
    trackpy.motion.is_typical
    trackpy.motion.diagonal_size
    trackpy.motion.theta_entropy
    trackpy.motion.min_rolling_theta_entropy
    trackpy.filtering.filter_stubs
    trackpy.filtering.filter_clusters

Prediction Framework
--------------------

Trackpy extends the Crocker--Grier algoritm using a prediction framework, described in the prediction tutorial.

.. autosummary::
   :toctree: generated/

   trackpy.predict.NullPredict     
   trackpy.predict.ChannelPredict
   trackpy.predict.DriftPredict
   trackpy.predict.NearestVelocityPredict
   trackpy.predict.predictor
   trackpy.predict.instrumented

Plotting Tools
--------------

Trackpy includes functions for plotting the data in ways that are commonly useful. If you don't find what you need here, you can plot the data any way you like using matplotlib, seaborn, or any other plotting library.

.. autosummary::
    :toctree: generated/

    trackpy.annotate
    trackpy.scatter
    trackpy.plot_traj
    trackpy.annotate3d
    trackpy.scatter3d
    trackpy.plot_traj3d
    trackpy.plot_displacements
    trackpy.subpx_bias
    trackpy.plot_density_profile

These two are almost too simple to justify their existence -- just a convenient shorthand for a common plotting task.

.. autosummary::
    :toctree: generated/

    trackpy.mass_ecc
    trackpy.mass_size

Image Conversion
----------------

By default, :func:`~trackpy.feature.locate` applies a bandpass and a percentile-based
threshold to the image(s) before finding features. You can turn off this functionality
using ``preprocess=False, percentile=0``.) In many cases, the default bandpass, which
guesses good length scales from the ``diameter`` parameter, "just works." But if you want
to executre these steps manually, you can.

.. autosummary::
    :toctree: generated/

    trackpy.find.percentile_threshold
    trackpy.preprocessing.bandpass
    trackpy.preprocessing.lowpass
    trackpy.preprocessing.scale_to_gamut
    trackpy.preprocessing.invert_image
    trackpy.preprocessing.convert_to_int

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

    trackpy.PandasHDFStore
    trackpy.PandasHDFStoreBig
    trackpy.PandasHDFStoreSingleNode
    trackpy.FramewiseData

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

    trackpy.quiet
    trackpy.handle_logging
    trackpy.ignore_logging

Utility functions
-----------------

.. autosummary::
    :toctree: generated/

    trackpy.minmass_v03_change
    trackpy.minmass_v04_change
    trackpy.utils.fit_powerlaw

Diagnostic functions
--------------------

.. autosummary::
   :toctree: generated/

   trackpy.diag.performance_report
   trackpy.diag.dependencies

Low-Level API (Advanced)
------------------------

Switching Between Numba and Pure Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trackpy implements the most intensive (read: slowest) parts of the core feature-finding and linking algorithm in pure Python (with numpy) and also in `numba <http://numba.pydata.org/>`_, which accelerates Python code. Numba can offer a major performance boost, but it is still relatively new, and it can be challenging to use. If numba is available, trackpy will use the numba implementation by default; otherwise, it will use pure Python. The following functions allow sophisticated users to manually switch between numba and pure-Python modes. This may be used, for example, to measure the performance of these two implementations on your data.

.. autosummary::
   :toctree: generated/

   trackpy.enable_numba
   trackpy.disable_numba


Low-Level Linking API
^^^^^^^^^^^^^^^^^^^^^

All of the linking functions in trackpy provide the same level of control over the linking algorithm itself. For almost all users, the functions above will be sufficient. But :func:`~trackpy.linking.link_df` and :func:`~trackpy.linking.link_df_iter` above do assume that the data is stored in a pandas DataFrame. For users who want to use some other iterable data structure, the functions below provide direct access to the linking code.

.. autosummary::
    :toctree: generated/

    trackpy.link_iter
    trackpy.link

And the following classes can be subclassed to implement a customized linking procedure.

.. autosummary::
    :toctree: generated/

    trackpy.SubnetOversizeException

Masks
^^^^^

These functions may also be useful for rolling your own algorithms:

.. autosummary::
    :toctree: generated/

    trackpy.masks.binary_mask
    trackpy.masks.r_squared_mask
    trackpy.masks.x_squared_masks
    trackpy.masks.cosmask
    trackpy.masks.sinmask
    trackpy.masks.theta_mask
    trackpy.masks.gaussian_kernel
    trackpy.masks.mask_image
    trackpy.masks.slice_image

Full API reference
------------------

A full overview of all modules and functions can be found below:

.. autosummary::
    :toctree: generated/
    :recursive:

    trackpy

..
  Note: we excluded trackpy.tests in conf.py (autosummary_mock_imports)
