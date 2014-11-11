.. _api_ref:

.. currentmodule:: trackpy

API reference
=============

Locating & Linking 
------------------

These functions acess the core functionality of trackpy:

1. Locating features in an image
2. Locating features in a batch of many images
3. Identifying features through time, linking them into trajectories.

.. autosummary::
    :toctree: generated/

    locate
    batch
    link_df
    link_df_iter

:func:`~trackpy.linking.link_df` and :func:`~trackpy.linking.link_df_iter` run the same underlying code, but :func:`~trackpy.linking.link_df_iter` streams through large data sets one frame at a time. See the tutorial on large data sets for more.

Motion Analysis
---------------

.. autosummary::
    :toctree: generated/

    imsd
    emsd
    compute_drift
    subtract_drift
    vanhove
    relate_frames
    velocity_corr
    direction_corr
    proximity
    is_typical
    diagonal_size

Plotting Tools
--------------

Trackpy includes functions for plotting the data in ways that are commonly useful. If you don't find what you need here, you can plot the data any way you like using matplotlib, seaborn, or any other plotting library.

.. autosummary::
    :toctree: generated/

    annotate
    plot_traj
    plot_displacements
    subpx_bias

These two are almost too simple to justify their existence -- just a convenient shorthand for a common plotting task.

.. autosummary::
    :toctree: generated/

    mass_ecc
    mass_size

Image Cleanup
-------------

By default, :func:`~trackpy.feature.locate` and :func:`~trackpy.feature.batch` apply a bandpass and a percentile-based threshold to the image(s) before finding features. (You can turn off this functionality using ``preprocess=False, percentile=0``.) In many cases, the default bandpass, which guesses good length scales from the ``diameter`` parameter, "just works." But if you want to executre these steps manually, you can.

.. autosummary::
    :toctree: generated/

    bandpass
    percentile_threshold

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
to support other formats.

Utility functions
-----------------

.. autosummary::
    :toctree: generated/

    utils.fit_powerlaw
    utils.print_update

Low-Level API (Advanced)
------------------------

Intermediate Steps of Feature-Finding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The key steps of the feature-finding algorithm are implemented as separate, modular functions. You can run them in sequence to inspect intermediate steps, or you can use them to roll your own variation on the algorithm.

.. autosummary::
    :toctree: generated/

    local_maxima
    refine
    estimate_mass
    estimate_size

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
    IndexedPointND
    Track
    DummyTrack
    HashTable
    SubnetOversizeException


Masks
^^^^^

These functions may also be useful for rolling your own algorithms:

.. autosummary::
    :toctree: generated/

    masks.binary_mask
    masks.r_squared_mask
    masks.cosmask
    masks.sinmask
    masks.theta_mask

