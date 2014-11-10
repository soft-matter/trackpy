.. _api_ref:

.. currentmodule:: trackpy

API reference
=============

Locating & Linking Features
---------------------------

These functions acess the core functionality of trackpy

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


Bare-Bones Linking API (Advanced)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All of the linking functions in trackpy provide the same level of control over the linking algorithm itself. For almost all users, the functions above will be sufficient.

But :func:`~trackpy.linking.link_df` and :func:`~trackpy.linking.link_df_iter` above do assume that the data is stored in a pandas DataFrame. For users who want to use some other iterable data structure, the functions below provide direct access to the linking code.

.. autosummary::
    :toctree: generated/

    link_iter
    link

Visualization Tools
-------------------

.. autosummary::
    :toctree: generated/

    annotate
    plot_traj
    plot_displacements

Utility functions
-----------------

.. autosummary::
    :toctree: generated/

    utils.fit_powerlaw
