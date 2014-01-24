=========================
 :mod:`linking` Module
=========================

Most users will rely only on ``link_df`` (for "link DataFrame") which expects
results the format given from ``locate`` and ``batch``.

.. autofunction:: trackpy.link_df

We continue to support ``link`` which expects ``trackpy.Point`` objects as
a list of lists.

.. autofunction:: trackpy.link
.. autoclass:: trackpy.Point
.. autoclass:: trackpy.PointND
.. autoclass:: trackpy.Track

At the lowest level is ``link_iter``, which can returns results iteratively
and process unlimited stream of data using a fixed amount of memory.

.. autofunction:: trackpy.link_iter
.. autoclass:: trackpy.IndexedPointND
.. autoclass:: trackpy.DummyTrack

The BTree link strategy uses a hash table that can be fully specified by
keyword arguments, but you can also build one yourself.

.. autoclass:: trackpy.HashTable

The KDTree link strategy uses a class that, at this point, is not exposed
to the user and shouldn't need to be, but here it is for completeness:

.. autoclass:: trackpy.TreeFinder
