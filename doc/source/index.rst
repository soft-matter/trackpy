.. trackpy documentation master file, created by
   sphinx-quickstart on Sun Sep 16 14:53:53 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

A Pure python implementation of Crocker-Grier
=============================================
:Release: |version|
:date: |today|

The Crocker-Grier algorithm is a method of tracking features in a
series of images from frame to frame.  The core of the algorithm is to
choose the frame-to-frame linking that globally minimizes the sum of
the squared displacements.

:mod:`trackpy` is a simple and extendable implementation of
Crocker-Grier suitable for tracking a few hundred to a few thousand
features per frame.  This implementation has been shown to consume an
unreasonable amount of memory when tracking >20k features.  For large
data sets, see the `c++ implementation`__ of Crocker-Grier.



.. _cpp: https://github.com/tacaswell/tracking
__ cpp_


Contents:
=========

.. toctree::
   :maxdepth: 3

   reference/trackpy

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
