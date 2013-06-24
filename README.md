=========================
mr: microrheology toolkit
=========================

What is it
==========

**mr** is a Python package providing tools for passive and active microrheology experiments.

[![build status](https://travis-ci.org/danielballan/mr.png)](https://travis-ci.org/danielballan/mr)

Main Features
=============

  * The widely-used particle locating algorithm originally implemented
    by John Crocker and Eric Weeks in IDL is reimplemented in
    Python. Wherever possible, I use existing tools from widely-used Python
    modules. Each logical step is broken into a single-purpose function,
    rendering a modular code that is easy to customize and maintain. 
  * A suite of tests confirms basic results (e.g., computing the viscosity of
    water from a video of diffusing particles).
  * For performance, array-intensive steps that are not available in
    standard scientific modules are written in C and imported.
  * Results are given as DataFrames (see Python pandas). Particle trajectories can be 
    characterized, grouped, and plotted using a suite of convenient functions.
  * Various models relate probe statistics to rheological response, including
    the Generalized Stokes-Einstein implementation used in the Crocker/Grier 
    code.
  * Feature locations are saved one frame at a time to conserve memory. They
    can be saved to a CSV file, a SQL database, an HDF5 datastore, or any
    custom-made storage.

Dependencies
------------

Essential:

  * ``numpy``
  * ``scipy``
  * ``matplotlib``
  * ``pandas``
  * ``lmfit``
  * ``[trackpy]``(https://github.com/tacaswell/trackpy)

Recommended:

  * ``pyopencv`` for processing video files directly

Optional:

  * ``MySQLdb`` for saving results in a MySQL database
  * ``PyTables`` for saving results in an HDF5 datastore
  * 

Related Projects
================

  * Particle tracking using IDL http://www.physics.emory.edu/~weeks/idl/
  * A C++ implementation (also wrapped in Python) https://github.com/tacaswell/tracking
  * A Matlab implementation by Daniel Blair and Eric Dufresne http://physics.georgetown.edu/matlab/

Background
==========

This package was developed and is maintained by Daniel Allan, as part of his
PhD thesis work on microrheology in Robert L. Leheny's group at Johns Hopkins
University in Baltimore, MD.

Dan can be reached at dallan@pha.jhu.edu.
