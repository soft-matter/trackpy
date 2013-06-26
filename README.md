mr: microrheology toolkit
=========================

What is it?
-----------

**mr** is a Python package providing tools for passive and active microrheology experiments.

Example:
  * [Compute the viscosity of water from a video of colloids in Brownian motion.](http://nbviewer.ipython.org/url/raw.github.com/danielballan/mr/master/examples/mr%20simple%20example.ipynb)

[![build status](https://travis-ci.org/danielballan/mr.png)](https://travis-ci.org/danielballan/mr)

Features
--------

  * The [widely-used particle locating algorithm](http://www.physics.emory.edu/~weeks/idl/tracking.html) originally implemented
    by John Crocker and Eric Weeks in IDL is reimplemented in
    Python. Wherever possible, existing tools from widely-used Python modules 
    are employed. Each logical step is broken into a single-purpose function,
    rendering a modular code that is **easy to customize and maintain**. 
  * To verify the code, a **suite of tests reproduces basic results** (e.g., 
    computing the viscosity of water from a video of diffusing particles).
  * **Uncertainty is estimated** using a method [proposed in this paper](http://dx.doi.org/10.1529/biophysj.104.042457).
  * For performance, array-intensive steps that are not available in
    standard scientific modules are written in C and imported.
  * Results are given as DataFrames, high-performance spreadsheet-like objects 
    from [Python pandas](http://pandas.pydata.org/pandas-docs/stable/overview.html) which can easily be saved to a **CSV file, Excel spreadsheet, 
    SQL database, HDF5 file**, or other.
  * Frames of video can be loaded from a **video file (AVI, MOV, etc.), a**
    **multi-frame TIFF, or a directory of sequential images (TIFF, 
    PNG, JPG, etc.)**.
  * Particle trajectories can be 
    characterized, grouped, and plotted using a suite of convenient functions.
  * Various models relate probe statistics to rheological response, including
    the Generalized Stokes-Einstein implementation used in the Crocker/Grier 
    code.

Dependencies
------------

Essential:

  * ``numpy``
  * ``scipy``
  * ``matplotlib``
  * [``pandas``](http://pandas.pydata.org/pandas-docs/stable/overview.html)
  * [``trackpy``](https://github.com/tacaswell/trackpy)


Optional:

  * [``cv2``](http://opencv.org/downloads.html) for reading video files
      and viewing video with annotations
  * ``libtiff`` for reading multi-frame tiff images
  * ``PyTables`` for saving results in an HDF5 file
  * ``sqlite`` or ``MySQLdb`` for saving results in a SQL database

Related Projects
----------------

  * Particle tracking using IDL http://www.physics.emory.edu/~weeks/idl/
  * A C++ implementation (also wrapped in Python) https://github.com/tacaswell/tracking
  * A Matlab implementation by Daniel Blair and Eric Dufresne http://physics.georgetown.edu/matlab/

Background
----------

This package was developed and is maintained by Daniel Allan, as part of his
PhD thesis work on microrheology in Robert L. Leheny's group at Johns Hopkins
University in Baltimore, MD.

Dan can be reached at dallan@pha.jhu.edu.
