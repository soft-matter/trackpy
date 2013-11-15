mr: microrheology toolkit
=========================

What is it?
-----------

**mr** is a Python package providing tools for particle tracking and
microrheology experiments

Examples and Tutorials:
  * [Compute the viscosity of water from a video of colloids in Brownian motion.](http://nbviewer.ipython.org/url/raw.github.com/soft-matter/mr/master/examples/mr%20simple%20example.ipynb)
  * [Load frames from a video file, a multi-frame TIFF, or a folder of images.](http://nbviewer.ipython.org/url/raw.github.com/soft-matter/mr/master/examples/loading%20video%20frames.ipynb)
  * [Save data in a variety of formats; handle large or concurrent jobs; access partial data sets while they are processed.](http://nbviewer.ipython.org/url/raw.github.com/soft-matter/mr/master/examples/tools%20for%20large%20or%20concurrent%20jobs.ipynb)

[![build status](https://travis-ci.org/soft-matter/mr.png?branch=master)](https://travis-ci.org/soft-matter/mr)

Features
--------

### Basics

  * The [widely-used particle locating algorithm](http://www.physics.emory.edu/~weeks/idl/tracking.html) originally implemented
    by John Crocker and Eric Weeks in IDL is reimplemented in
    Python. Wherever possible, existing tools from widely-used Python modules 
    are employed.
  * The module is actively used and tested on **Windows, Mac OSX, and Linux**,
    and it uses only free, open-source software.
  * Frames of video can be loaded from a **video file (AVI, MOV, etc.), a**
    **multi-frame TIFF, or a directory of sequential images (TIFF, 
    PNG, JPG, etc.)**.
  * Results are given as DataFrames, high-performance spreadsheet-like objects 
    from [Python pandas](http://pandas.pydata.org/pandas-docs/stable/overview.html) which can easily be saved to a **CSV file, Excel spreadsheet, 
    SQL database, HDF5 file**, and more.
  * Particle trajectories can be 
    characterized, grouped, and plotted using a suite of convenient functions.
  * To verify correctness and stability, a **suite of over 50 tests reproduces
    basic results**. 

### Special Capabilities

  * Both feature-finding and trajectory-linking can be performed on
    **arbitrarily long videos** using a fixed, modest amount of memory. (Results
    can be read and saved to disk throughout.)
  * Feature-finding works on **images with any number of dimensions**,
    making possible some creative applications.
  * Trajectory-linking is supported in 2 and 3 dimensions.
  * **Uncertainty is estimated** using a method [proposed in this paper](http://dx.doi.org/10.1529/biophysj.104.042457).
  * **High-performance** components (C extensions, FFTW support) are used
  if available. Since these can be tricky to install on some machines,
  the code will automatically fall back on slower pure Python implementations
  as needed.

Installation & Dependencies
---------------------------

To get started with Python on any platform, just download and install
[Anaconda](https://store.continuum.io/cshop/anaconda/). It comes with these
common scientific Python packages.

Essential:

  * ``numpy``
  * ``scipy``
  * ``matplotlib``
  * [``pandas``](http://pandas.pydata.org/pandas-docs/stable/overview.html)

You will also need these, which -- like ``mr`` itself -- are part of the
github.com/soft-matter organization.

  * [``trackpy``](https://github.com/soft-matter/trackpy)
  * [``yaml-serialize``](https://github.com/soft-matter/trackpy)

Optional:

  * [``cv2``](http://opencv.org/downloads.html) for reading video files
      and viewing video with annotations
  * ``libtiff`` for reading multi-frame tiff images
  * ``PyTables`` for saving results in an HDF5 file
  * ``sqlite`` or ``MySQLdb`` for saving results in a SQL database
  * [``pyFFTW``](https://github.com/hgomersall/pyFFTW) to speed up the band 
      pass, which is one of the slower steps in feature-finding

Related Projects
----------------

  * Particle tracking using IDL http://www.physics.emory.edu/~weeks/idl/
  * A C++ implementation (also wrapped in Python) https://github.com/tacaswell/tracking
  * A Matlab implementation by Daniel Blair and Eric Dufresne http://physics.georgetown.edu/matlab/

Background
----------

This package was developed and is maintained by Daniel Allan, as part of his
PhD thesis work on microrheology in Robert L. Leheny's group at Johns Hopkins
University in Baltimore, MD. The work was supported by the National Science Foundation under grant number CBET-1033985.

Dan can be reached at dallan@pha.jhu.edu.
