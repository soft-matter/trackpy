trackpy
=======

[![build status](https://travis-ci.org/soft-matter/trackpy.png?branch=master)](https://travis-ci.org/soft-matter/trackpy)

What is it?
-----------

**trackpy** is a Python package providing tools for particle tracking. **[Read the walkthrough](http://nbviewer.ipython.org/github/soft-matter/trackpy-examples/blob/master/notebooks/walkthrough.ipynb)** to skim or study an example project from start to finish.

More Examples and Tutorials:
  * [Load frames from a video file, a multi-frame TIFF, or a folder of images.](http://nbviewer.ipython.org/github/soft-matter/trackpy-examples/blob/master/notebooks/loading-video-frames.ipynb)
  * [Save data in a variety of formats; handle large or concurrent jobs; access partial data sets while they are processed.](http://nbviewer.ipython.org/github/soft-matter/trackpy-examples/blob/master/notebooks/tools-for-large-or-concurrent-jobs.ipynb)

There are many similar projects. (See table below.)
Our implementation is distinguished by succinct and flexible usage,
a thorough testing framework ensuring code stability and accuracy,
scalability, and complete documentation. 

Several researchers have merged their independent efforts into this code.
We would like to see others in the community adopt it and potentially
contribute code to it.

Features
--------

### Basics

  * The [widely-used particle locating algorithm](http://www.physics.emory.edu/~weeks/idl/tracking.html) originally implemented
    by John Crocker and Eric Weeks in IDL is reimplemented in
    Python. Wherever possible, existing tools from widely-used Python modules
    are employed.
  * This reimplemention is full-featured, including subpixel precision
    verified with test cases.
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
  * To verify correctness and stability, a **suite of 150 tests reproduces
    basic results**.

### Special Capabilities

  * Both feature-finding and trajectory-linking can be performed on
    **arbitrarily long videos** using a fixed, modest amount of memory. (Results
    can be read and saved to disk throughout.)
  * Feature-finding works on **images with any number of dimensions**,
    making possible some creative applications.
  * Trajectory-linking is supported in 2 and 3 dimensions.
  * **Uncertainty is estimated** following a method [described in this paper](http://dx.doi.org/10.1529/biophysj.104.042457) by Savin and Doyle.
  * **High-performance** components (C extensions, FFTW support, numba 
  support) are used only
  if available. Since these can be tricky to install on some machines,
  the code will automatically fall back on slower pure Python implementations
  as needed.

Documentation
-------------
The tutorials above are the best place to start. There is also
 **[complete documentation](http://trackpy.readthedocs.org/)** for every 
function in the package.

Installation
------------

### For Python Novices

Installation is simple on Windows, OSX, and Linux, even for Python novices.

To get started with Python on any platform, download and install
[Anaconda](https://store.continuum.io/cshop/anaconda/). It comes with the
common scientific Python packages built in.

If you are using Windows, I recommend 32-bit Anaconda even if your system is 64-bit.
(One of the optional dependencies is not yet compatible with 64-bit Python.)

Open a command prompt. That's "Terminal" on a Mac, and
"Start > Applications > Command Prompt" on Windows. Type these
lines:

    conda install numpy=1.7.1 scipy=0.13.0 matplotlib=1.3 pandas=0.13.0 numba=0.11 PIL pyyaml
    conda install pip
    pip install http://github.com/soft-matter/pims/zipball/master
    pip install http://github.com/soft-matter/trackpy/zipball/master

In the command prompt, type

    ipython notebook

This will automatically open a browser tab, ready to interpret Python code.
Follow the tutorials to get started.

### For Experienced Python Users

You can install any of the dependencies using pip or
[Anaconda](https://store.continuum.io/cshop/anaconda/), which comes
with some of the essential dependencies included.

If you are using Windows, I recommend 32-bit Anaconda even if your system is 64-bit.
(One of the optional dependencies, ``opencv``, is not readily compatible with 64-bit
Python.)

Essential Dependencies:

  * [``numpy``](http://www.scipy.org/)
  * [``scipy``](http://www.scipy.org/)
  * [``matplotlib``](http://matplotlib.org/)
  * [``pandas``](http://pandas.pydata.org/pandas-docs/stable/overview.html)
  * [``pyyaml``](http://pyyaml.org/)


You will also need the image- and video-reader pims, which is, like trackpy
itself, part of the github.com/soft-matter organization.

  * [``pims``](https://github.com/soft-matter/pims)

You can install pims and trackpy using pip:

    pip install http://github.com/soft-matter/pims/zipball/master
    pip install http://github.com/soft-matter/trackpy/zipball/master

Or, if you plan to edit the code, you can install them manually:

    git clone https://github.com/soft-matter/pims
    cd pims
    python setup.py develop

    cd ..

    git clone https://github.com/soft-matter/trackpy
    cd trackpy
    python setup.py develop

Optional Dependencies:

  * [``pyFFTW``](https://github.com/hgomersall/pyFFTW) to speed up the band
      pass, which is one of the slower steps in feature-finding
  * [``PyTables``](http://www.pytables.org/moin) for saving results in an 
      HDF5 file. This is included with Anaconda.
  * [``numba``] for accelerated feature-finding and linking. This is
      included with Anaconda. Installing it any other way is difficult;
      we recommend sticking with Anaconda. Note that `numba` v0.12.0
      (included with Anaconda 1.8.0) has a bug and will not work at all;
      if you have this version, you should update Anaconda.

Pims has its own optional dependencies for reading various formats. You
can read what you need for each format
[here on pims` README](https://github.com/soft-matter/pims).

### Updating Your Installation

The code is under active development. To update to the current development
version, run this in the command prompt:

    pip install --upgrade http://github.com/soft-matter/trackpy/zipball/master

Verions & Project Roadmap
-------------------------

A version 0.1 has been tagged and the v0.1.x branch will get bug
fixes.  This version does not depend on `pandas`.

On the current master branch, which the instructions above would download,
we have made significant changes:

 - merging most of Dan Allan's `mr` module
 - replacing `identification.py` with superior `feature.py`
 - making `link` iterative
 - merging Nathan Keim's KDTree-based linking, which is 2X faster on
   typical data 
 - merging Nathan Keim's numba-acceleration, falling back on pure Python
   if numba is not available
 - providing access to different linking strategies through 
   keyword arguments (Type ``help(link)`` or ``help(link_df)`` for details.)
 - reworking out-of-core (on-disk) processing of large data sets
   to suit 

Contributors
------------

* **Daniel Allan** feature-finding, uncertainty estimation,
motion characterization and discrimination, plotting tools, tests
* **Thomas Caswell** multiple implementations of sophisticated trajectory-linking, tests
* **Nathan Keim** alternative trajectory-linking implementations, major
speed-ups, prediction

Related Projects
----------------

| Author(s) | Project URL | Programming Language |
| --------- | ----------- | -------------------- |
| Crocker and Grier | http://physics.nyu.edu/grierlab/software.html | IDL |
| Crocker and Weeks | http://www.physics.emory.edu/~weeks/idl/ | IDL |
| Blair and Dufresne | http://physics.georgetown.edu/matlab/ | MATLAB |
| Maria Kilfoil | http://people.umass.edu/kilfoil/downloads.html | MATLAB and Python |
| Graham Milne | http://zone.ni.com/devzone/cda/epd/p/id/948 | LabVIEW |
| Ryan Smith and Gabe Spalding | http://titan.iwu.edu/~gspaldin/rytrack.html | stand alone/IDL GUI |
| Peter J Lu | https://github.com/peterlu/PLuTARC_centerfind2D | C++ (identification only) |
| Thomas A Caswell | https://github.com/tacaswell/tracking | C++ |

Support
-------

This package was developed in part by Daniel Allan, as part of his
PhD thesis work on microrheology in Robert L. Leheny's group at Johns Hopkins
University in Baltimore, MD. The work was supported by the National Science Foundation
under grant number CBET-1033985.  Dan can be reached at dallan@pha.jhu.edu.

This package was developed in part by Thomas A Caswell as part of his
PhD thesis work in Sidney R Nagel's and Margaret L Gardel's groups at
the University of Chicago, Chicago IL.  This work was supported in
part by NSF Grant DMR-1105145 and NSF-MRSEC DMR-0820054.  Tom can be
reached at tcaswell@gmail.com.

This package was developed in part by Nathan C. Keim, as part of his postdoctoral
research in Paulo Arratia's group at the University of Pennsylvania,
Philadelphia. This work was supported by NSF-MRSEC DMR-1120901.
