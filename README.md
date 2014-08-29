trackpy
=======

[![build status](https://travis-ci.org/soft-matter/trackpy.png?branch=master)](https://travis-ci.org/soft-matter/trackpy) [![doi](https://zenodo.org/badge/3990/soft-matter/trackpy.png)](https://zenodo.org/record/9971)

What is it?
-----------

**trackpy** is a Python package providing tools for particle tracking.
**[Read the walkthrough](http://nbviewer.ipython.org/github/soft-matter/trackpy-examples/blob/master/notebooks/walkthrough.ipynb)** to skim or study an example project from start to finish.

Then browse a list of **[more examples](http://nbviewer.ipython.org/github/soft-matter/trackpy-examples/tree/master/notebooks/)**, or
download the [full repository of sample code and data](https://github.com/soft-matter/trackpy-examples) to try them yourself.

There are many similar projects. (See table below.)
Our implementation is distinguished by succinct and flexible usage,
a thorough testing framework ensuring code stability and accuracy,
scalability, and complete documentation. 

Several researchers have merged their independent efforts into this code.
We would like to see others in the community adopt it and potentially
contribute code to it.

*If you use trackpy in published research, please read the section [Citing Trackpy](#citing-trackpy).*

Features
--------

### Basics

  * The [widely-used particle locating algorithm](http://www.physics.emory.edu/~weeks/idl/tracking.html)
    originally implemented by John Crocker and Eric Weeks in IDL is reimplemented
    in Python. Wherever possible, existing tools from widely-used Python modules
    are employed.
  * This reimplemention is full-featured, including subpixel precision
    verified with test cases.
  * The module is actively used and tested on **Windows, Mac OSX, and Linux**,
    and it uses only free, open-source software.
  * Frames of video can be loaded from a **video file (AVI, MOV, etc.), a**
    **multi-frame TIFF, or a directory of sequential images (TIFF,
    PNG, JPG, etc.)**.
  * Results are given as DataFrames, high-performance spreadsheet-like objects
    from [Python pandas](http://pandas.pydata.org/pandas-docs/stable/overview.html)
    which can easily be saved to a **CSV file, Excel spreadsheet,
    SQL database, HDF5 file**, and more.
  * Particle trajectories can be
    characterized, grouped, and plotted using a suite of convenient functions.
  * To verify correctness and stability, a **suite of 150+ tests reproduces
    basic results**.

### Special Capabilities

  * Both feature-finding and trajectory-linking can be performed on
    **arbitrarily long videos** using a fixed, modest amount of memory. (Results
    can be read and saved to disk throughout.)
  * A **prediction framework** helps track particles in fluid flows,
    or other scenarios where velocity is correlated between time steps.
  * Feature-finding and trajectory-linking works on **images with any number of dimensions**,
    making possible some creative applications.
  * **Uncertainty is estimated** following a method [described in this paper](http://dx.doi.org/10.1529/biophysj.104.042457) by Savin and Doyle.
  * **High-performance** components (numba acceleration and FFTW support) are used only if
  if available. Since these can be tricky to install on some machines,
  the code will automatically fall back on slower pure Python implementations
  as needed.

Documentation
-------------
The examples linked to above are the best place to start. To try them out on your
own computer, you will want to have the sample data as well; you can download
all of the examples and data from the
[examples repository](https://github.com/soft-matter/trackpy-examples). There is also
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
"Start > Applications > Command Prompt" on Windows. Type or paste these
lines to make certain that Anaconda will work well with trackpy:

    conda update conda
    conda install numpy=1.8 scipy=0.14.0 matplotlib=1.3 pandas=0.13.0 scikit-image=0.10.1 pyyaml numba=0.12.2
    conda install pip
    
Then, to install trackpy:

    pip install trackpy

Finally, to try it out, type

    ipython notebook

This will automatically open a browser tab, ready to interpret Python code.
To get started, check out the links to tutorials at the top of this document.

### For Experienced Python Users

You can install any of the dependencies using pip, but we
recommend starting with
[Anaconda](https://store.continuum.io/cshop/anaconda/), which comes
with several of the essential dependencies included.
[Canopy](https://www.enthought.com/products/canopy/) is another
distribution that makes a good starting point.

Essential Dependencies:

  * [``numpy``](http://www.scipy.org/)
  * [``scipy``](http://www.scipy.org/)
  * [``matplotlib``](http://matplotlib.org/)
  * [``pandas``](http://pandas.pydata.org/pandas-docs/stable/overview.html)
  * [``pyyaml``](http://pyyaml.org/)


You will also need the image- and video-reader PIMS, which is, like trackpy
itself, part of the github.com/soft-matter organization.

  * [``PIMS``](https://github.com/soft-matter/pims)

You can install PIMS and trackpy using pip:

    pip install pims
    pip install trackpy

Or, if you plan to edit the code, you can install them manually:

    git clone https://github.com/soft-matter/pims
    pip install -e pims

    git clone https://github.com/soft-matter/trackpy
    pip install -e trackpy

Optional Dependencies:

  * [``pyFFTW``](https://github.com/hgomersall/pyFFTW) to speed up the band
      pass, which is one of the slower steps in feature-finding
  * [``PyTables``](http://www.pytables.org/moin) for saving results in an 
      HDF5 file. This is included with Anaconda.
  * [``numba``](http://numba.pydata.org/) for accelerated feature-finding and linking. This is
      included with Anaconda and Canopy. Installing it any other way is difficult;
      we recommend sticking with one of these. Note that `numba` v0.12.0
      (included with Anaconda 1.9.0) has a bug and will not work at all;
      if you have this version, you should update Anaconda. We support numba 
      versions 0.11 and 0.12.2.

PIMS has its own optional dependencies for reading various formats. You
can read what you need for each format
[here on PIMS` README](https://github.com/soft-matter/pims).

### Updating Your Installation

The code is under active development. To update to the current development
version, run this in the command prompt:

    pip install --upgrade http://github.com/soft-matter/trackpy/zipball/master

Versions & Project Roadmap
--------------------------

See the [releases page](https://github.com/soft-matter/trackpy/releases) for details.

The original release is tagged Version 0.1. 
Although there have been major changes to the code, v0.2.x maintains complete
reverse compatibility with v0.1 and can be used as drop-in replacement.
We recommend all users upgrade.

The `master` branch on github contains the latest tested development code.
Changes are thoroughly tested before being merged. If you want to use the
latest features it should be safe to rely on the master branch.
(The primary contributors do.)

Roadmap:

* expansion of data structures to simplify sharing frame-wise data
between research groups
* interactive filtering and visualization tools
* continued performance improvments and benchmarking for a range of
use cases (frame size, particle density, etc.)
* tests that compare results again "battle-tested" Crocker-Grier code

Contributors
------------

* **Daniel Allan** feature-finding, uncertainty estimation,
motion characterization and discrimination, plotting tools, tests
* **Thomas Caswell** multiple implementations of sophisticated trajectory-linking, tests
* **Nathan Keim** alternative trajectory-linking implementations, major
speed-ups, prediction

Citing Trackpy
--------------

If you use trackpy for published research, please cite this repository,
including the primary contributors' names -- Daniel B. Allan, Thomas A. Caswell,
and Nathan C. Keim -- and `doi:10.5281/zenodo.9971`.
If your citation style *also* allows for a URL,
please include `github.com/soft-matter/trackpy` to help other
researchers discover trackpy. Our
[DOI record page](https://zenodo.org/record/9971)
provides more detail and citations in various formats.

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
