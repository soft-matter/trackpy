trackpy
=======

[![build status](https://travis-ci.org/soft-matter/trackpy.png?branch=master)](https://travis-ci.org/soft-matter/trackpy)

What is it?
-----------

**trackpy** is a Python package providing tools for particle tracking. **[Read the walkthrough](http://nbviewer.ipython.org/github/soft-matter/trackpy/blob/master/examples/walkthrough.ipynb)** to skim or study an example project from start to finish.

More Examples and Tutorials:
  * [Load frames from a video file, a multi-frame TIFF, or a folder of images.](http://nbviewer.ipython.org/github/soft-matter/trackpy/blob/master/examples/loading-video-frames.ipynb)
  * [Save data in a variety of formats; handle large or concurrent jobs; access partial data sets while they are processed.](http://nbviewer.ipython.org/github/soft-matter/trackpy/blob/master/examples/tools-for-large-or-concurrent-jobs.ipynb)

Features
--------

### Basics

  * The [widely-used particle locating algorithm](http://www.physics.emory.edu/~weeks/idl/tracking.html) originally implemented
    by John Crocker and Eric Weeks in IDL is reimplemented in
    Python. Wherever possible, existing tools from widely-used Python modules
    are employed.
  * This reimplemention is full-featured, including subpixel precision down to
    0.1 pixels, verified with test cases.
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

    conda install pip
    pip install http://github.com/soft-matter/yaml-serialize/zipball/master
    pip install http://github.com/soft-matter/pims/zipball/master
    pip install http://github.com/soft-matter/trackpy/zipball/master

In the command prompt, type

    ipython notebook

This will automatically open a browser tab, ready to interpret Python code.
Follow the tutorials to get started.

### For Experienced Python Users

You can install any of the dependencies using pip or
[Anaconda]((https://store.continuum.io/cshop/anaconda/)), which comes
with some of the essential dependencies included.

If you are using Windows, I recommend 32-bit Anaconda even if your system is 64-bit.
(One of the optional dependencies, ``opencv``, is not readily compatible with 64-bit
Python.)

Essential Dependencies:

  * [``numpy``](http://www.scipy.org/)
  * [``scipy``](http://www.scipy.org/)
  * [``matplotlib``](http://matplotlib.org/)
  * [``pandas``](http://pandas.pydata.org/pandas-docs/stable/overview.html)


You will also need these, which -- like ``trackpy`` itself -- are part of the
github.com/soft-matter organization.

  * [``pims``](https://github.com/soft-matter/pims)
  * [``yaml-serialize``](https://github.com/soft-matter/trackpy)

Install using pip:

    pip install http://github.com/soft-matter/yaml-serialize/zipball/master
    pip install http://github.com/soft-matter/pims/zipball/master

And finally, install ``trackpy`` itself.

    pip install http://github.com/soft-matter/trackpy/zipball/master

Optional Dependencies:

  * [``cv2``](http://opencv.org/downloads.html) for reading video files
      and viewing video with annotations
  * ``PyTables`` for saving results in an HDF5 file
  * ``sqlite`` or ``MySQLdb`` for saving results in a SQL database
  * [``pyFFTW``](https://github.com/hgomersall/pyFFTW) to speed up the band
      pass, which is one of the slower steps in feature-finding

To load video files directly, you need OpenCV. You can work around this
requirement by converting any video files to folders full of images
using a utility like [ImageJ](http://rsb.info.nih.gov/ij/). Reading folders
of images is supported out of the box, without OpenCV.

* Linux: OpenCV is included with Anaconda
* OSX: OpenCV is easy to install on OSX using [homebrew](http://brew.sh/).
* Windows: OpenCV can be installed on Windows in a few steps, outlined below.
It is not as simple as the steps above, so beginners are encouraged
to experiment with a folder full of images first.

### Installing OpenCV on Windows

1. Install the video software FFmepg using this [Windows installer](http://www.arachneweb.co.uk/software/windows/avchdview/FFmpegSetup.exe)
Make note of the directory where it is installed. It can be anywhere but, whatever it is,
you will need to know that location in the next step.
2. Right click on Computer (or perhaps "My Computer"), and click Properties.
Click "Advanced System Settings", then "Properties". With "Path" highlighted,
click "Edit." This is a list of file paths separated by semicolons, you must
type in an additional entry. ";C:\Program Files (x86)\ffmpeg" or wherever
FFmpeg was installed in Step 1.
3. Install the Windows 32 (Python 2.7) version of OpenCV available on [this page](http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv).
4. Download [OpenCV for Windows](http://opencv.org/).
5. You will now have a folder called ``opencv``. We just need one file
from this to make everything work.
6. Copy the file ``opencv\3rdparty\ffmpeg\opencv_ffmpeg.dll``.
7. Navigate to the directory where ffmpeg was installed, which you noted
in Step 1. From this directory, navigate into ``win32-static\bin``.
Paste ``opencv_ffmpeg.dll`` here.

Now run ``ipython``. If you can execute ``import cv`` without any errors, the
installation is probably successful. If you can read video files using
``trackpy.Video('path/to/video_file.avi')`` then the installation is definitely working
as expected.


### Updating Your Instllation

The code is under active development. To update to the current development
version, run this in the command prompt:

    pip install --upgrade http://github.com/soft-matter/trackpy/zipball/master

Verions & Project Roadmap
-------------------------

A version 0.1 has been tagged and the v0.1.x branch will get bug
fixes.  This version does not depend on `pandas`.

Any new features will be built on the master branch.  The new
features include:

 - merging most of Dan Allan's `mr` module
 - replacing `identification.py` with superior `feature.py`
 - making `link` iterative
 - merging Nathen Kiem's `numba` accelerated branch


Contributors
------------

* **Daniel Allan** feature-finding, uncertainty estimation,
motion characterization and discrimination, plotting tools, tests
* **Thomas Caswell** multiple implementations of sophisticated trajectory-linking, tests
* **Nathan Keim** alternative trajectory-linking implementations and major
speed-ups

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
| Thomas A Casswell | https://github.com/tacaswell/tracking | C++ |

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
