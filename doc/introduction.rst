.. _introduction:

Introduction to Trackpy
-----------------------

Trackpy is a package for tracking blob-like particles in video and analyzing
their trajectories. It implements and extends the widely-used Crocker--Grier
algorithm in Python.

There are many similar projects. (See table below.)
Our implementation is distinguished by succinct and flexible usage,
a thorough testing framework ensuring code stability and accuracy,
scalability, and thorough documentation. 

Several researchers have merged their independent efforts into this code.
We would like to see others in the community adopt it and potentially
contribute code to it.

Features
^^^^^^^^

Basics
""""""

  * The `widely-used particle tracking algorithm <http://www.physics.emory.edu/~weeks/idl/tracking.html>`__
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
    from `Python pandas <http://pandas.pydata.org/pandas-docs/stable/overview.html>`__
    which can easily be saved to a **CSV file, Excel spreadsheet,
    SQL database, HDF5 file**, and more.
  * Particle trajectories can be
    characterized, grouped, and plotted using a suite of convenient functions.
  * To verify correctness and stability, a **suite of 150+ tests reproduces
    basic results**.

Special Capabilities
""""""""""""""""""""

  * Both feature-finding and trajectory-linking can be performed on
    **arbitrarily long videos** using a fixed, modest amount of memory. (Results
    can be read and saved to disk throughout.)
  * A **prediction framework** helps track particles in fluid flows,
    or other scenarios where velocity is correlated between time steps.
  * Feature-finding and trajectory-linking works on **images with any number of dimensions**,
    making possible some creative applications.
  * **Uncertainty is estimated** following a method `described in this paper <http://dx.doi.org/10.1529/biophysj.104.042457>`__ by Savin and Doyle.
  * **High-performance** components (numba acceleration and FFTW support) are used only if
  if available. Since these can be tricky to install on some machines,
  the code will automatically fall back on slower pure Python implementations

Citing Trackpy
^^^^^^^^^^^^^^

If you use trackpy for published research, please cite this repository,
including the primary contributors' names -- Daniel B. Allan, Nathan C. Keim, and Thomas A. Caswell,
-- and ``doi:10.5281/zenodo.9971``.
If your citation style *also* allows for a URL,
please include `github.com/soft-matter/trackpy` to help other
researchers discover trackpy. Our
`DOI record pages <https://zenodo.org/search?ln=en&p=trackpy>`__
provides more detail and citations in various formats.

Related Projects
^^^^^^^^^^^^^^^^
 
============================ =============================================== =========================
Author(s)                    Project URL                                     Language
============================ =============================================== =========================
Crocker and Grier            http://physics.nyu.edu/grierlab/software.html   IDL
Crocker and Weeks            http://www.physics.emory.edu/~weeks/idl/        IDL
Blair and Dufresne           http://physics.georgetown.edu/matlab/           MATLAB
Maria Kilfoil                http://people.umass.edu/kilfoil/downloads.html  MATLAB and Python
Graham Milne                 http://zone.ni.com/devzone/cda/epd/p/id/948     LabVIEW
Ryan Smith and Gabe Spalding http://titan.iwu.edu/~gspaldin/rytrack.html     stand alone/IDL GUI
Peter J Lu                   https://github.com/peterlu/PLuTARC_centerfind2D C++ (identification only)
Thomas A Caswell             https://github.com/tacaswell/tracking           C++
============================ =============================================== =========================

Core Contributors
^^^^^^^^^^^^^^^^^

  * **Daniel Allan** feature-finding, uncertainty estimation,
    motion characterization and discrimination, plotting tools, tests
  * **Nathan Keim** alternative trajectory-linking implementations, major
    speed-ups, prediction
  * **Thomas Caswell** multiple implementations of sophisticated trajectory-linking, tests


Support
^^^^^^^

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
