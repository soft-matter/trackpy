.. _introduction:

Introduction to Trackpy
-----------------------

Trackpy is a package for tracking blob-like features in video images, following them
through time, and analyzing their trajectories. It started from a Python implementation
of the widely-used Crocker--Grier algorithm and is currently in transition
towards a general-purpose Python tracking library.

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
Following the `widely-used particle tracking algorithm <http://www.physics.emory.edu/~weeks/idl/tracking.html>`__,
we separate *tracking* into three separate steps. In the first step, *feature finding*
initial feature coordinates are obtained from the images. Subsequently, sub-pixel precision
is obtained in coordinate *refinement*. Finally, the coordinates are *linked* in time yielding
the feature trajectories.

  * The tracking algorithm originally implemented by John Crocker and Eric Weeks in IDL was
    completely reimplemented in Python.
  * A `flexible framework for least-squares fitting <https://arxiv.org/abs/1607.08819>`__
    allows for sub-pixel refinement using any radial model function in 2D and 3D.
  * Trackpy is actively used and tested on **Windows, Mac OSX, and Linux**,
    and it uses only **free, open-source** software.
  * Frames of video are loaded via the sister project `PIMS <http://github.com/soft-matter/pims>`__
    which enables reading of several types of **video files (AVI, MOV, etc.),
    specialized formats (LEI, ND2, SEQ, CINE), multi-frame TIFF, or a directory of sequential
    images (TIFF, PNG, JPG, etc.)**.
  * Results are given as DataFrames, high-performance spreadsheet-like objects
    from `Python pandas <http://pandas.pydata.org/pandas-docs/stable/overview.html>`__
    which can easily be saved to a **CSV file, Excel spreadsheet,
    SQL database, HDF5 file**, and more.
  * Particle trajectories can be
    characterized, grouped, and plotted using a suite of convenient functions.
  * To verify correctness and stability, a **suite of 500+ tests verifies basic results
    on each trackpy update**.

Special Capabilities
""""""""""""""""""""

  * Both feature-finding and trajectory-linking can be performed on
    **arbitrarily long videos** using a fixed, modest amount of memory. (Results
    can be read and saved to disk throughout.)
  * A **prediction framework** helps track particles in fluid flows,
    or other scenarios where velocity is correlated between time steps.
  * Feature-finding optionally makes use of the **history of feature coordinates**
    in a routine that combines linking and feature-finding.
  * Feature-finding and trajectory-linking works on **images with any number of dimensions**,
    making possible some creative applications.
  * **Uncertainty is estimated** following a method `described in this paper <http://dx.doi.org/10.1529/biophysj.104.042457>`__ by Savin and Doyle.
  * **High-performance** numba acceleration is used only if
    if available. Since these can be tricky to install on some machines,
    the code will automatically fall back on slower pure Python implementations
  * **Adaptive search** can prevent the tracking algorithm from failing
    or becoming too slow, by automatically making adjustments when needed.

Citing Trackpy
^^^^^^^^^^^^^^

Trackpy can be cited using a DOI provided through our Zenodo
`record page <https://zenodo.org/badge/latestdoi/4744355>`_. To direct your
readers to the specific version of trackpy that they can use to reproduce
your results, cite the release of trackpy that you used for your work
(available from the variable ``trackpy.__version__``). The
record pages linked below contain author lists, other details, and complete
citations in various formats. If your citation style allows for a URL,
please include a link to the github repository:
`github.com/soft-matter/trackpy`.

================= ========================================================================= ======================
Release (version) Zenodo Record Pages with info and citations                               DOI
================= ========================================================================= ======================
v0.4 and later    `Versioned Record Page <https://zenodo.org/badge/latestdoi/4744355>`__    (see Zenodo)
v0.3.2            `Record Page <https://zenodo.org/record/60550>`__                         10.5281/zenodo.60550
v0.3.1            `Record Page <https://zenodo.org/record/55143>`__                         10.5281/zenodo.55143
v0.3.0            `Record Page <https://zenodo.org/record/34028>`__                         10.5281/zenodo.34028
v0.2.4            `Record Page <https://zenodo.org/record/12255>`__                         10.5281/zenodo.12255
v0.2.3            `Record Page <https://zenodo.org/record/11956>`__                         10.5281/zenodo.11956
v0.2.2            `Record Page <https://zenodo.org/record/11132>`__                         10.5281/zenodo.11132
v0.2              `Record Page <https://zenodo.org/record/9971>`__                          10.5281/zenodo.9971
================= ========================================================================= ======================

Users often also cite this publication describing the core feature-finding
and linking algorithms that trackpy is based on:

Crocker, J. C., & Grier, D. G. (1996). Methods of Digital Video Microscopy for Colloidal Studies.
J. Colloid Interf. Sci., 179(1), 298–310. http://doi.org/10.1006/jcis.1996.0217

Related Projects
^^^^^^^^^^^^^^^^

============================ =================================================== =========================
Author(s)                    Project URL                                         Language
============================ =================================================== =========================
Crocker and Grier            http://physics.nyu.edu/grierlab/software.html       IDL
Crocker and Weeks            http://www.physics.emory.edu/~weeks/idl/            IDL
Blair and Dufresne           http://physics.georgetown.edu/matlab/               MATLAB
Maria Kilfoil et al.         https://github.com/rmcgorty/ParticleTracking-Python Python
Graham Milne                 http://zone.ni.com/devzone/cda/epd/p/id/948         LabVIEW
Ryan Smith and Gabe Spalding http://titan.iwu.edu/~gspaldin/rytrack.html         stand alone/IDL GUI
Peter J Lu                   https://github.com/peterlu/PLuTARC_centerfind2D     C++ (identification only)
Thomas A Caswell             https://github.com/tacaswell/tracking               C++
============================ =================================================== =========================

Core Contributors
^^^^^^^^^^^^^^^^^

  * **Casper van der Wel** anisotropic 3D feature-finding, plotting and analyses, framework
    for least-squares refinement, combined linking and feature finding
  * **Daniel Allan** feature-finding, uncertainty estimation,
    motion characterization and discrimination, plotting tools, tests
  * **Nathan Keim** alternative trajectory-linking implementations, major
    speed-ups, prediction, adaptive search
  * **Thomas Caswell** multiple implementations of sophisticated trajectory-linking, tests


Support
^^^^^^^

This package was developed in part by Daniel Allan, as part of his
PhD thesis work on microrheology in Robert L. Leheny's group at Johns Hopkins
University in Baltimore, MD, USA. The work was supported by the National Science Foundation
under grant number CBET-1033985.  Dan can be reached at dallan@pha.jhu.edu.

This package was developed in part by Thomas A Caswell as part of his
PhD thesis work in Sidney R Nagel's and Margaret L Gardel's groups at
the University of Chicago, Chicago IL, USA.  This work was supported in
part by NSF Grant DMR-1105145 and NSF-MRSEC DMR-0820054.  Tom can be
reached at tcaswell@gmail.com.

This package was developed in part by Nathan C. Keim at Cal Poly,
San Luis Obispo, California, USA and supported by NSF Grant DMR-1708870.
Portions were also developed at the University of Pennsylvania,
Philadelphia, USA, supported by NSF-MRSEC DMR-1120901.

This package was developed in part by Casper van der Wel, as part of his
PhD thesis work in Daniela Kraft’s group at the Huygens-Kamerlingh-Onnes laboratory,
Institute of Physics, Leiden University, The Netherlands. This work was
supported by the Netherlands Organisation for Scientific Research (NWO/OCW). 
