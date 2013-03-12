=========================
mr: microrheology toolkit
=========================

What is it
==========

**mr** is a Python package providing tools for passive and active microrheology experiments.

Main Features
=============

    - The widely-used particle locating algorithm originally implemented
      by John Crocker and Eric Weeks in IDL is reimplemented in
      Python. Wherever possible, I use existing tools from scipy and numpy modules. 
      Each logical step is broken into a single-purpose function,
      rendering a succinct and modular code that is easy to customize and maintain. Key
      array-intensive steps that are not available from Python's standard scientific modules are
      written in C and imported.
    - The trajectories of colloidal probes can be characteristized, grouped, and
      plotted using a suite of convenient functions.
    - Various models (with more to come) relate probe statistics to rheological response, including
      the Generalized Stokes-Einstein implementation used in the Crocker/Grier code.
    - A sql module provides convenient functions for storing and loading data
      from a MySQL database. (A sample database schema is included.)
    - A wrapper for the powerful video handling software FFmpeg slices video and helps with book-keeping.


Dependencies
============

  * numpy
  * scipy
  * pandas
  * MySQLdb (optional)
  * [trackpy](https://github.com/tacaswell/trackpy)

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
