=========================
mr: microrheology toolkit
=========================

What is it
==========

**mr** is a Python package providing tools for passive and active microrheology experiments, including multiple particle tracking, analysis of colloidal trajectories, and rheological models.

Main Features
=============

    - The widely used multiple particle tracking algorithm originally implemented
      by John Crocker and Eric Weeks in IDL is faithfully reimplemented in
      Python. Much is accomplished using the standard scipy and numpy modules,
      and each logical step is broken into a single-purpose function,
      rendering a succinct and modular code that is easy to customize. Key
      array-intensive steps that are not available from Python's standard scientific modules are
      imported from C.
    - The trajectories of colloidal probes can be characteristized, grouped, and
      plotted using a suite of simple, convenient functions.
    - Various models relate probe statistics to rheological response, including
      the all-important Generalized Stokes-Einstein and others, particularly
      models developed for interfacial microrheology. 
    - A sql module provides convenient functions for storing and loading data
      from a MySQL database. (A sample database schema is included.)
    - A wrapper for the powerful video handling software FFmpeg helps slice
      sections of video to analyze and ameliorate book keeping headaches.


Dependencies
============

  * numpy
  * scipy
  * MySQLdb (optional)
  * pIDLy, but until track.pro can be reimplemented in Python

Project Status
==============
The feature-finding is stable. This packages still relies on Crocker & Weeks's 
``track.pro`` to link feature positions into trajectories. Once these are 
obtained, the package provides tools for flitering, grouping, and analyzing 
them.

Related Projects
================

  * Particle tracking using IDL http://www.physics.emory.edu/~weeks/idl/
  * A C++ implementation wrapped in Python https://github.com/tacaswell/tracking

Background
==========

This package was developed and is maintained by Daniel Allan, as part of his
PhD thesis work on microrheology in Robert L. Leheny's group at Johns Hopkins
University in Baltimore, MD.

Dan can be reached through his website, http://www.danallan.com.
