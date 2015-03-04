.. raw:: html

    <style type="text/css">
    .thumbnail {{
        position: relative;
        float: left;
        margin: 10px;
        width: 180px;
        height: 200px;
    }}

    .thumbnail img {{
        position: absolute;
        display: inline;
        left: 0;
        width: 170px;
        height: 170px;
    }}

    </style>


Trackpy: Fast, Flexible Particle-Tracking Toolkit
=================================================

.. raw:: html

    <div style="clear: both"></div>
    <div class="container-fluid hidden-xs hidden-sm">
      <div class="row">
        <div class="col-md-2 thumbnail">
          <img src="_static/interfacial-particles.png">
        </div>
        <div class="col-md-2 thumbnail">
          <img src="_static/foam.png">
        </div>
        <div class="col-md-2 thumbnail">
          <img src="_static/tracking-sphere.png">
        </div>
        <div class="col-md-2 thumbnail">
          <img src="_static/trajectories-in-water.png">
        </div>
        <div class="col-md-2 thumbnail">
          <img src="_static/rearrangements-and-strain.png">
        </div>
        <div class="col-md-2 thumbnail">
          <img src="_static/large-particle-in-liquid-crystal.png">
        </div>
      </div>
    </div>
    <br>


Trackpy is a Python package for particle tracking in 2D, 3D, and higher dimensions.

For a brief introduction to the ideas behind the package, you can read the :ref:`introductory notes <introduction>`. Read the :doc:`walkthrough <tutorial/walkthrough>` to study an example project from start to finish.

Much more detail can be found in the trackpy :ref:`tutorial <tutorial>`. You can also browse the :ref:`API reference <api_ref>` to see available tools for tracking, motion analysis, plotting, and more.

To check out the code, report a bug, or contribute a new feature, please visit
the `github repository <https://github.com/soft-matter/trackpy>`_.

This much-expanded documentation is a recent addition to trackpy, and
it describes version 0.3, which is still in development, to be released in
Spring 2015. From the release of v0.3 onward, we will be maintain
separate documentation for each version. See the
:doc:`installation instructions <installation>` to obtain the current stable
release (v0.2.4) or the latest version in development. The majority of this
documentation applies to v0.2.4 as well as v0.3, but some of the
:doc:`newer functionality <whatsnew>` is only available in v0.3.

.. raw:: html

   <div class="container-fluid">
   <div class="row">
   <div class="col-md-6">
   <h2>Documentation</h2>

.. toctree::
   :maxdepth: 1

   introduction
   installation
   api
   whatsnew

.. raw:: html

   </div>
   <div class="col-md-6">
   <h2>Tutorial</h2>

.. toctree::
   :maxdepth: 1

   tutorial/walkthrough
   tutorial/prediction
   tutorial/tracking-3d
   tutorial/uncertainty
   tutorial/subnets
   tutorial/adaptive-search
   tutorial/on-disk
   tutorial/performance
   tutorial/linking-diagnostics
   tutorial/parallel-locate
   tutorial/custom-feature-detection

.. raw:: html

   </div>
   </div>
   </div>
