.. _installation:

Installing Trackpy
------------------

For Python Novices
^^^^^^^^^^^^^^^^^^

Installation is simple on Windows, OSX, and Linux, even for Python novices.

1. Get Scientific Python
""""""""""""""""""""""""

To get started with Python on any platform, download and install
`Anaconda <(https://store.continuum.io/cshop/anaconda/>`_. It comes with the
common scientific Python packages built in.

2. Install trackpy
""""""""""""""""""

Open a command prompt. On Windows, you can use the "Anaconda Command Prompt"
installed by Anaconda or Start > Applications > Command Prompt. On a Mac, look
for Applications > Utilities > Terminal. Type these commands:

.. code-block:: bash

   conda update conda
   conda install -c soft-matter trackpy

The above installs trackpy and all its requirements. Our tutorials also use
the IPython notebook. To install that as well, type

.. code-block:: bash

    conda install ipython-notebook

3. Try it out!
""""""""""""""
    
Finally, to try it out, type

.. code-block:: bash

    ipython notebook

This will automatically open a browser tab, ready to interpret Python code.
To get started, check out the links to tutorials at the top of this document.

Updating Your Installation
--------------------------

Latest Stable Release
^^^^^^^^^^^^^^^^^^^^^

The code is under active development. To update to the latest stable release,
run this in the command prompt:

.. code-block:: bash

    conda update -c soft-matter trackpy

Latest Version Under Development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `master` branch on github contains the latest tested development code.
Changes are thoroughly tested before being merged. If you want to use the
latest features it should be safe to rely on the master branch.
(The primary contributors do.)

You can easily install a recent build from the
soft-matter development channel on conda

.. code-block:: bash

    conda config --add channels soft-matter
    conda install -c soft-matter/channel/dev trackpy

If you plan to edit the code yourself, you should use git and pip as 
explained below.

More Information for Experienced Python Users
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We strongly recommend using conda install trackpy, as described above,
but pip is also supported.

Essential Dependencies:

  * Python 2.7, 3.3, or 3.4.
  * `setuptools <http://pythonhosted.org/setuptools/>`__
  * `six <http://pythonhosted.org/six/>`__ >=1.8
  * `numpy <(http://www.scipy.org/>`__ >=1.7
  * `scipy <(http://www.scipy.org/>`__ >=0.12.0
  * `matplotlib <(http://matplotlib.org/>`__
  * `pandas <http://pandas.pydata.org/pandas-docs/stable/overview.html>`__ >=0.12.0
  * `pyyaml <(http://pyyaml.org/>`__

You will also need the image- and video-reader PIMS, which is, like trackpy
itself, part of the github.com/soft-matter organization.

  * `PIMS <https://github.com/soft-matter/pims>`__

You can install PIMS from the soft-matter binstar channel using conda:

.. code-block:: bash

   conda install -c soft-matter pims

or from PyPI using pip:

.. code-block:: bash

   pip install pims

Or, if you plan to edit the code, you can install both packages manually:
  
.. code-block:: bash

   git clone https://github.com/soft-matter/pims
   pip install -e pims

   git clone https://github.com/soft-matter/trackpy
   pip install -e trackpy

Optional Dependencies:

  * `pyFFTW <https://github.com/hgomersall/pyFFTW>`__ to speed up the band
      pass, which is one of the slower steps in feature-finding
  * `PyTables <http://www.pytables.org/moin>`__ for saving results in an 
      HDF5 file. This is included with Anaconda.
  * `numba <http://numba.pydata.org/>`__ for accelerated feature-finding and linking. This is
      included with Anaconda and Canopy. Installing it any other way is difficult;
      we recommend sticking with one of these. Note that numba v0.12.0
      (included with Anaconda 1.9.0) has a bug and will not work at all;
      if you have this version, you should update Anaconda. We support numba 
      versions 0.11 and 0.12.2.

PIMS has its own optional dependencies for reading various formats. You
can read what you need for each format
`here on PIMS' README <(https://github.com/soft-matter/pims>`__.

