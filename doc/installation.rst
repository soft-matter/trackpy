.. _installation:

Installing Trackpy
------------------

For Python Novices
^^^^^^^^^^^^^^^^^^

Installation is simple on Windows, OSX, and Linux, even for Python novices.

1. Get Scientific Python
""""""""""""""""""""""""

To get started with Python on any platform, download and install
`Anaconda <https://www.anaconda.com/distribution/>`_. It comes with the
common scientific Python packages built in.

2. Install trackpy
""""""""""""""""""

Open a command prompt. On Windows, you can use the "Anaconda Command Prompt"
installed by Anaconda or Start > Applications > Command Prompt. On a Mac, look
for Applications > Utilities > Terminal. Type these commands:

.. code-block:: bash

   conda update conda
   conda install -c conda-forge trackpy
   conda install -c conda-forge pims

The above installs trackpy and all its requirements, plus the recommended
`PIMS <http://soft-matter.github.io/pims/>`_ package that simplifies image-reading,
and that is used in the trackpy tutorials.

3. Try it out!
""""""""""""""

Finally, to try it out, type

.. code-block:: bash

    jupyter notebook

.. note::  For older Python versions, use ``ipython notebook``

This will automatically open a browser tab, ready to interpret Python code.
To get started, check out the links to tutorials at the top of this document.

Updating Your Installation
""""""""""""""""""""""""""

Before updating to a new version of trackpy, be sure to read the
:doc:`release notes<whatsnew>` for a list of new features and any changes
that may affect your existing analysis code.

Latest Stable Release
^^^^^^^^^^^^^^^^^^^^^

The code is under active development. To update to the latest stable release,
run this in the command prompt:

.. code-block:: bash

    conda update -c conda-forge trackpy

Latest Version Under Development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `master` branch on github contains the latest tested development code.
Changes are thoroughly tested before being merged. If you want to use the
latest features it should be safe to rely on the master branch.
(The primary contributors do.)

You can easily install a recent build by downloading the source from
`Github <https://github.com/soft-matter/trackpy>`_:

.. code-block:: bash

    pip install https://github.com/soft-matter/trackpy/archive/master.zip

If you plan to edit the code yourself, you should use git and pip as
explained below.

Using environments
""""""""""""""""""

Some users often like to separate Python instances into environments, so that
each project can have its own set of packages. Create a trackpy-dedicated
environment with the name softmatter as follows:

.. code-block:: bash

    conda create --name softmatter python=3.7 trackpy nb_conda

The `nb_conda` is optional, but we added it to ensure that Jupyter sees this
environment as well. You can switch to the environment from within Jupyter in
the Kernels menu. To access the environment on the commandline, type the
following:

.. code-block:: bash

    source activate softmatter

On Windows systems, leave out the `source`. You can go back to the root conda
environment by activating `root`.


More Information for Experienced Python Users
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Archlinux
"""""""""

Package is available for Archlinux on AUR: `Python 3 <https://aur.archlinux.org/packages/python-trackpy/>`__

pip
"""

We strongly recommend using conda to install trackpy, as described above,
but pip is also supported.

Essential Dependencies:

* Python 3.7 or newer. (trackpy on Python 2.7 and older Python 3.x is no longer officially supported, though it is still likely to run.)
* `setuptools <http://pythonhosted.org/setuptools/>`__
* `numpy <http://www.scipy.org/>`__
* `scipy <http://www.scipy.org/>`__
* `matplotlib <http://matplotlib.org/>`__
* `pandas <http://pandas.pydata.org/pandas-docs/stable/overview.html>`__
* `pyyaml <http://pyyaml.org/>`__
* `looseversion <https://pypi.org/project/looseversion/>`__

You will also want to install the `pims <http://soft-matter.github.io/pims/>`_
package that simplifies image-reading, and that is used in the trackpy tutorials.
The remaining optional dependencies, listed below, are strongly recommended but
can be challenging to install yourself, hence our recommendation that you
use `Anaconda <https://www.anaconda.com/distribution/>`_.

Manual installation
"""""""""""""""""""

If you want to be able to edit the code yourself, you can install the package
manually. First, make sure you have `git <https://git-scm.com/>`__ version
management software installed. Go to a folder where you want to have your
source code, then:

.. code-block:: bash

   git clone https://github.com/soft-matter/trackpy
   cd trackpy
   python setup.py develop

We welcome any contribution to the trackpy source code, so feel free to send
in your contributions on Github! To do so, make an account, fork
`trackpy <https://github.com/soft-matter/trackpy>`__ and create a local copy
using:

.. code-block:: bash

   git clone https://github.com/<your_account>/trackpy

Now you have a local copy of the code which you can edit, but don't start
editing right away as you are currently on the ``master`` branch. We think it
is good practice to keep your ``master`` branch mirroring the upstream
trackpy version, so first create a new branch and push it to the remote as
follows:

.. code-block:: bash

   git branch fix-something
   git push --set-upstream origin fix-something

Now you can edit your code in any way you like, commit your changes, and push
them again to the remote.

Before sending in your code, please consult
`our guidelines <https://github.com/soft-matter/trackpy/wiki/Information-for-contributors>`__.
Also, see `here <https://git-scm.com/book/en/v1/Getting-Started>`__ for getting
started using git.

Optional Dependencies
"""""""""""""""""""""

These are strongly recommended to make using trackpy more convenient and faster:

* The `PIMS <http://soft-matter.github.io/pims/>`_ package simplifies image-reading,
      and is used in the trackpy tutorials.
* `PyTables <http://www.pytables.org/moin>`__ for saving results in an HDF5 file. 
      This is included with Anaconda.
* `numba <http://numba.pydata.org/>`__ for accelerated feature-finding and linking. 
      This is included with Anaconda and Canopy. Installing it any other way is
      difficult; we recommend sticking with one of these.
* `Pillow <https://pillow.readthedocs.org/>`__ or `PIL <http://www.pythonware.com/products/pil/>`__ for some display routines.
      This is included with Anaconda.

PIMS has its own optional dependencies for reading various formats. You
can read what you need for each format
`here on PIMS' README <http://soft-matter.github.io/pims/stable>`__.
