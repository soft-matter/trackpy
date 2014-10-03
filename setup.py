# This downloads and install setuptools if it is not installed.
from ez_setup import use_setuptools
use_setuptools()

import os
import sys
import warnings

# try bootstrapping setuptools if it doesn't exist
try:
    import pkg_resources
    try:
        pkg_resources.require("setuptools>=0.6c5")
    except pkg_resources.VersionConflict:
        from ez_setup import use_setuptools
        use_setuptools(version="0.6c5")
    from setuptools import setup, Extension
    _have_setuptools = True
except ImportError:
    # no setuptools installed
    from numpy.distutils.core import setup
    _have_setuptools = False

import versioneer
versioneer.VCS = 'git'
versioneer.versionfile_source = 'trackpy/_version.py'
versioneer.versionfile_build = 'trackpy/_version.py'
versioneer.tag_prefix = 'v'
versioneer.parentdir_prefix = '.'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# In some cases, the numpy include path is not present by default.
# Let's try to obtain it.
try:
    import numpy
except ImportError:
    ext_include_dirs = []
else:
    ext_include_dirs = [numpy.get_include(),]

setup_parameters = dict(
    name = "trackpy",
    version = versioneer.get_version(),
    cmdclass = versioneer.get_cmdclass(),
    description = "particle-tracking toolkit",
    author = "Trackpy Contributors",
    author_email = "daniel.b.allan@gmail.com",
    url = "https://github.com/soft-matter/trackpy",
    install_requires = ['numpy', 'scipy', 'six', 'pandas',
                        'pyyaml', 'matplotlib', 'pims'],
    packages = ['trackpy'],
    long_description = read('README.md'),
)

setup(**setup_parameters)
