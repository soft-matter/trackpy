import os
import sys
import warnings
import setuptools
import versioneer
from setuptools import setup, Extension


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
