# This downloads and install setuptools if it is not installed.
from ez_setup import use_setuptools
use_setuptools()

import os
import setuptools
from numpy.distutils.core import setup, Extension
import warnings

MAJOR = 0
MINOR = 6
MICRO = 0
ISRELEASED = False 
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''

FULLVERSION = VERSION
print FULLVERSION

if not ISRELEASED:
    FULLVERSION += '.dev'
    try:
        import subprocess
        try:
            pipe = subprocess.Popen(["git", "describe", "HEAD"],
                                    stdout=subprocess.PIPE).stdout
        except OSError:
            # msysgit compatibility
            pipe = subprocess.Popen(
                ["git.cmd", "describe", "HEAD"],
                stdout=subprocess.PIPE).stdout
        rev = pipe.read().strip()
        # makes distutils blow up on Python 2.7
        import sys
        if sys.version_info[0] >= 3:
            rev = rev.decode('ascii')

        FULLVERSION = rev.lstrip('v')

    except:
        warnings.warn("WARNING: Couldn't get git revision")
else:
    FULLVERSION += QUALIFIER

def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'trackpy', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()

write_version_py()

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# Installation tries ext_modules but omits them if compiling fails.
setup_parameters = dict(
    name = "trackpy",
    version = FULLVERSION,
    description = "particle-tracking toolkit",
    author = "Daniel Allan and Thomas Caswell",
    author_email = "dallan@pha.jhu.edu",
    url = "https://github.com/soft-matter/trackpy",
    install_requires = ['numpy', 'scipy', 'six', 'pandas', 'yaml-serialize',
                        'pims'],
    dependency_links = ['http://github.com/soft-matter/pims/zipball/master',
                        'http://github.com/soft-matter/yaml-serialize/zipball/master'],
    packages = ['trackpy'],
    long_description = read('README.md'),
    ext_modules = [Extension('_Cfilters', ['trackpy/src/Cfilters.c'])],
)

try:
    setup(**setup_parameters)
except SystemExit:
    warnings.warn(
        """DON'T PANIC! Compiling C is not working, so I will fall back
on pure Python code that does the same thing, just slower.""")
    # Try again without ext_modules.
    del setup_parameters['ext_modules']
    setup(**setup_parameters)
