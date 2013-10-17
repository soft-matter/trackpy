import os
import setuptools
from numpy.distutils.core import setup, Extension

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

MAJOR = 0
MINOR = 3
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''

FULLVERSION = VERSION
print FULLVERSION

if not ISRELEASED:
    FULLVERSION += '.dev'
    print FULLVERSION
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

        print FULLVERSION
        FULLVERSION = rev.lstrip('v')
        print FULLVERSION

    except:
        import warnings
        warnings.warn("WARNING: Couldn't get git revision")
else:
    FULLVERSION += QUALIFIER

print FULLVERSION

def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'mr', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()

write_version_py()

setup(
    name = "mr",
    version = FULLVERSION,
    description = "microrheology toolkit",
    author = "Daniel Allan",
    author_email = "dallan@pha.jhu.edu",
    url = "https://github.com/danielballan/mr",
    packages = ['mr', 'mr.core', 'mr.video', 'mr.wire', 'mr.uncertainty'],
    long_description = read('README.md'),
    ext_modules = [Extension('_Cfilters', ['mr/src/Cfilters.c'])],
    package_dir = {'mr': 'mr'},
    package_data = {'mr': ['doc/*', 'db_schema.sql']},
)
