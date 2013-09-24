import os
import setuptools
from numpy.distutils.core import setup, Extension

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

from mr import version

setup(
    name = "mr",
    version = __version__,
    description = "microrheology toolkit",
    author = "Daniel Allan",
    author_email = "dallan@pha.jhu.edu",
    url = "https://github.com/danielballan/mr",
    packages = ['mr'],
    long_description = read('README.md'),
    ext_modules = [Extension('_Cfilters', ['mr/src/Cfilters.c'])],
    package_dir = {'mr': 'mr'},
    package_data = {'mr': ['doc/*', 'db_schema.sql']},
)
