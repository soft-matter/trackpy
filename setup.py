import os
from distutils.core import setup, Extension

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "mr",
    version = "0.1",
    description = "microrheology toolkit",
    author = "Daniel Allan",
    author_email = "dallan@pha.jhu.edu",
    url = "https://github.com/danielballan/mr",
    packages=['mr'],
    long_description=read('README.rst'),
    ext_modules=[Extension('_Cfilters', ['mr/src/Cfilters.c'])]
)
