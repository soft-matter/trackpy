import os
import versioneer
from setuptools import setup


try:
    descr = open(os.path.join(os.path.dirname(__file__), 'README.md')).read()
except IOError:
    descr = ''

try:
    from pypandoc import convert
    descr = convert(descr, 'rst', format='md')
except ImportError:
    pass


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
    install_requires = ['numpy>=1.7', 'scipy>=0.12', 'six>=1.8',
	                    'pandas>=0.13', 'pims>=0.3.3',
                        'pyyaml', 'matplotlib'],
    packages = ['trackpy'],
    long_description = descr,
)

setup(**setup_parameters)
