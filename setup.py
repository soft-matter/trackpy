import os
import versioneer
from setuptools import setup


try:
    descr = open(os.path.join(os.path.dirname(__file__), 'README.md')).read()
except OSError:
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
    install_requires = ['numpy>=1.14', 'scipy>=1.1', 'pandas>=0.22', 'pyyaml', 'matplotlib'],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages = ['trackpy', 'trackpy.refine', 'trackpy.linking', 'trackpy.locate_functions'],
    long_description = descr,
)

setup(**setup_parameters)
