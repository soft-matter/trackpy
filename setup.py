import os
import versioneer
from setuptools import setup


try:
    descr = open(os.path.join(os.path.dirname(__file__), 'README.md')).read()
except OSError:
    descr = ''

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
    install_requires = ['numpy>=1.18', 'scipy>=1.4', 'pandas>=1', 'pyyaml', 'matplotlib', "looseversion>=1.0.1"],
    extras_require={"test": "pytest"},
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages = ['trackpy', 'trackpy.refine', 'trackpy.linking', 'trackpy.locate_functions'],
    long_description = descr,
    long_description_content_type='text/markdown'
)

setup(**setup_parameters)
