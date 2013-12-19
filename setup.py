#!/usr/bin/env python

import setuptools
from distutils.core import setup

print("Installing trackpy")

setup(
    name='trackpy',
    version='0.1',
    author='Thomas A Caswell',
    author_email='tcaswell@uchicago.edu',
    url='https://github.com/soft-matter/trackpy',
    packages=['trackpy'],
    install_requires=['numpy', 'six', 'scipy']
    )
