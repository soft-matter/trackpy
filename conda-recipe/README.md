What is this?
-------------

This directory contains a recipe for building a 
[conda](http://conda.pydata.org/docs/index.html) package for trackpy. It is 
configured for creating "nightly builds," snapshots of the code in development 
between releases. The built packages are uploaded to trackpy's 
[dev channel on binstar](https://binstar.org/soft-matter/trackpy/channels),
where they can easily be installed on any platform as described in the 
doucmentation.

Tagged releases are built using a modified recipe that includes the MD5 hash
of the released code. They are uploaded to trackpy's main channel on binstar.

Building a Package
------------------

See more detailed background information in the
[release checklist](https://github.com/soft-matter/trackpy/wiki/Release-Checklist).
This is just a handy cheatsheet.

    cd trackpy/conda-recipe
    conda build . --no-binstar-upload --python=2.7
    conda build . --no-binstar-upload --python=3.3
    # Note: As of this writing, trackpy does not actually support a version
    # of numba compatible with Python 3.4.

    conda convert `conda build . --output --python=2.7` --platform all
    conda convert `conda build . --output --python=3.3` --platform all


    binstar upload win-32/trackpy* -u soft-matter -c dev
    binstar upload win-62/trackpy* -u soft-matter -c dev
    binstar upload linux-32/trackpy* -u soft-matter -c dev
    binstar upload linux-64/trackpy* -u soft-matter -c dev
    binstar upload osx-64/trackpy* -u soft-matter -c dev
