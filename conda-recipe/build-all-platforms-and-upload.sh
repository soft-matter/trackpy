#!/bin/sh
VERSIONS="27 33 34"

for VERSION in $VERSIONS; do
	conda build . --python=$VERSION;
	TARBALL_PATH=$(conda build . --python=$VERSION --output);
	TARBALL_NAME=$(basename $TARBALL_PATH);
	conda convert $TARBALL_PATH --platform=all;
	binstar upload win-32/$TARBALL_NAME -u soft-matter -c dev;
	binstar upload win-64/$TARBALL_NAME -u soft-matter -c dev;
	binstar upload osx-64/$TARBALL_NAME -u soft-matter -c dev;
	binstar upload linux-32/$TARBALL_NAME -u soft-matter -c dev;
	binstar upload linux-64/$TARBALL_NAME -u soft-matter -c dev;
done
