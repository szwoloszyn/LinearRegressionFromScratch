#!/bin/bash

mkdir -p build
cd build

if [[ $(pwd) != *"LinearRegressionFromScratch/build" ]]; then
	echo "ERROR! Script probably was not run from root project directory"
	exit 1
fi

cmake ..
make

cp ../include/* /usr/local/include/
mv ./liblinregfromscratch.so /usr/local/lib/

/usr/sbin/ldconfig /usr/local/lib


if [[ $(pwd) != *"LinearRegressionFromScratch/build" ]]; then
	echo "ERROR! Script probably was not run from root project directory"
	exit 1
fi

# optional: grant normal user privilages to manage build/ dir
chmod -R o+w .
exit 0
