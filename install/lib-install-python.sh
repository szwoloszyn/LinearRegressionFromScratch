#!/bin/bash

mkdir -p build
cd build

cmake .. -DBUILD_PYTHON_LIBS=ON
make

PACKAGE_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "copying .so file to:"
echo $PACKAGE_PATH
cp ./linregpy.cpython*.so "$PACKAGE_PATH"
