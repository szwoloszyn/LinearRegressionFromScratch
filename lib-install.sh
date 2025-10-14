#!/bin/bash

mkdir -p build
cd build
cmake ..
make

cp ../include/* /usr/local/include/

mv ./liblinregfromscratch.so /usr/local/lib/

ldconfig /usr/local/lib
