#!/bin/bash

mkdir -p build
cd build
cmake ..
make

cp ../include/* /usr/local/include/

mv ./libLinRegFromScratch.so /usr/local/lib/


