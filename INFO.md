2 options of compiling:

# user decided whether he wants to compile as static or shared library
# default (when doing cmake .) is OFF
cmake -DBUILD_SHARED_LIBS=OFF/ON


put .so file into /usr/local/lib
put all .h files into /usr/local/include

g++ user_file.cpp -lLinRegFromScratch -larmadillo -o output_filename.outITS WORKINGGG
