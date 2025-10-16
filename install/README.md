### Linear Regression From Scratch - Instalation Guide
[NOTE]: This guide is suited for Linux users, if you want to install this library on windows machine, you **will not** be able to run [my install script](./lib-install.sh), instead build manually.


In order to build this project you need:
- **C++ 17** or newer
- **CMake >= 3.10**
- **Armadillo Library**
- **pybind11** optionally if willing to use it with python

All functionalities the library has to offer are stated in main [README.md](../README.md) file.

---

#### C++ INSTALLATION

You can build either **shared** or **static** version
```
# for shared (.so) lib
cmake -DBUILD_SHARED_LIBS=ON /root/dir/to/project

# for static (.a) lib
cmake -DBUILD_SHARED_LIBS=OFF /root/dir/to/project
```

To install this library system-wide (Linux only) you can use [this bash script](./lib-install.sh) which will build the library and put all needed  files in `/usr/local/[...]` directory.

- Run [lib-install.sh](./lib-install.sh) script from **project root** directory `(path/on/your/pc/LinearRegressionFromScratch)`

- You need to run [lib-install.sh](./lib-install.sh) as `ROOT` so it may save `.so` and `.h` files into correct directiories.


In [example.cpp](./example.cpp) there is already pre-trained basic model. In order to compile the file into executable, follow:
```
g++ -O3 example.cpp -llinregfromscratch -o example.out
```
##### INCLUDES:
In your `.cpp` file you can use following includes:
- `#include "normalequation.h"` ( Linear Regression using normal equation for calculations (accurate but **very** slow when number of features raises) ) - move this to global readme !
- `#include "batchgradientdescent.h"`

Details are described in [main README.md](../README.md) file.

---

#### PYTHON INSTALLATION

`cmake -DBUILD_PYTHON_LIBS=ON` will generate a `linregpy*.so` file. You can run python files from the same directory as `.so` file or move into python packages directory to run files globally.
You probably can check it with `echo $(python3 -c "import site; print(site.getsitepackages()[0])")`

Python installation [script](./lib-install-python.sh) basically follow aforementioned steps.

##### INCLUDES
- `import linregpy` will allow you to run every functionality delivered by this library
