#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "normalequation.h"
#include "batchgradientdescent.h"


namespace py = pybind11;

PYBIND11_MODULE(linregpy, m) {
    m.doc() = "Python bindings for the C++ LinearRegression library";
    py::class_<LinearRegression>(m, "LinearRegression")
        .def("fit", &LinearRegression::fit, "Train the model",
             py::arg("X_train"), py::arg("y_train"))
        .def("predict", &LinearRegression::predict, "Make predictions",
             py::arg("X_test"));

    py::class_<NormalEquation, LinearRegression, std::shared_ptr<NormalEquation>>(m, "NormalEquation")
        .def(py::init<>())
        .def("fit", &NormalEquation::fit, "Train the model",
            py::arg("X_train"), py::arg("y_train"))
        .def("predict", &NormalEquation::predict, "Make predictions",
             py::arg("X_test"));
}
