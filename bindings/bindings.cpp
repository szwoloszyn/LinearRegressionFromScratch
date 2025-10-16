#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "normalequation.h"
#include "batchgradientdescent.h"


namespace py = pybind11;


py::array_t<double> arma_mat_to_numpy(const arma::mat& mat) {
    // Copy data because Armadillo is column-major and NumPy is row-major
    auto result = py::array_t<double>({mat.n_rows, mat.n_cols});
    auto buf = result.mutable_unchecked<2>();
    for (arma::uword i = 0; i < mat.n_rows; ++i)
        for (arma::uword j = 0; j < mat.n_cols; ++j)
            buf(i, j) = mat(i, j);
    return result;
}


py::array_t<double> arma_vec_to_numpy(const arma::vec& vec) {
    auto result = py::array_t<double>(vec.n_elem);
    auto buf = result.mutable_unchecked<1>();
    for (arma::uword i = 0; i < vec.n_elem; ++i)
        buf(i) = vec(i);
    return result;
}

arma::mat foo(arma::mat x)
{
    return x;
}

arma::mat numpy_to_arma_mat(const py::array_t<double>& array) {
    py::buffer_info buf = array.request();
    if (buf.ndim != 2)
        throw std::runtime_error("NumPy array must be 2D for arma::mat");

    arma::mat mat(buf.shape[0], buf.shape[1]);
    double* ptr = static_cast<double*>(buf.ptr);

    // Copy data row-major → column-major
    for (ssize_t i = 0; i < buf.shape[0]; ++i)
        for (ssize_t j = 0; j < buf.shape[1]; ++j)
            mat(i, j) = ptr[i * buf.shape[1] + j];

    return mat;
}

arma::vec numpy_to_arma_vec(const py::array_t<double>& array) {
    py::buffer_info buf = array.request();
    if (buf.ndim != 1)
        throw std::runtime_error("NumPy array must be 1D for arma::vec");

    arma::vec vec(buf.shape[0]);
    double* ptr = static_cast<double*>(buf.ptr);
    for (ssize_t i = 0; i < buf.shape[0]; ++i)
        vec(i) = ptr[i];

    return vec;
}

PYBIND11_MODULE(linregpy, m) {
    m.def("arma_mat_to_numpy", &arma_mat_to_numpy);
    m.def("arma_vec_to_numpy", &arma_vec_to_numpy);
    m.def("numpy_to_arma_mat", &numpy_to_arma_mat);
    m.def("numpy_to_arma_vec", &numpy_to_arma_vec);
    m.def("foo", &foo);
}

// PYBIND11_MODULE(linregpy, m) {
//     m.doc() = "Python bindings for the C++ LinearRegression library";
//     py::class_<LinearRegression>(m, "LinearRegression")
//         .def("fit", &LinearRegression::fit, "Train the model",
//              py::arg("X_train"), py::arg("y_train"))
//         .def("predict", &LinearRegression::predict, "Make predictions",
//              py::arg("X_test"));

//     py::class_<NormalEquation, LinearRegression, std::shared_ptr<NormalEquation>>(m, "NormalEquation")
//         .def(py::init<>())
//         .def("fit", &NormalEquation::fit, "Train the model",
//             py::arg("X_train"), py::arg("y_train"))
//         .def("predict", &NormalEquation::predict, "Make predictions",
//              py::arg("X_test"));
// }




// PYBIND11_MODULE(linregpy, m) {
//     py::class_<LinearRegression, std::shared_ptr<LinearRegression>>(m, "LinearRegression")
//         .def("fit", &LinearRegression::fit)
//         .def("predict", &LinearRegression::predict);


//     py::class_<NormalEquation, LinearRegression, std::shared_ptr<NormalEquation>>(m, "NormalEquation")
//         .def(py::init<>());

// }
