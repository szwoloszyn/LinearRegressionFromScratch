#include <iostream>
#include <armadillo>

#include "linearregression.h"
using arma::vec, arma::mat;
using namespace std;

int main()
{
    // cout << "linear regression" << endl;
    // arma::mat matrix = arma::mat("0.0 0.1 0.2 ; 1.0 1.1 1.2 ; 2.0 2.1 2.2");

    // matrix(1,1) = 0.0123956;
    // matrix.print(std::cout, "org");

    // matrix = matrix.t();
    // matrix.print(std::cout, "transposed");
    // try {
    //     matrix = matrix.i();
    // }
    // catch(...) {
    //     cout << "matrix does not have inversion";
    // }

    // matrix.print(std::cout, "inversedxx");

    LinearRegression testVec{};
    mat X = { {1, 1},
             {1, 2} };
    vec y = {2,4};
    cout << testVec.solveNormalEquation(X,y);
}
