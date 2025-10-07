#include "normalequation.h"
#include <iostream>
#include <vector>

using arma::vec, arma::mat;

NormalEquation::NormalEquation() : LinearRegression{} { }

vec NormalEquation::solveNormalEquation(const arma::mat& X, const arma::vec& y) const
{
    mat X_aug = arma::join_horiz(arma::ones<vec>(X.n_rows), X);
    // normal equation: params = (X.t * X)^-1 * (X.t * y)
    return (X_aug.t() * X_aug).i() * (X_aug.t() * y);
}

void NormalEquation::fit(const arma::mat& X, const arma::vec& y)
{
    std::cout << "aaa";
    this->linearParams = this->solveNormalEquation(X,y);
    std::cout << "theta: " << linearParams;
}


