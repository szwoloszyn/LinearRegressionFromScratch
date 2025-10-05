#include "linearregression.h"
using arma::vec, arma::mat;

LinearRegression::LinearRegression() {}

vec LinearRegression::solveNormalEquation(mat X, vec y)
{
    // normal equation: params = (X.t * X)^-1 * (X.t * y)
    return (X.t() * X).i() * (X.t() * y);
}
