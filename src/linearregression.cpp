#include "linearregression.h"
using arma::vec, arma::mat;

LinearRegression::LinearRegression(ModelMethod m) : linearParams{}, method{m}
{

}


vec LinearRegression::solveNormalEquation(mat X, vec y)
{
    // normal equation: params = (X.t * X)^-1 * (X.t * y)
    return (X.t() * X).i() * (X.t() * y);
}

double LinearRegression::predict(arma::mat X)
{
    // when linearParams.size() == 0 -> throw model not fitted exception
    return 0;
}
