#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <armadillo>

class LinearRegression
{
public:
    LinearRegression();
    arma::vec solveNormalEquation(arma::mat X, arma::vec y);
};

#endif // LINEARREGRESSION_H
