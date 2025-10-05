#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <armadillo>

class LinearRegression
{
public:
    enum ModelMethod {
        NORMAL_EQ,
        gradient
    };

    LinearRegression(ModelMethod m);
    arma::vec solveNormalEquation(arma::mat X, arma::vec y);
    void fit(arma::mat X, arma::vec y);
    double predict(arma::mat X);
    arma::vec crossValidation();
private:

    arma::vec linearParams;
    ModelMethod method;
};

#endif // LINEARREGRESSION_H
