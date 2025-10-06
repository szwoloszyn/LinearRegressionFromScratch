#include <iostream>
#include "linearregression.h"
using arma::vec, arma::mat;

LinearRegression::LinearRegression(ModelMethod m) : linearParams{}, method{m}
{

}

void LinearRegression::fit(arma::mat X, arma::vec y)
{
    linearParams = solveNormalEquation(X,y);
}


vec LinearRegression::solveNormalEquation(mat X, vec y)
{
    mat X_aug = arma::join_horiz(arma::ones<vec>(X.n_rows), X);
    //std::cout << X_aug;
    // normal equation: params = (X.t * X)^-1 * (X.t * y)
    return (X_aug.t() * X_aug).i() * (X_aug.t() * y);
}

double LinearRegression::predict(arma::vec X_pred)
{
    // when linearParams.size() == 0 -> throw model not fitted exception
    if (linearParams.size() == 0) {
        throw ModelNotFittedException{"call LinearRegression::fit() method to train model first!"};
    }
    X_pred = arma::join_vert(arma::vec({1}), X_pred);
    if (linearParams.size() != X_pred.size()) {
        throw FeaturesDiffFromTraining{"data does not fit to last used training set!"};
    }
    double prediction = 0;
    for (auto i = 0; i < linearParams.size(); ++i) {
        prediction += linearParams[i] * X_pred[i];
    }
    return prediction;
}
