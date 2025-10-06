#include <iostream>
#include "linearregression.h"
using arma::vec, arma::mat;

LinearRegression::LinearRegression() : linearParams{}
{

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

void LinearRegression::printCoeffs()
{
    std::cout << "predicted coefficients: " << linearParams;
}
