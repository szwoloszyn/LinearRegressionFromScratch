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

arma::vec LinearRegression::kFoldCrossValidation(arma::mat X, arma::vec y, size_t k)
{
    std::vector<mat> X_folds{k};
    std::vector<vec> y_folds{k};
    this->splitFolds(X,y,k,X_folds, y_folds);

    std::cout << "X FOLDS: \n";
    for (auto i = 0; i < k; ++i) {
        std::cout << i+1 << ". fold: " << X_folds[i] << "\n";
    }
    std::cout << "\nY FOLDS: \n";
    for (auto i = 0; i < k; ++i) {
        std::cout << i+1 << ". fold: " << y_folds[i] << "\n";
    }
    std::cout << concatExcept(y_folds,1) << "!";

    for (auto i = 0; i < k; ++i) {
        auto X_testFold = X_folds[i];
        auto y_testFold = y_folds[i];
    }

    return vec{};
}

void LinearRegression::printCoeffs()
{
    std::cout << "predicted coefficients: " << linearParams;
}

void LinearRegression::splitFolds(arma::mat X, arma::vec y, size_t k, std::vector<arma::mat> &X_folds, std::vector<arma::vec> &y_folds)
{
    arma::uvec indices = arma::randperm(X.n_rows);
    arma::mat X_shuffled = X.rows(indices);
    arma::vec y_shuffled = y.elem(indices);

    size_t foldSize = X_shuffled.n_rows / k;
    for (auto i = 0; i < k; ++i) {
        X_folds[i] = X_shuffled.rows(foldSize*i,foldSize*(i+1) - 1);
        // WARNING IT MIGHT BE WRONG !!
        y_folds[i] = y_shuffled.rows(foldSize*i,foldSize*(i+1) - 1);
    }

    if (X_shuffled.n_rows > foldSize*k) {
        std::cout << k*foldSize << ", " << X_shuffled.n_rows - 1 << '\n';
        //std::cout << X_shuffled.rows(k*foldSize, X_shuffled.size() - 1);
        //std::cout << "#" << arma::join_cols(X_folds[X_folds.size() - 1], X_shuffled.rows(k*foldSize, X_shuffled.size() - 1)) << "#";
        X_folds[X_folds.size() - 1] = arma::join_cols(X_folds[X_folds.size() - 1],
                                                      X_shuffled.rows(k*foldSize, X_shuffled.n_rows - 1));
        y_folds[y_folds.size() - 1] = arma::join_cols(y_folds[y_folds.size() - 1],
                                                      y_shuffled.rows(k*foldSize, y_shuffled.n_rows - 1));
    }
}

arma::mat LinearRegression::concatExcept(const std::vector<arma::mat> folds, size_t excludeIdx)
{
    if (excludeIdx >= folds.size()) {
        throw std::invalid_argument{"index greater than array size"};
    }
    arma::mat finalMat;
    for (auto i = 0; i < folds.size(); ++i) {
        if (i == excludeIdx) {
            continue;
        }
        if (finalMat.n_rows == 0) {
            finalMat = folds[i];
        }
        else {
            finalMat = arma::join_cols(finalMat, folds[i]);
        }
    }
    return finalMat;
}
