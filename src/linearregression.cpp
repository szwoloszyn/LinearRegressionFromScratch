#include <iostream>
#include "linearregression.h"
using arma::vec, arma::mat;

LinearRegression::LinearRegression() : linearParams{}
{

}






double LinearRegression::predict(const arma::vec& X_pred) const
{
    // when linearParams.size() == 0 -> throw model not fitted exception
    if (linearParams.size() == 0) {
        throw ModelNotFittedException{"call LinearRegression::fit() method to train model first!"};
    }
    arma::vec X_predAug = arma::join_vert(arma::vec({1}), X_pred);
    if (linearParams.size() != X_predAug.size()) {
        throw FeaturesDiffFromTraining{"data does not fit to last used training set!"};
    }
    double prediction = 0;
    for (auto i = 0; i < linearParams.size(); ++i) {
        prediction += linearParams[i] * X_predAug[i];
    }
    return prediction;
}

arma::vec LinearRegression::kFoldCrossValidation(const arma::mat& X, const arma::vec& y, const size_t k) const
{
    if (k > X.n_rows) {
        throw std::invalid_argument{"desired number of "
                                    "folds is greater than number of learning data!"};
    }
    if (k <= 1) {
        throw std::invalid_argument{"number of folds needs to be at least 2!"};
    }
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
    for (auto i = 0; i < k; ++i) {
        auto X_train = concatFolds(X_folds,i);
        auto y_train = concatFolds(y_folds,i);
        auto X_test = X_folds[i];
        auto y_test = y_folds[i];
    }
    // TODO I will come back here when I will have loss function ready to go

    return vec{};
}

arma::vec LinearRegression::getCoeffs() const
{
    return linearParams;
}

void LinearRegression::splitFolds(const arma::mat& X, const arma::vec& y, const size_t k,
                                  std::vector<arma::mat> &X_folds, std::vector<arma::vec> &y_folds) const
{
    if (X_folds.size() != k or y_folds.size() != k) {
        throw std::invalid_argument{"vectors must be size-fitted to number of folds!"};
    }

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

template <typename T>
T LinearRegression::concatFolds(const std::vector<T>& folds, const size_t excludeIdx) const
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
