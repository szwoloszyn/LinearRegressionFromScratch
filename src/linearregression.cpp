#include <iostream>
#include <cmath>
#include "linearregression.h"
using arma::vec, arma::mat;

LinearRegression::LinearRegression() : linearParams{}
{

}


double LinearRegression::predictSingleValue(const arma::vec& X_pred, const vec& params) const
{
    auto theta = params;
    if (params.n_elem == 0) {
        theta = this->linearParams;
    }
    if (theta.size() == 0) {
        throw ModelNotFittedException{"call LinearRegression::fit() method to train model first!"};
    }
    arma::vec X_predAug = arma::join_vert(arma::vec({1}), X_pred);
    if (theta.size() != X_predAug.size()) {
        throw FeaturesDiffFromTraining{"data does not fit to last used training set!"};
    }
    double prediction = 0;
    for (size_t i = 0; i < theta.size(); ++i) {
        prediction += theta[i] * X_predAug[i];
    }
    return prediction;
}

arma::vec LinearRegression::predict(const arma::mat &X_pred, const arma::vec& params) const
{
    auto theta = params;
    if (params.n_elem == 0) {
        theta = this->linearParams;
    }
    vec predictions(X_pred.n_rows);
    for (size_t i = 0; i < X_pred.n_rows; ++i) {
        predictions(i) = predictSingleValue((X_pred.row(i)).t(), theta);
    }
    return predictions;
}
std::vector<double> LinearRegression::kFoldCrossValidation(const arma::mat& X, const arma::vec& y, const size_t k) const
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
    std::vector<double> RMSEValues;
    for (size_t i = 0; i < k; ++i) {
        auto X_train = concatFolds(X_folds,i);
        auto y_train = concatFolds(y_folds,i);
        auto X_test = X_folds[i];
        auto y_test = y_folds[i];
        vec thetas;
        try {
            thetas = getFitResults(X_train, y_train);
        }
        catch(std::invalid_argument) {
            std::cerr << "data splitted in lineary dependent"
                         "features. Skipping this split.";
                continue;
        }
        auto predictions = predict(X_test,thetas);
        RMSEValues.push_back(RMSE(y_test,predictions));
    }
    return RMSEValues;
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
    for (size_t i = 0; i < k; ++i) {
        X_folds[i] = X_shuffled.rows(foldSize*i,foldSize*(i+1) - 1);
        y_folds[i] = y_shuffled.rows(foldSize*i,foldSize*(i+1) - 1);
    }

    if (X_shuffled.n_rows > foldSize*k) {
        X_folds[X_folds.size() - 1] = arma::join_cols(X_folds[X_folds.size() - 1],
                                                      X_shuffled.rows(k*foldSize, X_shuffled.n_rows - 1));
        y_folds[y_folds.size() - 1] = arma::join_cols(y_folds[y_folds.size() - 1],
                                                      y_shuffled.rows(k*foldSize, y_shuffled.n_rows - 1));
    }
}


// template <typename T>
// T LinearRegression::concatFolds(const std::vector<T>& folds, const size_t excludeIdx) const


void LinearRegression::RMSEReport(const arma::mat &X_test, const arma::vec &y_test) const
{
    auto y_pred = predict(X_test);
    std::cout << "\nMean RMSE for whole testing set: \n";
    std::cout << RMSE(y_test,y_pred) << "\n";
}

double LinearRegression::RMSE(const arma::vec &actual, const arma::vec &predicted) const
{
    if (actual.n_elem != predicted.n_elem) {
        throw std::invalid_argument{"incompatible datasets (size differs) !"};
    }
    int n = actual.n_elem;
    auto vecDiff = actual - predicted;
    mat multiplication = vecDiff.t() * vecDiff;
    if (multiplication.n_elem != 1) {
        throw std::logic_error{"vector multiplication did not produce single value. Probably incorrect data dimensions!"};
    }
    double MSE = multiplication(0,0) / n;
    return sqrt(MSE);
}
