#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <armadillo>

class LinearRegression
{
public:
    enum ModelMethod {
        NORMAL_EQ,
        GRADIENT,
        UNIDENTIFIED
    };

    LinearRegression(ModelMethod m = ModelMethod::UNIDENTIFIED);
    arma::vec solveNormalEquation(arma::mat X, arma::vec y);
    void fit(arma::mat X, arma::vec y);
    double predict(arma::vec X_pred);
    arma::vec crossValidation();
    void printCoeffs();
private:

    arma::vec linearParams;
    ModelMethod method;
};

// custom exception for unfitted model
class ModelNotFittedException : public std::runtime_error
{
public:
    explicit ModelNotFittedException(const std::string& msg)
        : std::runtime_error(msg) {}
};

class FeaturesDiffFromTraining : public std::runtime_error
{
public:
    explicit FeaturesDiffFromTraining(const std::string& msg)
        : std::runtime_error(msg) {}
};

#endif // LINEARREGRESSION_H
