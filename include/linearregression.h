#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <armadillo>

class LinearRegression
{
public:
    // enum ModelMethod {
    //     NORMAL_EQ,
    //     GRADIENT,
    //     UNIDENTIFIED
    // };

    LinearRegression();

    virtual void fit(arma::mat X, arma::vec y) = 0;
    double predict(arma::vec X_pred);
    virtual arma::vec kFoldCrossValidation() = 0;
    void printCoeffs();

    virtual ~LinearRegression() { }
protected:
    //ModelMethod method;
    arma::vec linearParams;
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
