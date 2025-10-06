#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <armadillo>

// TODO parameters reference/copy cleanup
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
    virtual arma::vec getFitResults(arma::mat X, arma::vec y) = 0;
    double predict(arma::vec X_pred);
    arma::vec kFoldCrossValidation(arma::mat X, arma::vec y, size_t folds = 5);
    void printCoeffs();

    virtual ~LinearRegression() { }
protected:
    void splitFolds(arma::mat X, arma::vec y, size_t k, std::vector<arma::mat>& X_folds, std::vector<arma::vec>& y_folds);
    arma::mat concatExcept(const std::vector<arma::mat> folds, size_t excludeIdx);
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
