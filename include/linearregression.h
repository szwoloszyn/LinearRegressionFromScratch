#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <armadillo>


#ifdef UNIT_TEST
#define TESTABLE public
#else
#define TESTABLE protected
#endif

class LinearRegression
{
public:
    LinearRegression();

    virtual void fit(const arma::mat& X, const arma::vec& y) = 0;
    virtual arma::vec getFitResults(arma::mat X, arma::vec y) const = 0;
    double predict(const arma::vec& X_pred) const;
    arma::vec kFoldCrossValidation(const arma::mat& X, const arma::vec& y,
                                   const size_t k = 5) const;
    arma::vec getCoeffs() const;

    virtual ~LinearRegression() { }
TESTABLE:
    void splitFolds(const arma::mat& X, const arma::vec& y, const size_t k,
                    std::vector<arma::mat>& X_folds, std::vector<arma::vec>& y_folds) const;

    template <typename T>
    T concatFolds(const std::vector<T>& folds, const size_t excludeIdx) const;

protected:
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
