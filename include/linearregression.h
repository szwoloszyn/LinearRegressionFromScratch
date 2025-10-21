#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <armadillo>


#ifdef UNIT_TEST
#define TESTABLE public
#else
#define TESTABLE protected
#endif
// TODO 1. update instalation script and readme 2. start pull requesting

class LinearRegression
{
public:
    LinearRegression();

    virtual void fit(const arma::mat& X, const arma::vec& y) = 0;

    double predictSingleValue(const arma::vec& X_pred, const arma::vec& params = arma::vec{}) const;
    arma::vec predict(const arma::mat& X_pred, const arma::vec& params = arma::vec{}) const;
    std::vector<double> kFoldCrossValidation(const arma::mat& X, const arma::vec& y,
                                   const size_t k = 5) const;
    arma::vec getCoeffs() const;
    double getRMSE(const arma::mat& X_test, const arma::vec& y_test) const;
    void RMSEReport(const arma::mat& X_test, const arma::vec& y_test) const;

    virtual ~LinearRegression() { }
TESTABLE:
    virtual arma::vec getFitResults(const arma::mat& X, const arma::vec& y) const = 0;
    void splitFolds(const arma::mat& X, const arma::vec& y, const size_t k,
                    std::vector<arma::mat>& X_folds, std::vector<arma::vec>& y_folds) const;

    template <typename T>
    T concatFolds(const std::vector<T>& folds, const size_t excludeIdx) const
    {
        if (excludeIdx >= folds.size()) {
            throw std::invalid_argument{"index greater than array size"};
        }
        arma::mat finalMat;
        for (size_t i = 0; i < folds.size(); ++i) {
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

    double RMSE(const arma::vec& actual, const arma::vec& predicted) const;


protected:
    arma::vec linearParams;
};

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

class LabelsDontFitLearningData : public std::runtime_error
{
public:
    explicit LabelsDontFitLearningData(const std::string& msg)
        : std::runtime_error(msg) {}
};

#endif // LINEARREGRESSION_H
