#ifndef NORMALEQUATION_H
#define NORMALEQUATION_H

#include <linearregression.h>

// TODO change public in the end of testing
class NormalEquation : public LinearRegression
{
public:
    NormalEquation();
    void fit(const arma::mat& X, const arma::vec& y) override;
    arma::vec getFitResults(arma::mat X, arma::vec y) const override { return arma::vec{}; }
    arma::vec solveNormalEquation(const arma::mat& X, const arma::vec& y) const;
    //arma::vec kFoldCrossValidation(arma::mat X, arma::vec y, size_t folds = 5) override;
};

#endif // NORMALEQUATION_H
