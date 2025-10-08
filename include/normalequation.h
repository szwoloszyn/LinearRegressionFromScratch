#ifndef NORMALEQUATION_H
#define NORMALEQUATION_H

#include <linearregression.h>

class NormalEquation : public LinearRegression
{
public:
    NormalEquation();
    void fit(const arma::mat& X, const arma::vec& y) override;
TESTABLE:
    // NOTE probably needed for KFold ?
        // implement when needed
    arma::vec getFitResults(arma::mat X, arma::vec y) const override { return arma::vec{}; }
    arma::vec solveNormalEquation(const arma::mat& X, const arma::vec& y) const;
};

#endif // NORMALEQUATION_H
