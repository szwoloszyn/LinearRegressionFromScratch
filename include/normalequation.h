#ifndef NORMALEQUATION_H
#define NORMALEQUATION_H

#include <linearregression.h>

class NormalEquation : public LinearRegression
{
public:
    NormalEquation();
    void fit(const arma::mat& X, const arma::vec& y) override;
TESTABLE:
    arma::vec getFitResults(const arma::mat& X, const arma::vec& y) const override;
    arma::vec solveNormalEquation(const arma::mat& X, const arma::vec& y) const;
};

#endif // NORMALEQUATION_H
