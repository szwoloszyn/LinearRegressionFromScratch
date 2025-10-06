#ifndef NORMALEQUATION_H
#define NORMALEQUATION_H

#include <linearregression.h>

class NormalEquation : public LinearRegression
{
public:
    NormalEquation();
    void fit(arma::mat X, arma::vec y) override;
    arma::vec solveNormalEquation(arma::mat X, arma::vec y);
    arma::vec kFoldCrossValidation() override;
};

#endif // NORMALEQUATION_H
