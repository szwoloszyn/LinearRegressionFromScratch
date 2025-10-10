#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include <linearregression.h>
// NOTE need a StandardScaler
class BatchGradientDescent : public LinearRegression
{
public:
    BatchGradientDescent(double eta, size_t n = 1000);
    void fit(const arma::mat& X, const arma::vec& y) override;
    arma::vec getFitResults(const arma::mat& X, const arma::vec& y) const override;
public:
           arma::vec calculateGradient(const arma::mat& X, const arma::vec& y,
                                           const arma::vec& thetas) const;
private:
    const double learningRate;
    const size_t nEpochs;
};

#endif // GRADIENTDESCENT_H
