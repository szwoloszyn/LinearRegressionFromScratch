#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include <linearregression.h>
class BatchGradientDescent : public LinearRegression
{
public:
    BatchGradientDescent(double eta, size_t n = 1000);
    void fit(const arma::mat& X, const arma::vec& y) override;
    arma::vec getFitResults(const arma::mat& X, const arma::vec& y) const override;
TESTABLE:
    arma::vec getMeans(const arma::mat& X) const;
    arma::vec getStdDevs(const arma::mat& X) const;
    arma::mat standardize(const arma::mat& X) const;
    arma::vec calculateGradient(const arma::mat& X, const arma::vec& y,
                                           const arma::vec& thetas) const;
arma::vec rescaleTheta(const arma::vec& means, const arma::vec& stddevs,
                           const arma::vec& theta) const;
private:
    const double learningRate;
    const size_t nEpochs;
};

#endif // GRADIENTDESCENT_H
