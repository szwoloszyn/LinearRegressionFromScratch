#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include <linearregression.h>
// NOTE need a StandardScaler
class BatchGradientDescent : public LinearRegression
{
public:
    BatchGradientDescent(double eta, size_t n = 1000);

private:
    double learningRate;
    size_t nEpochs;
};

#endif // GRADIENTDESCENT_H
