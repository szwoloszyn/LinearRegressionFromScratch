#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include <linearregression.h>
// NOTE need a StandardScaler
class GradientDescent : public LinearRegression
{
public:
    GradientDescent(double learningRate);
};

#endif // GRADIENTDESCENT_H
