#include <iostream>
#include "normalequation.h"

int main()
{
	NormalEquation a;
	a.fit(arma::mat({1,2,3}).t(), arma::vec({2,4,6}));
	double a = 4;
	std::cout << "Predicted value for " << a << " is " << a.predict(arma::mat({a})) << "\n";
	return 0;
}
