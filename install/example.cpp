#include <iostream>
#include "normalequation.h"

int main()
{
	NormalEquation a;
	a.fit(arma::mat({1,2,3}).t(), arma::vec({2,4,6}));
	std::cout << a.predict(arma::mat({812}));
	return 0;
}
