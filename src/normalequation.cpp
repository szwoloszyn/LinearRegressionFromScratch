#include "normalequation.h"


using arma::vec, arma::mat;

NormalEquation::NormalEquation() : LinearRegression{} { }

vec NormalEquation::solveNormalEquation(const arma::mat& X, const arma::vec& y) const
{
    // normal equation: Theta = (X.t * X)^-1 * (X.t * y)
    // changed approach - moore-penrose pseudo-inverse matrix
    mat X_aug = arma::join_horiz(arma::ones<vec>(X.n_rows), X);
    return arma::pinv(X_aug) * y;
}

void NormalEquation::fit(const arma::mat& X, const arma::vec& y)
{
    if (y.n_elem > X.n_rows) {
        throw LabelsDontFitLearningData{"there are less learning examples than labels!"};
    }
    if (y.n_elem < X.n_rows) {
        throw LabelsDontFitLearningData{"there are more learning examples than labels!"};
    }
    this->linearParams = this->solveNormalEquation(X,y);
}

arma::vec NormalEquation::getFitResults(const arma::mat& X, const arma::vec& y) const
{
    return solveNormalEquation(X,y);
}
