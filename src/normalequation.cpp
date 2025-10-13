#include "normalequation.h"


using arma::vec, arma::mat;

NormalEquation::NormalEquation() : LinearRegression{} { }

vec NormalEquation::solveNormalEquation(const arma::mat& X, const arma::vec& y) const
{
    mat X_aug = arma::join_horiz(arma::ones<vec>(X.n_rows), X);
    // normal equation: Theta = (X.t * X)^-1 * (X.t * y)
    auto multipliedX = X_aug.t() * X_aug;
    if (det(multipliedX) == 0) {
        // TODO to be updated. Also when to little examples comparing to features, it also gets here! (should be other exception
        throw std::invalid_argument{"there is the same feature twice in dataset! "
                                    "Will cover this case in further development. "
                                    "For now you need to manually delete one of these features. "
                                    "(There is no learning gain from it)!"};
    }
    return multipliedX.i() * (X_aug.t() * y);
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

