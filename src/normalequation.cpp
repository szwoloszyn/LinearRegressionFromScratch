#include "normalequation.h"
#include <iostream>
#include <vector>

using arma::vec, arma::mat;

NormalEquation::NormalEquation() : LinearRegression{} { }

vec NormalEquation::solveNormalEquation(mat X, vec y)
{
    mat X_aug = arma::join_horiz(arma::ones<vec>(X.n_rows), X);
    // normal equation: params = (X.t * X)^-1 * (X.t * y)
    return (X_aug.t() * X_aug).i() * (X_aug.t() * y);
}

void NormalEquation::fit(arma::mat X, arma::vec y)
{
    std::cout << "aaa";
    this->linearParams = this->solveNormalEquation(X,y);
    std::cout << "theta: " << linearParams;
}

arma::vec NormalEquation::kFoldCrossValidation(arma::mat X, arma::vec y, size_t k)
{
    arma::uvec indices = arma::randperm(X.n_rows);
    arma::mat X_shuffled = X.rows(indices);
    arma::vec y_shuffled = y.elem(indices);

    size_t foldSize = X_shuffled.n_rows / k;
    std::vector<mat> X_folds{k};
    std::vector<vec> y_folds{k};
    for (auto i = 0; i < k; ++i) {
        X_folds[i] = X_shuffled.rows(foldSize*i,foldSize*(i+1) - 1);
        // WARNING IT MIGHT BE WRONG !!
        y_folds[i] = y_shuffled.rows(foldSize*i,foldSize*(i+1) - 1);
    }
    // TODO LEFTOVERS FROM FOLDSIZE
    return vec{};
}
