#include "batchgradientdescent.h"

using arma::vec, arma::mat;

BatchGradientDescent::BatchGradientDescent(double eta, size_t n) :
    learningRate{eta}, nEpochs{n}
{

}

void BatchGradientDescent::fit(const arma::mat &X, const arma::vec &y)
{
    this->linearParams = getFitResults(X,y);
}

arma::vec BatchGradientDescent::getFitResults(const arma::mat &X, const arma::vec &y) const
{
    const mat X_aug = arma::join_horiz(arma::ones<vec>(X.n_rows), X);
    // TODO start with what thetas ?
    vec thetas(X_aug.n_cols);
    thetas.fill(0.1);
    vec lastGrad;
    // learning loop
    for (auto epoch = 0; epoch < nEpochs; ++epoch) {
        vec grad = calculateGradient(X_aug,y,thetas);
        bool isFinished = true;
        // for (auto i = 0; i < grad.n_elem; ++i) {
        //     if (grad(i) != 0) {
        //         isFinished = false;
        //     }
        // }
        // if (isFinished) {
        //     return thetas;
        // }
        thetas = thetas - (learningRate * grad);
        lastGrad = grad;
    }
    std::cout << "\nI have finished with gradient: " << lastGrad;
    return thetas;
}

arma::vec BatchGradientDescent::calculateGradient(const arma::mat &X, const arma::vec &y,
                                                  const arma::vec &thetas) const
{
    if (y.n_elem > X.n_rows) {
        throw LabelsDontFitLearningData{"there are less learning examples than labels!"};
    }
    if (y.n_elem < X.n_rows) {
        throw LabelsDontFitLearningData{"there are more learning examples than labels!"};
    }
    if (thetas.n_elem != X.n_cols) {
        throw std::invalid_argument{"there are more/less thetas than labels. Critical error."};
    }
    double param = 2 / double(y.n_elem);
    vec gradient = param * X.t() * (X*thetas - y);
    return gradient;
}
