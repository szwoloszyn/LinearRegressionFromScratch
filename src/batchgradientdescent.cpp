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
    // learning loop
    for (auto epoch = 0; epoch < nEpochs; ++epoch) {
        vec grad = calculateGradient(X_aug,y,thetas);
        bool isFinished = true;
        thetas = thetas - (learningRate * grad);

        // print every 10th epoch
        if (epoch % 10) {
            continue;
        }
        std::cout << "[";
        for (auto i = 0; i < thetas.size(); ++i) {
            std::cout << thetas(i) << ", ";
        }
        std::cout << "],\n";
    }

    //std::cout << "\nI have finished with gradient: " << lastGrad;
    return thetas;
}

arma::mat BatchGradientDescent::standardize(const arma::mat &X) const
{
    mat standardizedX;
    for (auto i = 0; i < X.n_cols; ++i) {
        double meanX = arma::mean(X.col(i));
        double stdDevX = arma::stddev(X.col(i));

        vec column(X.col(i).n_elem);
        for (auto idx = 0; idx < column.size(); ++idx) {
            double value = X.col(i)(idx);
            column(idx) = (value - meanX) / stdDevX;
        }
        standardizedX = arma::join_horiz(standardizedX, column);
    }
    return standardizedX;
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
