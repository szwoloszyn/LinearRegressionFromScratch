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
    vec means = getMeans(X);
    vec stddevs = getStdDevs(X);

    mat stdX = standardize(X);

    mat X_aug = arma::join_horiz(arma::ones<vec>(stdX.n_rows), stdX);
    // TODO start with what thetas ?
    vec theta(X_aug.n_cols);
    theta.fill(0.1);

    for (size_t epoch = 0; epoch < nEpochs; ++epoch) {
        vec grad = calculateGradient(X_aug,y,theta);
        theta = theta - (learningRate * grad);
    }
    return rescaleTheta(means, stddevs, theta);
}

arma::vec BatchGradientDescent::getMeans(const arma::mat &X) const
{
    vec means(X.n_cols);
    for (size_t i = 0; i < X.n_cols; ++i) {
        means(i) = arma::mean(X.col(i));
    }
    return means;
}

arma::vec BatchGradientDescent::getStdDevs(const arma::mat &X) const
{
    vec devs(X.n_cols);
    for (size_t i = 0; i < X.n_cols; ++i) {
        devs(i) = arma::stddev(X.col(i));
    }
    return devs;
}

arma::mat BatchGradientDescent::standardize(const arma::mat &X) const
{
    mat standardizedX;
    for (size_t i = 0; i < X.n_cols; ++i) {
        double meanX = arma::mean(X.col(i));
        double stdDevX = arma::stddev(X.col(i));
        if (stdDevX == 0) {
            stdDevX = 1;
        }
        vec column(X.col(i).n_elem);
        for (size_t idx = 0; idx < column.size(); ++idx) {
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

    double param = 2.0 / double(y.n_elem);
    vec gradient = param * (X.t() * (X*thetas - y));
    return gradient;
}

arma::vec BatchGradientDescent::rescaleTheta(const arma::vec &means, const arma::vec &stddevs,
                                             const arma::vec &theta) const
{
    vec rescaledTheta(theta.n_elem);
    for (size_t i = 1; i < theta.n_elem; ++i) {
        rescaledTheta(i) = theta(i) / stddevs(i-1);
    }
    double sum = 0;
    for (size_t i = 1; i < rescaledTheta.n_elem; ++i) {
        sum += rescaledTheta(i) * means(i-1);
    }

    rescaledTheta(0) = theta(0) - sum;
    return rescaledTheta;
}
