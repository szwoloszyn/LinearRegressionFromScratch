#include <iostream>
#include <cstdlib>
#include <armadillo>
#include <ctime>
#include <vector>

#include "normalequation.h"
#include "batchgradientdescent.h"
using arma::vec, arma::mat;
using namespace std;



void generateTrainingData(mat& X, vec& y)
{
    // modifies X and y in place !
    // generates linear data with random gaussian noise
    // used strictly for model testing

    // y = 4x + 3 is expected parametrization
    int a = 4;
    int b = 3;
    int NUM_OF_EXAMPLES = 50;

    vector<double> featureValues;
    featureValues.reserve(NUM_OF_EXAMPLES);
    vector<double> labelValues;
    labelValues.reserve(NUM_OF_EXAMPLES);
    for (auto i = 0; i < NUM_OF_EXAMPLES; ++i) {
        double noise = rand() % 101;
        // simulating random +/- noise
        if ((int(noise) % 10) % 2) {
            noise = -noise;
        }
        noise = noise / 100;
        featureValues.push_back(i);
        labelValues.push_back(a*i + b + noise);
    }

    X = mat(featureValues.size(),1, arma::fill::zeros);
    y = vec(labelValues.size(), arma::fill::zeros);
    cout << "\n";
    for (size_t i = 0; i < featureValues.size(); ++i) {
        cout << featureValues[i] << ", ";
        X(i,0) = featureValues[i];
    }
    cout << "\n";
    for (auto i = 0uL; i < labelValues.size(); ++i) {
        cout << labelValues[i] << ", ";
        y(i) = labelValues[i];
    }

}


size_t generateComplexTrainingData(mat& XX, vec& yy)
{
    arma::arma_rng::set_seed_random();

    size_t nSamples = 100;

    arma::vec x = arma::randu<arma::vec>(nSamples) * 10;
    arma::vec y = arma::randu<arma::vec>(nSamples) * 5;

    arma::vec noise = arma::randn<arma::vec>(nSamples);

    // z = 2x + 3y + 1 + noise
    arma::vec z = 2 * x + 3 * y + 1 + noise;

    arma::mat X(nSamples, 2);
    X.col(0) = x;
    X.col(1) = y;

    XX = X;
    yy = z;
    return nSamples;
}

int main()
{
    srand(123);

    mat XX;
    vec yy;
    const size_t TSS = 0.8 * generateComplexTrainingData(XX,yy); // returning how many samples
    BatchGradientDescent grad{0.01,8000};
    NormalEquation norm;
    norm.fit(XX.rows(0,TSS),yy.rows(0,TSS));
    grad.fit(XX.rows(0,TSS),yy.rows(0,TSS));
    auto gradPreds = grad.predict(XX.rows(TSS + 1,XX.n_rows-1));
    auto normeqPreds = norm.predict(XX.rows(TSS + 1,XX.n_rows-1));
    auto actual = yy.rows(TSS + 1,yy.n_rows-1);
    std::cout << gradPreds.size() << " " << normeqPreds.size() << " " << actual.n_elem << "\n";
    std::cout << "gradient" << " ; " << "normal eq" << " ; " << "real values" << "\n";
    for (size_t i = 0; i < gradPreds.size(); ++i) {
        std::cout << gradPreds[i] << " ; " << normeqPreds[i] << " ; " << actual[i] << "\n";
    }
    cout << "\ngradient coefs:" << grad.getCoeffs();
    cout << "\nnormeq coefs:" << norm.getCoeffs();
    grad.RMSEReport(XX.rows(TSS + 1,XX.n_rows-1), yy.rows(TSS + 1, yy.n_rows - 1));
    return 0;

}
