#include <iostream>
#include <cstdlib>
#include <armadillo>
#include <ctime>
#include <vector>

#include "normalequation.h"
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
        double noise = rand() % 901;
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
    for (auto i = 0; i < featureValues.size(); ++i) {
        cout << featureValues[i] << ", ";
        X(i,0) = featureValues[i];
    }
    cout << "\n";
    for (auto i = 0; i < labelValues.size(); ++i) {
        cout << labelValues[i] << ", ";
        y(i) = labelValues[i];
    }

}


int main()
{
    srand(time(0));
    // cout << "linear regression" << endl;
    // arma::mat matrix = arma::mat("0.0 0.1 0.2 ; 1.0 1.1 1.2 ; 2.0 2.1 2.2");

    // matrix(1,1) = 0.0123956;
    // matrix.print(std::cout, "org");

    // matrix = matrix.t();
    // matrix.print(std::cout, "transposed");
    // try {
    //     matrix = matrix.i();
    // }
    // catch(...) {
    //     cout << "matrix does not have inversion";
    // }

    // matrix.print(std::cout, "inversedxx");

    NormalEquation testVec{};
    mat X{ 1,2};
    X = X.t();
    vec y = {2,4};
    testVec.fit(X,y);
    cout << testVec.predict(vec{4});

    generateTrainingData(X,y);
    //cout << X;
    testVec.fit(X,y);
    testVec.printCoeffs();

    mat foldX = {
        {1,1},
        {2,2},
        {3,3},
        {4,4},
        {5,5},
        {6,6},
    };
    vec foldy = {
        1,2,3,4,5,6
    };
    testVec.kFoldCrossValidation(foldX, foldy,1);

}
