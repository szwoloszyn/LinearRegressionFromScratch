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


int main()
{
    cout << "f:" << 8./7.;
    //srand(time(0));
    srand(123);
//     // cout << "linear regression" << endl;
//     // arma::mat matrix = arma::mat("0.0 0.1 0.2 ; 1.0 1.1 1.2 ; 2.0 2.1 2.2");

//     // matrix(1,1) = 0.0123956;
//     // matrix.print(std::cout, "org");
// // noise / 100;
//     // matrix = matrix.t();
//     // matrix.print(std::cout, "transposed");
//     // try {
//     //     matrix = matrix.i();
//     // }
//     // catch(...) {
//     //     cout << "matrix does not have inversion";
//     // }

//     // matrix.print(std::cout, "inversedxx");

//     NormalEquation testVec{};
//     mat X{ 1,2};
//     X = X.t();
//     vec y = {2,3};
//     testVec.fit(X,y);
//     cout << testVec.predictSingleValue(vec{4});

//     generateTrainingData(X,y);
//     //cout << X;
//     testVec.fit(X,y);
//     testVec.RMSEReport(X.rows(1,10),y.subvec(1,10));
//     //AtestVec.printCoeffs();

//     mat foldX = {
//         {1,1},
//         {2,2},
//         {3,3},
//         {4,4},
//         {5,5.1},
//         {6,6},
//     };
//     //foldX = (mat{1,2,3,4,5,6}).t();
//     vec foldy = {
//         1,2,3,4,5,6
//     };
//     testVec.fit(foldX,foldy);
//     //cout << "\n-> " << testVec.predict(foldX) << "\n";

//     auto vect = testVec.kFoldCrossValidation(X,y,5);

//     cout << "\nKFold Cross Validation with 5 folds RMSEs: \n";
//     for (auto x : vect) {
//         std::cout << x << ", ";
//     }

//     // mat XCVExample=( mat{0,1,2,3,4,5,6,7,8,9}).t();
//     // vec yCVExample = {0,1,2,3,4,5,6,7,8,9};
//     // std::vector<arma::mat> X_folds;
//     // std::vector<arma::vec> y_folds;
//     // newModel.splitFolds(XCVExample, yCVExample, 4, X_folds, y_folds);
//     // cout << X_folds[1];

//     mat XTwoFeature = (mat{ {1,5,6}, {6,9,10} }).t();
//     vec yTwoFeature = {-10,8,20};

//     mat Xtest_a = { {1,6}, {5,9}};
//     //cout << "\ntt: \n" << Xtest_a;

    // NormalEquation testVec{};
    // mat X;
    // vec y;
    // //cout << y;
    // BatchGradientDescent grd{0.2,20};
    // generateTrainingData(X,y);

    // mat Xgrd = { {1,2}, {3,4}, {6,9}};
    // //cout << "rows: " << Xgrd.n_rows;
    // vec ygrd = {2,3,5};
    // vec tgrd = {1,1,1};
    // //cout << "\n" << grd.calculateGradient(Xgrd,ygrd,tgrd);
    // cout << "\nX:" << X;
    // cout << "\ny:" << y;
    // cout << "\n\n";
    // auto thetas = grd.getFitResults(X,y);

    // testVec.fit(Xgrd,ygrd);
    // cout << "\nNormthetas: \n" << testVec.getCoeffs();
    // cout << "\nthetas: \n" << thetas;

    size_t EXAMPLES = 50;
    size_t FEATURES = 10;
    mat X(EXAMPLES,FEATURES);
    for (size_t col = 0; col < X.n_cols; ++col) {
        for (size_t row = 0; row < X.n_rows; ++row) {
            X(row, col) = (row)*(col+1);
            if (col == row) {
                X(row,col) += 1;
            }
        }
    }
    vec y(EXAMPLES);
    for (size_t idx = 0; idx < y.n_elem; ++idx) {
        y(idx) = arma::sum(X.row(idx));
    }
    NormalEquation comparator;
    BatchGradientDescent testedModel{0.1018,10000};
    //std::cout << X << "\ny: " << y;
    auto gradient = testedModel.getFitResults(X,y);
    comparator.fit(X,y);
    auto normeq = comparator.getCoeffs();
    cout << "gradient: " << gradient;
    cout << "\nnormeq: " << normeq;
}
