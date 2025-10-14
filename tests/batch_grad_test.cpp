#include <gtest/gtest.h>
#include <armadillo>
#include <cmath>
#include "batchgradientdescent.h"
#include "normalequation.h"

using arma::mat, arma::vec;

class BatchGradTest : public ::testing::Test
{
protected:
    BatchGradientDescent gradModel{1,1};
    mat X1;
    vec y1;
    vec thetas1;

    mat XX;
    void SetUp() override
    {
        X1 = {
            {1,1,3},
            {1,2,4}
        };
        y1 = {2,3};
        thetas1 = {1,1,1};

        XX = {
            {1,1},
            {2,3},
            {3,14}
        };
    }
};

TEST_F(BatchGradTest, calcGradWorks)
{
    vec gradient = gradModel.calculateGradient(X1,y1,thetas1);
    vec expectedGradient{7,11,25};
    ASSERT_EQ(gradient.n_elem, expectedGradient.n_elem);
    for (size_t i = 0; i < gradient.n_elem; ++i) {
        ASSERT_EQ(gradient(i), expectedGradient(i));
    }
}

TEST_F(BatchGradTest, standardizeWorks)
{

    mat expectedOutput = {
        {-1,(-5./7.)},
        {0,(-3./7.)},
        {1,8./7.}
    };
    ASSERT_TRUE(arma::approx_equal(gradModel.standardize(XX),
                                   expectedOutput, "absdiff", 0.001));
}

TEST_F(BatchGradTest, getMeansWorks)
{
    vec expectedOutput = {2,6};
    ASSERT_TRUE(arma::approx_equal(gradModel.getMeans(XX),
                                   expectedOutput, "absdiff", 0.001));
}

TEST_F(BatchGradTest, getStdDevWorks)
{
    vec expectedOutput = {1,7};
    ASSERT_TRUE(arma::approx_equal(gradModel.getStdDevs(XX),
                                   expectedOutput, "absdiff", 0.001));
}

// data generator

void generateComplexTrainingData(mat& XX, vec& yy)
{
    arma::arma_rng::set_seed_random();

    size_t nSamples = 1000;

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
}


TEST_F(BatchGradTest, getFitResultsWorks)
{
    size_t EXAMPLES = 10;
    size_t FEATURES = 3;
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
    BatchGradientDescent testedModel{0.3,3000};
    ASSERT_TRUE(
        arma::approx_equal(testedModel.getFitResults(X,y),
                           comparator.getFitResults(X,y),
                           "absdiff", 0.1)
    );
}

TEST_F(BatchGradTest, modelComplexExample)
{
    mat XX;
    vec yy;
    generateComplexTrainingData(XX,yy);
    size_t TSS = 800;
    BatchGradientDescent grad{0.01,8000};
    NormalEquation normeq;
    normeq.fit(XX.rows(0,TSS),yy.rows(0,TSS));
    grad.fit(XX.rows(0,TSS),yy.rows(0,TSS));
    auto gradPreds = grad.predict(XX.rows(TSS + 1,XX.n_rows-1));
    auto normeqPreds = normeq.predict(XX.rows(TSS + 1,XX.n_rows-1));
    ASSERT_EQ(gradPreds.size(), normeqPreds.size());
    auto actual = yy.rows(TSS + 1,yy.n_rows-1);
    for (size_t i = 0; i < gradPreds.size(); ++i) {
        ASSERT_NEAR(
            gradPreds[i],
            normeqPreds[i],
            0.1
            );
        ASSERT_NEAR(
            gradPreds[i],
            actual[i],
            5
            );
    }
    ASSERT_LE(grad.RMSE(actual,gradPreds), 1.5);
}
