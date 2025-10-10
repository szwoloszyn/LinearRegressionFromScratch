#include <gtest/gtest.h>
#include <armadillo>
#include <cmath>
#include "batchgradientdescent.h"

using arma::mat, arma::vec;

class BatchGradTest : public ::testing::Test
{
protected:
    BatchGradientDescent gradModel{1,1};
    mat X1;
    vec y1;
    vec thetas1;
    void SetUp() override
    {
        X1 = {
            {1,1,3},
            {1,2,4}
        };
        y1 = {2,3};
        thetas1 = {1,1,1};
    }
};

TEST_F(BatchGradTest, calcGradWorks)
{
    vec gradient = gradModel.calculateGradient(X1,y1,thetas1);
    vec expectedGradient{7,11,25};
    ASSERT_EQ(gradient.n_elem, expectedGradient.n_elem);
    for (auto i = 0; i < gradient.n_elem; ++i) {
        ASSERT_EQ(gradient(i), expectedGradient(i));
    }
}
