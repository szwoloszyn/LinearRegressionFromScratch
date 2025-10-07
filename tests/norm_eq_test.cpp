#include <gtest/gtest.h>
#include <armadillo>
#include "normalequation.h"

using arma::mat, arma::vec;

class NormEqTest : public ::testing::Test
{
protected:
    NormalEquation normEqModel;

    // y = 2x
    mat XTrivial;
    vec yTrivial;

    // y = 2x + 1
    mat XOneFeature;
    vec yOneFeature;

    // kinda random
    mat XTwoFeature;
    vec yTwoFeature;
    void SetUp() override
    {

        XTrivial = (mat{ 1,2}).t();
        //XTrivial = XTrivial.t();
        yTrivial = {2,4};

        XOneFeature = (mat{0,1,2,3,4,5}).t();
        yOneFeature = {1,3,5,7,9,11};

        XTwoFeature = (mat{ {1,5,6}, {6,9,10} }).t();
        yTwoFeature = {-10,8,20};
    }
};

TEST_F(NormEqTest, normalEqProperlySolved)
{
    auto thetas = normEqModel.solveNormalEquation(XOneFeature,yOneFeature);
    vec expectedThetas{1.0,2.0};
    std::cout << thetas;
    ASSERT_EQ(thetas.size(), expectedThetas.size());
    for (auto i = 0; i < thetas.size(); ++i) {
        ASSERT_EQ(thetas[i], expectedThetas[i]);
    }
}
