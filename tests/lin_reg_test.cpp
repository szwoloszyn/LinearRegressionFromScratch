#include <gtest/gtest.h>
#include <armadillo>
#include "normalequation.h"

using arma::mat, arma::vec;

class LinRegTest : public ::testing::Test
{
protected:
    NormalEquation trivialTrained;
    NormalEquation oneFeatureTrained;
    // y = 2x
    mat XTrivial;
    vec yTrivial;

    // y = 2x + 1
    mat XOneFeature;
    vec yOneFeature;

    // kinda random
    mat XTwoFeature;
    vec yTwoFeature;

    mat XCVExample;
    vec yCVExample;

    void SetUp() override
    {

        XTrivial = (mat{ 1,2}).t();
        //XTrivial = XTrivial.t();
        yTrivial = {2,4};

        XOneFeature = (mat{0,1,2,3,4,5}).t();
        yOneFeature = {1,3,5,7,9,11};

        XTwoFeature = (mat{ {1,5,6}, {6,9,10} }).t();
        yTwoFeature = {-10,8,20};

        trivialTrained.fit(XTrivial, yTrivial);
        oneFeatureTrained.fit(XOneFeature, yOneFeature);

        XCVExample =( mat{0,1,2,3,4,5,6,7,8,9}).t();
        yCVExample = {0,1,2,3,4,5,6,7,8,9};
    }
};

TEST_F(LinRegTest, predictWorks)
{
    for (double i = -53.2; i < 16; i += 0.198) {
        double predictedTrivialOutcome = 2*i;
        double predictedOneFeatureOutcome = 2*i + 1;
        ASSERT_EQ(predictedTrivialOutcome, trivialTrained.predict({i}));
        ASSERT_EQ(predictedOneFeatureOutcome, oneFeatureTrained.predict({i}));
    }
}

TEST_F(LinRegTest, splitFoldsWorks)
{
    const size_t k = 4;
    std::vector<arma::mat> X_folds;
    std::vector<arma::vec> y_folds;

    EXPECT_THROW(
        trivialTrained.splitFolds(XCVExample, yCVExample, k, X_folds, y_folds),
        std::invalid_argument
    );
    X_folds.resize(k);
    y_folds.resize(k);

    trivialTrained.splitFolds(XCVExample, yCVExample, k, X_folds, y_folds);

    ASSERT_EQ(X_folds.size(), k);
    ASSERT_EQ(y_folds.size(), k);

    ASSERT_EQ(X_folds[0].n_elem,2);
    ASSERT_EQ(X_folds[3].n_elem,4);

    ASSERT_EQ(y_folds[0].n_elem,2);
    ASSERT_EQ(y_folds[3].n_elem,4);
}
