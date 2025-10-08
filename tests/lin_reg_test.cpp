#include <gtest/gtest.h>
#include <armadillo>
#include <cmath>
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

TEST_F(LinRegTest, predictSingleValueWorks)
{
    for (double i = -53.2; i < 16; i += 0.198) {
        double predictedTrivialOutcome = 2*i;
        double predictedOneFeatureOutcome = 2*i + 1;
        ASSERT_EQ(predictedTrivialOutcome, trivialTrained.predictSingleValue({i}));
        ASSERT_EQ(predictedOneFeatureOutcome, oneFeatureTrained.predictSingleValue({i}));
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

TEST_F(LinRegTest, concatFoldsWorks)
{
    std::vector<arma::vec> folds = { {1,2}, {3,4}, {5,6} };
    const size_t excludeIdx = 1;
    auto concatedFolds = trivialTrained.concatFolds(folds,excludeIdx);

    vec excpectedConcatedFolds = {1,2,5,6};
    ASSERT_EQ(concatedFolds.size(), excpectedConcatedFolds.size());
    for (auto i = 0; i < excpectedConcatedFolds.size(); ++i) {
        ASSERT_EQ(concatedFolds[i], excpectedConcatedFolds[i]);
    }
}

TEST_F(LinRegTest, CrossValWorks)
{
    // TODO when cv finished
}

TEST_F(LinRegTest, RmseWorks)
{
    vec actual = {100,200,300};
    vec predicted = {101,199,275};
    double RMSE = trivialTrained.RMSE(actual, predicted);
    double predictedRMSE = sqrt(209);

    ASSERT_EQ(RMSE, predictedRMSE);

    predicted = {1,2};

    EXPECT_THROW(
        trivialTrained.RMSE(actual, predicted),
        std::logic_error
    );


}
