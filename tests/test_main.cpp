#include <gtest/gtest.h>
#include <armadillo>

TEST(test_of_tests, always_true) {
    EXPECT_EQ(42,42);
}

TEST(arma_operations, isPlusElementWise) {
    arma::vec a = {1,2,3};
    arma::vec b = {2,3,4};
    arma::vec diff = b-a;
    arma::vec expectedDiff = {1,1,1};
    ASSERT_EQ(diff.size(), expectedDiff.size());
    for (size_t i = 0; i < diff.size(); ++i) {
        ASSERT_EQ(diff[i], expectedDiff[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
