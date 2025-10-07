#include <gtest/gtest.h>

TEST(AnotherOne, always_true) {
    EXPECT_EQ(42,42);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
