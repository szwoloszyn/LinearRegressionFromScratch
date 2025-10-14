# tests/CMakeLists.txt

add_executable(unit_tests
	test_main.cpp
	norm_eq_test.cpp
	lin_reg_test.cpp
	batch_grad_test.cpp
)

target_link_libraries(unit_tests
	LinRegFromScratch
	gtest
	gtest_main
	pthread
)

add_test(NAME unit_tests COMMAND unit_tests)
target_compile_definitions(unit_tests PRIVATE UNIT_TEST)
