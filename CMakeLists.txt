cmake_minimum_required(VERSION 3.16)

project(LinearRegressionFromScratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Armadillo REQUIRED)

include_directories(
    ${ARMADILLO_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)
add_compile_options(-Wall -Wextra -pedantic)
file(GLOB SRC_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB HEADER_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/include/*.h)

option(BUILD_SHARED_LIBS "Build shared libraries instead of static" ON)

add_library(LinRegFromScratch ${LIB_TYPE}
    ${SRC_FILES}
    ${HEADER_FILES}
)

target_include_directories(LinRegFromScratch PUBLIC ${PROJECT_SOURCE_DIR}/include)


target_link_libraries(LinRegFromScratch PUBLIC ${ARMADILLO_LIBRARIES})


add_executable(LinearRegressionFromScratch src/main.cpp
)
target_link_libraries(LinearRegressionFromScratch PRIVATE LinRegFromScratch)

enable_testing()
add_subdirectory(tests)

include(GNUInstallDirs)
install(TARGETS LinearRegressionFromScratch
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
