cmake_minimum_required(VERSION 3.16)

project(LinearRegressionFromScratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Armadillo REQUIRED)

include_directories(
    ${ARMADILLO_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB HEADER_FILES ${PROJECT_SOURCE_DIR}/include/*.h)

add_executable(LinearRegressionFromScratch ${SRC_FILES} ${HEADER_FILES})

target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES})

include(GNUInstallDirs)
install(TARGETS LinearRegressionFromScratch
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
