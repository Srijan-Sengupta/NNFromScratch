cmake_minimum_required(VERSION 4.3)
project(NNFromScratch)

set(CMAKE_CXX_STANDARD 20)

find_package(xtensor-blas REQUIRED)
find_package(xtensor REQUIRED)

add_library(nn
        src/Matrix.cpp
        src/Linear.cpp
        src/Sigmoid.cpp
        src/Sequential.cpp
        src/MSELoss.cpp
)
target_include_directories(nn PUBLIC
        ${xtensor_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(nn PUBLIC xtensor BLAS::BLAS)
# TESTING

enable_testing()

add_executable(TensorTest test/mat_test.cpp)
add_executable(LinearTest test/lin_test.cpp)
add_executable(SigmoidTest test/sigmoid_test.cpp)
add_executable(SequentialTest test/seq_test.cpp)
add_executable(LibraryFullTest test/lib_test.cpp)
add_executable(HiggsModelTest test/higgs_test.cpp)

target_link_libraries(TensorTest PRIVATE nn)
target_link_libraries(LinearTest PRIVATE nn)
target_link_libraries(SigmoidTest PRIVATE nn)
target_link_libraries(SequentialTest PRIVATE nn)
target_link_libraries(LibraryFullTest PRIVATE nn)
target_link_libraries(HiggsModelTest PRIVATE nn)

add_test(NAME TensorTest COMMAND TensorTest)
add_test(NAME LinearTest COMMAND LinearTest)
add_test(NAME SigmoidTest COMMAND SigmoidTest)
add_test(NAME SequentialTest COMMAND SequentialTest)
add_test(NAME LibraryFullTest COMMAND LibraryFullTest)
add_test(NAME HiggsTest COMMAND HiggsModelTest)

