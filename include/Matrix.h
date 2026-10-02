//
// Created by srijan on 13/09/26.
//

#ifndef NNFROMSCRATCH_TENSOR_H
#define NNFROMSCRATCH_TENSOR_H

#include "xtensor/xarray.hpp"

namespace nn {
    class Matrix {
    public:
        xt::xarray<std::float_t> data;
        Matrix(std::size_t rows, std::size_t cols);
        Matrix(std::initializer_list<std::initializer_list<std::float_t>> list);

        std::size_t get_rows() const;
        std::size_t get_cols() const;
        Matrix multiply(const Matrix& x) const;
        Matrix T() const;
        std::vector<std::size_t> shape() const;
        Matrix sigmoid() const;
        Matrix sigmoid_derivative() const;
        Matrix sum_rows() const;

        float& operator()(std::size_t i, std::size_t j);
        float operator()(std::size_t i, std::size_t j) const;

        Matrix operator+(const Matrix& x) const;
        Matrix operator-(const Matrix& x) const;
        Matrix operator*(const Matrix& x) const;
        Matrix operator*(const float& scalar) const;

        void print() const;
    };
} // nn

#endif //NNFROMSCRATCH_TENSOR_H
