//
// Created by srijan on 13/09/26.
//

#include "Matrix.h"
#include <xtensor/xio.hpp>
#include <xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <vector>

namespace nn {
    Matrix::Matrix(std::size_t rows, std::size_t cols) : data(xt::zeros<float>({rows, cols})) {
    }

    Matrix::Matrix(std::initializer_list<std::initializer_list<float>> list) {
        std::size_t rows = list.size();
        std::size_t cols = list.begin()->size();

        data = xt::xarray<float>::from_shape({rows, cols});

        std::size_t i = 0;

        for (const auto& row : list) {
            assert(row.size() == cols);
            std::size_t j = 0;
            for (const float value : row) {
                data(i, j) = value;
                ++j;
            }
            ++i;
        }
    }

    std::size_t Matrix::get_rows() const {
        return data.shape()[0];
    }

    std::size_t Matrix::get_cols() const {
        return data.shape()[1];
    }

    float &Matrix::operator()(std::size_t i, std::size_t j) {
        return data(i, j);
    }

    float Matrix::operator()(std::size_t i, std::size_t j) const {
        return data(i, j);
    }

    Matrix Matrix::operator+(const Matrix &x) const {
        std::cout << "Add: " << get_rows() << " " << get_cols() << " " << x.get_rows()  << " " << x.get_cols()<< std::endl;
        assert(get_cols() == x.get_cols());

        Matrix result(get_rows(), get_cols());

        result.data = data + x.data;

        return result;
    }

    Matrix Matrix::operator-(const Matrix &x) const {
        assert(get_cols() == x.get_cols());

        Matrix result(get_rows(), get_cols());

        result.data = data - x.data;

        return result;
    }

    Matrix Matrix::operator*(const Matrix &x) const {
        std::cout<< "Multiply: " << get_rows() << " " << get_cols() << " " << x.get_rows()  << " " << x.get_cols()<< std::endl;
        assert(get_cols() == x.get_rows());

        Matrix result(get_rows(), x.get_cols());

        result.data = xt::linalg::dot(data, x.data);

        return result;
    }
    Matrix Matrix::operator*(const float &scalar) const {
        Matrix result(get_rows(), get_cols());
        result.data = data * scalar;
        return result;
    }

    // Hamard multiplication
    Matrix Matrix::multiply(const Matrix &x) const {
        assert(get_rows() == x.get_rows());
        assert(get_cols() == x.get_cols());

        Matrix result(get_rows(), get_cols());
        result.data = data * x.data;
        return result;
    }

    Matrix Matrix::T() const {
        Matrix result(get_rows(), get_cols());

        result.data = xt::transpose(data);
        return result;
    }

    std::vector<std::size_t> Matrix::shape() const {
        return {get_rows(), get_cols()};
    }

    Matrix Matrix::sigmoid() const {
        Matrix result(get_rows(), get_cols());
        result.data = 1.0f / (1.0f + xt::exp(-data));
        return result;
    }

    Matrix Matrix::sigmoid_derivative() const {
        Matrix result(get_rows(), get_cols());
        result.data = data * (1.0f - data);
        return result;
    }

    Matrix Matrix::sum_rows() const {
        Matrix result(1, get_cols());
        result.data = xt::sum(data, 0);
        return result;
    }

    void Matrix::print() const {
        std::cout << data << std::endl;
    }
} // nn