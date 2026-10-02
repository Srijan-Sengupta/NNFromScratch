//
// Created by srijan on 13/09/26.
//

#include "Sigmoid.h"
#include "Matrix.h"

namespace nn {
Matrix Sigmoid::forward(const Matrix &x) {
    saved_out = x.sigmoid();
    return saved_out;
}

Matrix Sigmoid::backward(const Matrix &x) {
    Matrix derivative = saved_out.sigmoid_derivative();
    return x.multiply(derivative);
}
}
