//
// Created by srijan on 13/09/26.
//

#include "Sigmoid.h"
#include "Matrix.h"

namespace nn {
Matrix Sigmoid::forward(const Matrix &x) {
    return x.sigmoid();
}

Matrix Sigmoid::backward(const Matrix &x) {
    return  x.multiply(x.sigmoid_derivative());
}
}
