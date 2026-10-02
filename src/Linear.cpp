//
// Created by srijan on 13/09/26.
//

#include "Linear.h"

#include <xtensor/xrandom.hpp>

namespace nn {
    Linear::Linear(const std::size_t in, const std::size_t out) :
        weights(in, out),
        bias(1, out),
        input(0, 0),
        weights_grad(in, out),
        bias_grad(1, out)
    {
        weights.data = xt::random::rand(weights.shape(), 0.0001, 0.9999);
        bias.data =  xt::random::rand(weights.shape(), 0.0001, 0.9999);;
    }

    Matrix Linear::forward(const Matrix &x) {
        input = x;
        return ((x*weights) + bias);
    }

    Matrix Linear::backward(const Matrix &delta) {
        weights_grad = input.T() *  delta;
        bias_grad = delta.sum_rows();
        Matrix grad_input = delta * weights.T();
        return grad_input;
    }
} // nn