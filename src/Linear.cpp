//
// Created by srijan on 13/09/26.
//

#include "Linear.h"

#include <xtensor/xrandom.hpp>
#include <xtensor/xmath.hpp>

namespace nn {
    Linear::Linear(const std::size_t in, const std::size_t out, double lr) :
        weights(in, out),
        bias(1, out),
        input(0, 0),
        learning_rate(lr)
    {
        weights.data = xt::random::rand(weights.shape(), -0.9999, 0.9999);
        bias.data =  xt::random::rand(bias.shape(), -0.9999, 0.9999);;
    }

    Matrix Linear::forward(const Matrix &x) {
        input = x;
        return ((x*weights) + bias);
    }

    Matrix Linear::backward(const Matrix &delta) {
        Matrix grad_inp = delta * weights.T();
        Matrix grad_w = input.T() * delta;

        weights = weights + (grad_w * learning_rate);
        bias = bias + (delta * learning_rate);
        return grad_inp;
    }
} // nn