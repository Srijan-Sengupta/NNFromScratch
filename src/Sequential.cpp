//
// Created by srijan on 13/09/26.
//

#include "Sequential.h"

#include "Linear.h"

namespace nn {
    Matrix Sequential::forward(const Matrix &x) {
        Matrix output = x;

        for (const auto& module: modules) {
            output = module->forward(output);
        }

        return output;
    }

    Matrix Sequential::backward(const Matrix &x) {
        Matrix grad = x;

        for (auto it = modules.rbegin(); it != modules.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
        return grad;
    }
}
