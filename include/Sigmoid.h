//
// Created by srijan on 13/09/26.
//

#ifndef NNFROMSCRATCH_SIGMOID_H
#define NNFROMSCRATCH_SIGMOID_H
#include "Module.h"

namespace nn {
    class Sigmoid: public Module {
    private:
        Matrix saved_out;
    public:
        Sigmoid() = default;
        Matrix forward(const Matrix &x) override;
        Matrix backward(const Matrix &x) override;
    };
}

#endif //NNFROMSCRATCH_SIGMOID_H
