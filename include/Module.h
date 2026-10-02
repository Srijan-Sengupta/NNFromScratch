//
// Created by srijan on 13/09/26.
//

#ifndef NNFROMSCRATCH_MODULE_H
#define NNFROMSCRATCH_MODULE_H
#include "Matrix.h"

namespace nn {
    class Module {
    public:
        virtual Matrix forward(const Matrix &x) = 0;
        virtual Matrix backward(const Matrix &x) = 0;
        virtual ~Module() = default;
    };
} // nn

#endif //NNFROMSCRATCH_MODULE_H
