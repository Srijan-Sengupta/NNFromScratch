//
// Created by srijan on 15/09/26.
//

#ifndef NNFROMSCRATCH_MSELOSS_H
#define NNFROMSCRATCH_MSELOSS_H

#include "Matrix.h"

namespace nn {
        class MSELoss {
        public:
                float forward(Matrix &pred, Matrix &target);
                Matrix backward(Matrix &pred, Matrix &target);
        };
}

#endif //NNFROMSCRATCH_MSELOSS_H
