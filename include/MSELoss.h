//
// Created by srijan on 15/09/26.
//

#ifndef NNFROMSCRATCH_MSELOSS_H
#define NNFROMSCRATCH_MSELOSS_H

#include "Matrix.h"

namespace nn {
        class MSELoss {
        private:
                Matrix saved_pred;
                Matrix saved_target;
        public:
                MSELoss() = default;
                float forward(const Matrix &pred, const Matrix &target);
                Matrix backward() const;
        };
}

#endif //NNFROMSCRATCH_MSELOSS_H
