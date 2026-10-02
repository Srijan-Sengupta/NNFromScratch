//
// Created by srijan on 15/09/26.
//

#include  "MSELoss.h"

namespace nn {
    float MSELoss::forward(const Matrix &pred, const Matrix &target) {
        saved_pred = pred;
        saved_target = target;

        const Matrix diff = target - pred;
        const float loss = 0.5f * xt::sum((diff.multiply(diff).data))();
        return loss;
    }

    Matrix MSELoss::backward() const {
        return saved_target - saved_pred;
    }
}