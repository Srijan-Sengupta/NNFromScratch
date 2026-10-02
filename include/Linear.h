//
// Created by srijan on 13/09/26.
//

#ifndef NNFROMSCRATCH_LINEAR_H
#define NNFROMSCRATCH_LINEAR_H
#include "Module.h"

namespace nn {
    class Linear : public Module {
    private:
        Matrix weights;
        Matrix bias;

        Matrix input;

        Matrix weights_grad;
        Matrix bias_grad;
    public:
        Linear(std::size_t in, std::size_t out);
        Matrix forward(const Matrix &x) override;
        Matrix backward(const Matrix &grad_out) override;
        void update(float alpha);
    };
} // nn

#endif //NNFROMSCRATCH_LINEAR_H
