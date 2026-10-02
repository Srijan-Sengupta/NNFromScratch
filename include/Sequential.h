//
// Created by srijan on 13/09/26.
//

#ifndef NNFROMSCRATCH_SEQUENTIAL_H
#define NNFROMSCRATCH_SEQUENTIAL_H
#include "Module.h"
#include <utility>
#include <memory>
#include <vector>

namespace nn{
    class Sequential: public Module {
    private:
        std::vector<std::unique_ptr<Module>> modules;
    public:
        template<typename T, typename... Args>
        void add(Args&&... args) {
            modules.push_back(
                std::make_unique<T>(
                    std::forward<Args>(args)...
                    ));
        }

        Matrix forward(const Matrix &x) override;
        Matrix backward(const Matrix &x) override;
        void update(const float alpha);
    };
}

#endif //NNFROMSCRATCH_SEQUENTIAL_H
