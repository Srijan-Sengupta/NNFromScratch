#include <iostream>
#include <memory>
#include "Sequential.h"
#include "Linear.h"
#include "Sigmoid.h"
#include "MSELoss.h"
#include "Matrix.h"


void test_xor_training() {
    nn::Sequential model;
    model.add<nn::Linear>(2, 4, 0.5);
    model.add<nn::Sigmoid>();
    model.add<nn::Linear>(4, 1, 0.5);
    model.add<nn::Sigmoid>();

    nn::MSELoss criterion;

    // Update your X inputs to be row vectors instead of column vectors:
    std::vector X = {
        nn::Matrix({{0.0, 0.0}}), // Shape is now (1, 2)
        nn::Matrix({{0.0, 1.0}}),
        nn::Matrix({{1.0, 0.0}}),
        nn::Matrix({{1.0, 1.0}})
    };

    // Y targets should remain as (1, 1):
    std::vector Y = {
        nn::Matrix({{0.0}}),
        nn::Matrix({{1.0}}),
        nn::Matrix({{1.0}}),
        nn::Matrix({{0.0}})
    };

    int epochs = 10000;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double epoch_loss = 0.0;

        for (size_t i = 0; i < X.size(); ++i) {
            nn::Matrix pred = model.forward(X[i]);
            epoch_loss += criterion.forward(pred, Y[i]);

            nn::Matrix grad = criterion.backward();
            model.backward(grad);
        }

        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << epoch_loss / X.size() << "\n";
        }
    }

    std::cout << "\nEvaluating trained model:\n";
    for (size_t i = 0; i < X.size(); ++i) {
        nn::Matrix pred = model.forward(X[i]);
        double out_val = pred(0, 0);
        double target_val = Y[i](0, 0);

        std::cout << "Input: " << X[i](0,0) << "," << X[i](1,0)
                  << " | Target: " << target_val
                  << " | Pred: " << out_val << "\n";

        if (std::abs(out_val - target_val) > 0.1) {
            std::cerr << "Test failed for input " << i << "\n";
        }
    }
}

int main() {
    test_xor_training();
    return 0;
}