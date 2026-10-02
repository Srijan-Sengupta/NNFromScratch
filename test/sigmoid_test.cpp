//
// Created by srijan on 13/09/26.
//

#include "Matrix.h"
#include "Sigmoid.h"

int main() {
    nn::Matrix X({
        {0.0f, 1.0f},
        {-1.0f, 2.0f}
    });

    nn::Sigmoid sigmoid;

    auto Y = sigmoid.forward(X);
    Y.print();

    std::cout << "Y(0,1) = " << Y(0, 1) << '\n';
    assert(std::abs(Y(0, 0) - 0.5f) < 1e-5f);
    assert(std::abs(Y(0, 1) - 0.7310586f) < 1e-5f);
    assert(std::abs(Y(1, 0) - 0.2689414f) < 1e-5f);
    assert(std::abs(Y(1, 1) - 0.8807971f) < 1e-5f);
}
