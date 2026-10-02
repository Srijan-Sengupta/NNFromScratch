//
// Created by srijan on 13/09/26.
//

#include "Matrix.h"

int main() {
    nn::Matrix X({
    {1.0f, 2.0f, 3.0f},
    {4.0f, 5.0f, 6.0f}
});

    nn::Matrix bias({
        {10.0f, 20.0f, 30.0f}
    });

    auto Y = X + bias;

    assert(Y(0, 0) == 11.0f);
    assert(Y(0, 1) == 22.0f);
    assert(Y(0, 2) == 33.0f);

    assert(Y(1, 0) == 14.0f);
    assert(Y(1, 1) == 25.0f);
    assert(Y(1, 2) == 36.0f);
    Y.print();
}
