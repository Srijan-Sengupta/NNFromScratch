//
// Created by srijan on 13/09/26.
//


#include "Linear.h"
#include "Sequential.h"
#include "Sigmoid.h"

int main() {

    nn::Sequential model;

    model.add<nn::Linear>(2, 4, 0.5);
    model.add<nn::Sigmoid>();
    model.add<nn::Linear>(4, 1, 0.5);
    model.add<nn::Sigmoid>();

    const nn::Matrix input({
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    });

    const auto output = model.forward(input);

    assert(output.get_rows() == 2);
    assert(output.get_cols() == 1);

    output.print();

    return 0;
}
