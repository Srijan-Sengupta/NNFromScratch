//
// Created by srijan on 13/09/26.
//

#include "../include/Matrix.h"

int main() {
    nn::Matrix a({
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    });

    a.print();

    a(0,0) = 10;
    a.print();

    nn::Matrix b({
    {5.0f, 6.0f},
    {7.0f, 8.0f}});
    b.print();
    const nn::Matrix c = b*a;
    c.print();
    c.T().print();
}
