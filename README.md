# Neural Network From Scratch (C++)

A modular, object-oriented deep learning micro-framework implemented entirely in C++. This project builds the foundational components of neural networks—including tensor operations, forward propagation, and backpropagation—without relying on external linear algebra or machine learning libraries.

## Architecture and Design

The framework mirrors the modular design of modern deep learning libraries, abstracting mathematical operations into reusable, chainable components based on the core project files.

* **`Module` Base Class:** The interface for all neural network layers, enforcing forward and backward propagation contracts via `Module.h`.


* **`Sequential` Container:** A model wrapper implemented in `Sequential.cpp` that chains multiple modules. It automatically pipes outputs to inputs during the forward pass and propagates gradients in reverse during the backward pass.


* **`Matrix` Library:** A custom linear algebra backend in `Matrix.cpp` handling memory allocation, dot products, transpositions, and element-wise operations.


* **Layers and Activations:** Includes dense, fully-connected operations via `Linear.cpp` and activation functions via `Sigmoid.cpp`.


* **Loss Functions:** Implements Mean Squared Error in `MSELoss.cpp` for evaluating network performance and computing the initial gradient.



## Repository Structure

```text
nnfromscratch/
├── CMakeLists.txt
├── include/
│   ├── Linear.h
│   ├── MSELoss.h
│   ├── Matrix.h
│   ├── Module.h
│   ├── Sequential.h
│   └── Sigmoid.h
├── src/
│   ├── Linear.cpp
│   ├── MSELoss.cpp
│   ├── Matrix.cpp
│   ├── Sequential.cpp
│   └── Sigmoid.cpp
└── test/
    ├── lib_test.cpp
    ├── lin_test.cpp
    ├── mat_test.cpp
    ├── seq_test.cpp
    └── sigmoid_test.cpp

```

## Mathematical Foundation

The framework executes standard backpropagation via the chain rule. For a linear layer processing input $X$ with weights $W$ and bias $b$:

* **Forward Pass:** $Z = X \cdot W^T + b$
* **Backward Pass:** Given an incoming gradient $dZ$ from the subsequent layer:
* Weight Gradient: $dW = dZ^T \cdot X$
* Input Gradient (passed to previous layer): $dX = dZ \cdot W$



## Build Instructions

This project requires CMake to generate the build files and compile the source and test executables.

```bash
git clone https://github.com/srijan-sengupta/nnfromscratch.git
cd nnfromscratch

mkdir build && cd build
cmake ..
make

```

## Usage Example

The object-oriented design allows for network instantiation and training loops to be written cleanly and sequentially.

```cpp
#include "Sequential.h"
#include "Linear.h"
#include "Sigmoid.h"
#include "MSELoss.h"
#include "Matrix.h"

int main() {
    // Initialize a 2-layer neural network
    Sequential model;
    model.add(new Linear(2, 4));    // 2 inputs, 4 hidden nodes
    model.add(new Sigmoid());
    model.add(new Linear(4, 1));    // 4 hidden nodes, 1 output
    model.add(new Sigmoid());

    MSELoss criterion;

    // Forward Pass (Assuming 'x' and 'y' are pre-defined Matrix objects)
    Matrix predictions = model.forward(x);

    // Compute Loss
    float loss = criterion.forward(predictions, y);

    // Backward Pass
    Matrix initial_gradient = criterion.backward();
    model.backward(initial_gradient);

    return 0;
}

```

## Testing

The `test/` directory contains isolated unit tests for the mathematical components and layer operations. Dedicated test files include `lib_test.cpp`, `lin_test.cpp`, `mat_test.cpp`, `seq_test.cpp`, and `sigmoid_test.cpp`. Run the compiled test executables from your build directory to verify matrix arithmetic, forward pass outputs, and gradient consistency across the network layers.