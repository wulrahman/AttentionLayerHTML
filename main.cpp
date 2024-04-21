#include <iostream>
#include <vector>
#include <cmath>
#include "Matrix.h"
#include "MultiHeadAttention.h"

int main() {
    MultiHeadAttention attention(10, 10, 0.001);
    Matrix inputs(10, 10);
    // Define sample input
    double add = 0.1;
    for (int i = 0; i < inputs.rows; i++) {
        for (int j = 0; j < inputs.cols; j++) {
            inputs.data[i][j] = add;
            add*=1.01;
        }
    }

    double add2 = 0.1;

    Matrix targets(10, 10);
    // Define sample target
    for (int i = 0; i < targets.rows; i++) {
        for (int j = 0; j < targets.cols; j++) {
            targets.data[i][j] = add2;
            add2*=1.02;

        }
    }

    int numEpochs = 1000;
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        Matrix outputs = attention.predict(inputs);
        Matrix backpropagated = targets - outputs;
        attention.backpropagate(backpropagated);
    }
    Matrix outputs = attention.predict(inputs);

    std::cout << "Output\t\tTarget" << std::endl;
    for (int i = 0; i < outputs.rows; i++) {
        for (int j = 0; j < outputs.cols; j++) {
            std::cout << outputs.data[i][j] << "\t\t" << targets.data[i][j] << std::endl;
        }
    }
    return 0;
}
