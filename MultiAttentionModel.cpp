//
// Created by WUlRa on 21/04/2024.
//
#include <iostream>
#include <utility>
#include <vector>
#include <cmath>
#include "Matrix.h"

#ifndef ATTENTION_MULTIHEADATTENTION_H
#define ATTENTION_MULTIHEADATTENTION_H

class MultiHeadAttention {
private:
    int dModel; // Dimensionality of the model
    int numHeads; // Number of attention heads
    int headSize; // Size of each attention head
    double dropout; // Dropout rate
    Matrix queryWeights = Matrix(0, 0);
    Matrix keyWeights = Matrix(0, 0);
    Matrix valueWeights = Matrix(0, 0);
    Matrix outputWeights = Matrix(0, 0);
    std::vector<Matrix> outputs;
    std::vector<std::vector<Matrix>> inputs;

public:
    MultiHeadAttention(int dModel, int numHeads, double dropout) {
        this->dModel = dModel;
        this->numHeads = numHeads;
        this->headSize = dModel / numHeads;
        this->dropout = dropout;

        // Weight initialization
        this->queryWeights = this->_initializeWeights(this->dModel, this->dModel);
        this->keyWeights = this->_initializeWeights(this->dModel, this->dModel);
        this->valueWeights = this->_initializeWeights(this->dModel, this->dModel);
        this->outputWeights = this->_initializeWeights(this->dModel, this->dModel);
    }

    Matrix _initializeWeights(int rows, int cols) {
        Matrix weights(rows, cols);
        weights.randomize(-1 / sqrt(cols), 1 / sqrt(cols)); // Random initialization with a normal distribution
        return weights;
    }

    Matrix _scaledDotProductAttention(Matrix query, Matrix key, Matrix value) {
        // Scaled dot-product attention
        Matrix attention = this->calculateAttentionScores(std::move(query),  std::move(key));
        attention = attention.softmax(); // Apply softmax to get attention weights
        attention = attention.dot(std::move(value));
        return attention;
    }

    Matrix calculateAttentionScores(Matrix query, Matrix key) {
        // Calculate attention scores
        Matrix attention = key.dot(query.transpose());
        attention = attention / sqrt(this->headSize);
        return attention;
    }

    Matrix _feedForward(Matrix input) {
        // Feedforward network
        Matrix attentionOutputs(this->dModel, 0);

        for (int i = 0; i < this->numHeads; i++) {
            int start = i * this->headSize;
            int end = (i + 1) * this->headSize;
            Matrix querySplice = this->queryWeights.getSubMatrix(start, end, 0, this->dModel);
            Matrix keySplice = this->keyWeights.getSubMatrix(start, end, 0, this->dModel);
            Matrix valueSplice = this->valueWeights.getSubMatrix(start, end, 0, this->dModel);
            Matrix query = input.dot(querySplice.transpose());
            Matrix key = input.dot(keySplice.transpose());
            Matrix value = input.dot(valueSplice.transpose());
            attentionOutputs = attentionOutputs.append(this->_scaledDotProductAttention(query, key, value));

        }
        attentionOutputs.concat();
        attentionOutputs = attentionOutputs * this->outputWeights;

        return attentionOutputs;
    }

    Matrix backward(Matrix gradients) {
        // Backpropagation
        Matrix queryGradients(this->dModel, 0);
        Matrix keyGradients(this->dModel, 0);
        Matrix valueGradients(this->dModel, 0);

        Matrix outputGradients = gradients * this->outputWeights.transpose();
        this->outputWeights = this->outputWeights - gradients * this->outputs[0] * this->dropout;

        for(int i = 0; i < this->numHeads; i++) {

            int start = i * this->headSize;
            int end = (i + 1) * this->headSize;
            Matrix querySplice = this->queryWeights.getSubMatrix(start, end, 0, this->dModel);
            Matrix keySplice = this->keyWeights.getSubMatrix(start, end, 0, this->dModel);
            Matrix valueSplice = this->valueWeights.getSubMatrix(start, end, 0, this->dModel);

            Matrix query = this->inputs[0][0].dot(querySplice.transpose());
            Matrix key = this->inputs[0][1].dot(keySplice.transpose());
            Matrix value = this->inputs[0][2].dot(valueSplice.transpose());


            // Backpropagate through attention mechanism
            std::vector<Matrix> grad_attention = this->_scaledDotProductAttentionBackpropagate(outputGradients, query, key, value);

            // Update input gradients
            queryGradients = queryGradients.append(grad_attention[0]); // query gradient
            keyGradients = keyGradients.append(grad_attention[1]); // key gradient
            valueGradients = valueGradients.append(grad_attention[2]);// value gradient
        }


        // Update input weights
        this->queryWeights = this->queryWeights - (queryGradients * this->dropout);
        this->keyWeights = this->keyWeights - (keyGradients * this->dropout);
        this->valueWeights = this->valueWeights - (valueGradients * this->dropout);

        return outputGradients;
    }

    std::vector<Matrix> _scaledDotProductAttentionBackpropagate(Matrix grad_output, Matrix query, Matrix key, Matrix value) {

        // Compute gradients with respect to attention weights
        Matrix scores = this->calculateAttentionScores(query, key);
        Matrix attention = scores.softmax();

        // Calculate the gradient of the loss with respect to value
        Matrix grad_value = attention.dot(grad_output);

        // Calculate the gradient of the loss with respect to attention
        Matrix grad_attention = grad_output.dot(std::move(value));

        // Calculate the gradient of the loss with respect to attention
        grad_attention = (grad_attention.softmaxBackpropagate(attention)) *  ((float) 1 / (float) this->headSize);

        // Calculate the gradient of the loss with respect to key
        Matrix grad_key = grad_attention.dot(query.transpose());

        // Calculate the gradient of the loss with respect to query
        Matrix grad_query = grad_attention.dot(key.transpose());

        return std::vector<Matrix>{grad_query, grad_key, grad_value};
    }

    Matrix predict(const Matrix& input) {
        this->outputs = std::vector<Matrix>();
        this->inputs.push_back(std::vector<Matrix>{input, input, input});
        Matrix output = this->_feedForward(input);
        this->outputs.push_back(output);
        return output;
    }

    Matrix backpropagate(Matrix gradients) {
        Matrix inputsGradients = this->backward(std::move(gradients));
        return inputsGradients;
    }

    void print() {
        for (auto & output : this->outputs) {
            output.print();
        }
    }
};

#endif //ATTENTION_MULTIHEADATTENTION_H
