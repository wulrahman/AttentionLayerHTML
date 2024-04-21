//
// Created by WUlRa on 21/04/2024.
//

#ifndef ATTENTION_MATRIX_H
#include <vector>
#include <cmath>

class Matrix {


public:
    int rows;
    int cols;
    std::vector<std::vector<double>> data;
    Matrix(int rows, int cols) {
        this->rows = rows;
        this->cols = cols;
        this->data = std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0));
    }

    void randomize(double lower, double upper) {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                this->data[i][j] = lower + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (upper - lower)));
            }
        }
    }


    Matrix operator*(Matrix other) {
        Matrix result(this->rows, other.cols);

        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < this->cols; k++) {
                    result.data[i][j] += this->data[i][k] * other.data[k][j];
                }
            }
        }

        return result;
    }

    Matrix dot(Matrix other) {
        if (this->cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions are not compatible for dot product");
        }

        Matrix result(this->rows, other.cols);

        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < this->cols; k++) {
                    result.data[i][j] += this->data[i][k] * other.data[k][j];
                }
            }
        }

        return result;
    }
    Matrix concat() {
        Matrix result(this->rows, this->cols);

        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result.data[i][j] = this->data[i][j];
            }
        }

        return result;
    }

    Matrix append(Matrix other) {
        Matrix result(this->rows, this->cols + other.cols);

        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result.data[i][j] = this->data[i][j];
            }
        }

        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                result.data[i][this->cols + j] = other.data[i][j];
            }
        }

        return result;
    }

    Matrix getSubMatrix(int rowStart, int rowEnd, int colStart, int colEnd) {
        Matrix result(rowEnd - rowStart, colEnd - colStart);

        for (int i = rowStart; i < rowEnd; i++) {
            for (int j = colStart; j < colEnd; j++) {
                result.data[i - rowStart][j - colStart] = this->data[i][j];
            }
        }

        return result;
    }

    Matrix operator*(double scalar) {
        Matrix result(this->rows, this->cols);

        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result.data[i][j] = this->data[i][j] * scalar;
            }
        }

        return result;
    }

    Matrix operator/(double scalar) {
        Matrix result(this->rows, this->cols);

        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result.data[i][j] = this->data[i][j] / scalar;
            }
        }

        return result;
    }

    Matrix operator+(Matrix other) {
        Matrix result(this->rows, this->cols);

        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result.data[i][j] = this->data[i][j] + other.data[i][j];
            }
        }

        return result;
    }

    Matrix operator-(Matrix other) {
        Matrix result(this->rows, this->cols);

        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result.data[i][j] = this->data[i][j] - other.data[i][j];
            }
        }

        return result;
    }

    Matrix transpose() {
        Matrix result(this->cols, this->rows);

        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                result.data[j][i] = this->data[i][j];
            }
        }

        return result;
    }

    Matrix softmax() {
        Matrix result(this->rows, this->cols);

        for (int i = 0; i < this->rows; i++) {
            double sum = 0;
            for (int j = 0; j < this->cols; j++) {
                result.data[i][j] = exp(this->data[i][j]);
                sum += result.data[i][j];
            }

            for (int j = 0; j < this->cols; j++) {
                result.data[i][j] /= sum;
            }
        }

        return result;
    }

    Matrix softmaxBackpropagate(const Matrix& grad) const {
        Matrix result(this->rows, this->cols);

        for (int i = 0; i < this->rows; i++) {
            double sum = 0;
            for (int j = 0; j < this->cols; j++) {
                sum += exp(this->data[i][j]);
            }
            for (int j = 0; j < this->cols; j++) {
                double s = exp(this->data[i][j]) / sum;
                result.data[i][j] = (1 - s) * s * grad.data[i][j];
            }
        }

        return result;
    }


    Matrix gradient(std::vector<Matrix> inputs) {
        Matrix result(this->rows, this->cols);

        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                for (int k = 0; k < inputs.size(); k++) {
                    result.data[i][j] += this->data[i][j] * inputs[k].data[i][j];
                }
            }
        }

        return result;
    }

    void print() {
        for (int i = 0; i < this->rows; i++) {
            for (int j = 0; j < this->cols; j++) {
                std::cout << this->data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }


};
#define ATTENTION_MATRIX_H

#endif //ATTENTION_MATRIX_H
