//
// Created by srijan on 02/10/26.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

#include "Matrix.h"
#include "Sequential.h"
#include "Linear.h"
#include "Sigmoid.h"
#include "MSELoss.h"

void load_higgs_data(const std::string& filepath, std::vector<std::vector<double>>& X_data, std::vector<std::vector<double>>& y_data, int limit) {
    std::ifstream file(filepath);
    std::string line;
    int count = 0;

    while (std::getline(file, line) && count < limit) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        std::getline(ss, cell, ',');
        y_data.push_back({std::stod(cell)});

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        X_data.push_back(row);
        count++;
    }
}

void evaluate_metrics(const nn::Matrix& predictions, const nn::Matrix& labels, int n, double& accuracy, double& f1_score) {
    int tp = 0, fp = 0, tn = 0, fn = 0;

    for (int i = 0; i < n; ++i) {
        double pred = predictions(i, 0) >= 0.5 ? 1.0 : 0.0;
        double actual = labels(i, 0);

        if (pred == 1.0 && actual == 1.0) tp++;
        else if (pred == 1.0 && actual == 0.0) fp++;
        else if (pred == 0.0 && actual == 0.0) tn++;
        else if (pred == 0.0 && actual == 1.0) fn++;
    }

    accuracy = static_cast<double>(tp + tn) / n;

    double precision = (tp + fp) > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
    double recall = (tp + fn) > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;

    f1_score = (precision + recall > 0) ? (2.0 * precision * recall) / (precision + recall) : 0.0;
}

int main() {
    std::vector<std::vector<double>> X_vec, y_vec;
    load_higgs_data("/home/srijan/Workspace/workdrive/srijan-sengupta/Workspace/CPP/NNFromScratch/extras/dataset/HIGGS.csv", X_vec, y_vec, 5000);

    int num_samples = X_vec.size();
    int num_features = X_vec.empty() ? 0 : X_vec[0].size();

    int train_samples = static_cast<int>(num_samples * 0.7);
    int test_samples = num_samples - train_samples;

    nn::Matrix X_train(train_samples, num_features);
    nn::Matrix y_train(train_samples, 1);
    nn::Matrix X_test(test_samples, num_features);
    nn::Matrix y_test(test_samples, 1);

    for(int i = 0; i < train_samples; i++) {
        for(int j = 0; j < num_features; j++) {
            X_train(i, j) = X_vec[i][j];
        }
        y_train(i, 0) = y_vec[i][0];
    }

    for(int i = train_samples; i < num_samples; i++) {
        int test_idx = i - train_samples;
        for(int j = 0; j < num_features; j++) {
            X_test(test_idx, j) = X_vec[i][j];
        }
        y_test(test_idx, 0) = y_vec[i][0];
    }

    for (int j = 0; j < num_features; j++) {
        double mean = 0.0;
        for (int i = 0; i < train_samples; i++) mean += X_train(i, j);
        mean /= train_samples;

        double variance = 0.0;
        for (int i = 0; i < train_samples; i++) variance += (X_train(i, j) - mean) * (X_train(i, j) - mean);
        variance /= train_samples;
        double std_dev = std::sqrt(variance + 1e-8);

        for (int i = 0; i < train_samples; i++) X_train(i, j) = (X_train(i, j) - mean) / std_dev;
        for (int i = 0; i < test_samples; i++) X_test(i, j) = (X_test(i, j) - mean) / std_dev;
    }

    double lr = 0.00001;
    nn::Sequential model;
    model.add<nn::Linear>(28, 128, lr);
    model.add<nn::Sigmoid>();
    model.add<nn::Linear>(128, 256, lr);
    model.add<nn::Sigmoid>();
    model.add<nn::Linear>(256, 128, lr);
    model.add<nn::Sigmoid>();
    model.add<nn::Linear>(128, 64, lr);
    model.add<nn::Sigmoid>();
    model.add<nn::Linear>(64, 1, lr);
    model.add<nn::Sigmoid>();

    nn::MSELoss criterion;
    int epochs = 1000;

    std::cout << "Starting training on " << train_samples << " samples..." << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        nn::Matrix out = model.forward(X_train);
        double loss = criterion.forward(out, y_train);
        nn::Matrix grad = criterion.backward();

        model.backward(grad);

        if (epoch % 10 == 0) {
            double acc, f1;
            evaluate_metrics(out, y_train, train_samples, acc, f1);
            std::cout << "Epoch: " << epoch << " | Loss: " << loss << " | Acc: " << acc << " | F1: " << f1 << "\n";
        }
    }

    std::cout << "\nEvaluating on " << test_samples << " test samples..." << std::endl;

    // Workaround: Pad test matrices to match train_samples (3500) to avoid broadcast errors
    nn::Matrix X_test_padded(train_samples, num_features);

    for (int i = 0; i < train_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            if (i < test_samples) {
                X_test_padded(i, j) = X_test(i, j);
            } else {
                X_test_padded(i, j) = 0.0; // Dummy data for padding
            }
        }
    }

    // Forward pass with padded data
    nn::Matrix test_out_padded = model.forward(X_test_padded);

    // Extract only the valid predictions
    nn::Matrix test_out(test_samples, 1);
    for (int i = 0; i < test_samples; i++) {
        test_out(i, 0) = test_out_padded(i, 0);
    }

    double test_loss = criterion.forward(test_out, y_test);

    double test_acc, test_f1;
    evaluate_metrics(test_out, y_test, test_samples, test_acc, test_f1);

    std::cout << "Test Loss: " << test_loss << "\n";
    std::cout << "Test Accuracy: " << test_acc << "\n";
    std::cout << "Test F1-Score: " << test_f1 << std::endl;

    return 0;
}