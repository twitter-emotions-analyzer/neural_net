import random

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class NeuralNetwork:

    def __init__(self):
        self.weights = np.random.normal(0, 1, 12)
        self.bias = np.random.normal(0, 1, 3)

    def func1(self, x):
        return self.weights[0] * x[0] + self.weights[1] * x[1] + self.weights[2] * x[2] + self.weights[3] * x[3] + \
               self.weights[4] * x[4] + self.bias[0]

    def func2(self, x):
        return self.weights[5] * x[0] + self.weights[6] * x[1] + self.weights[7] * x[2] + self.weights[8] * x[3] + \
               self.weights[9] * x[4] + self.bias[1]

    def feedforward(self, x):
        h1 = sigmoid(self.func1(x))
        h2 = sigmoid(self.func2(x))
        o1 = sigmoid(self.weights[10] * h1 + self.weights[11] * h2 + self.bias[2])
        return o1

    def train(self, data, all_y_trues):

        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.func1(x)
                h1 = sigmoid(sum_h1)

                sum_h2 = self.func2(x)
                h2 = sigmoid(sum_h2)

                sum_o1 = self.weights[10] * h1 + self.weights[11] * h2 + self.bias[2]
                o1 = sigmoid(sum_o1)
                y_pred = o1

                d_L_d_ypred = -2 * (y_true - y_pred)

                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.weights[4] * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.weights[5] * deriv_sigmoid(sum_o1)

                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_w3 = x[2] * deriv_sigmoid(sum_h1)
                d_h1_d_w4 = x[3] * deriv_sigmoid(sum_h1)
                d_h1_d_w5 = x[4] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                d_h2_d_w6 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w7 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_w8 = x[2] * deriv_sigmoid(sum_h2)
                d_h2_d_w9 = x[3] * deriv_sigmoid(sum_h2)
                d_h2_d_w10 = x[4] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                self.weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.weights[2] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w3
                self.weights[3] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w4
                self.weights[4] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w5
                self.bias[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                self.weights[5] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w6
                self.weights[6] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w7
                self.weights[7] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w8
                self.weights[8] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w9
                self.weights[9] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w10
                self.bias[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                self.weights[10] -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.weights[11] -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.bias[2] -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))


def average(d_list):
    result = 0
    for t in d_list:
        result += t
    if len(d_list) > 0:
        result /= len(d_list)
    return result


network = NeuralNetwork()


def calc(q_list):
    global network
    return network.feedforward(np.asarray(q_list))


def run():
    global network
    data = []
    y_trues = []
    for i in range(0, 50):
        data.append([])
        for j in range(0, 5):
            data[i].append(random.uniform(-1, 1))
        y_trues.append(average(data[i]))

    network.train(np.asarray(data), np.asarray(y_trues))
