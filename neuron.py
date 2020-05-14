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
        self.weights = np.random.normal(0, 1, 6)
        self.bias = np.random.normal(0, 1, 3)

    def feedforward(self, x):
        h1 = sigmoid(self.weights[0] * x[0] + self.weights[1] * x[1] + self.bias[0])
        h2 = sigmoid(self.weights[2] * x[0] + self.weights[3] * x[1] + self.bias[1])
        o1 = sigmoid(self.weights[4] * h1 + self.weights[5] * h2 + self.bias[2])
        return o1

    def train(self, data, all_y_trues):

        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):

                sum_h1 = self.weights[0] * x[0] + self.weights[1] * x[1] + self.bias[0]
                h1 = sigmoid(sum_h1)

                sum_h2 = self.weights[2] * x[0] + self.weights[3] * x[1] + self.bias[1]
                h2 = sigmoid(sum_h2)

                sum_o1 = self.weights[4] * h1 + self.weights[5] * h2 + self.bias[2]
                o1 = sigmoid(sum_o1)
                y_pred = o1

                d_L_d_ypred = -2 * (y_true - y_pred)

                # Нейрон o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.weights[4] * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.weights[5] * deriv_sigmoid(sum_o1)

                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                self.weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.bias[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                self.weights[2] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.weights[3] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.bias[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                self.weights[4] -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.weights[5] -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.bias[2] -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))


def run():
    data = np.array([
        [-2, -1],
        [25, 6],
        [17, 4],
        [-15, -6],
    ])

    all_y_trues = np.array([
        1,
        0,
        0,
        1,
    ])
    network = NeuralNetwork()
    network.train(data, all_y_trues)
