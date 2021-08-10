"""
Linear Regression using gradient descent from scratch using pytorch.
y = W1x1 + W2x2 + W3x3 + ... + B

Let m = #training examples, n = #number of features Sizes differ
a little bit from blog notation. It takes as input the following:
y is R^(1 x m), X is R^(n x m), w is R^(n x 1)

Programmed by Shivam Chhetry
* 10-08-2021
"""

import numpy as np

class LinearRegression:
    def __init__(self, print_cost=False):
        self.learning_rate = 1e-2
        self.total_iterations = 1000
        self.print_cost = print_cost

    def y_hat(self, X, w):
        return np.dot(w.T, X)

    def loss(self, yhat, y):
        C = 1 / self.m * np.sum(np.power(yhat - y, 2))

        return C

    def gradient_descent(self, w, X, y, yhat):
        dCdW = 2 / self.m * np.dot(X, (yhat - y).T)
        w = w - self.learning_rate * dCdW

        return w

    def main(self, X, y):
        # Add x1 = 1
        ones = np.ones((1, X.shape[1]))
        X = np.append(ones, X, axis=0)

        self.m = X.shape[1]
        self.n = X.shape[0]

        w = np.zeros((self.n, 1))

        for it in range(self.total_iterations + 1):
            yhat = self.y_hat(X, w)
            loss = self.loss(yhat, y)

            if it % 200 == 0:
                print(f"Loss at iteration {it} is {loss}")

            w = self.gradient_descent(w, X, y, yhat)

        return w


if __name__ == "__main__":
    X = np.random.rand(1, 500)
    y = 3 * X + 5 + np.random.randn(1, 500) * 0.1
    regression = LinearRegression()
    w = regression.main(X, y)


