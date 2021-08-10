"""
Logistic Regression classification from scratch
on Load_breast_cancer inbuild datasts.
Feel free to use this code for your learning.

Programmed by @Shivam Chhetry
* 10-08-2021
"""
# Import Libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

class LogisticRegression():
    def __init__(self, learning_rate, n_iters):
        self.lr = learning_rate
        self.iterations = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # number of parameters
        n_samples, n_features = X.shape
        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent Cal
        for _ in range(self.iterations):
            # approximate y with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # derivative of weights and bias
            dW = (1/n_samples) * np.dot(X.T, (y_predicted - y)) # Derivatives of weight
            dB = (1/n_samples) * np.sum(y_predicted - y)

            self.weights = self.weights - self.lr*dW
            self.bias = self.bias - self.lr*dB

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    breast_cancer = datasets.load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target

    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=1234)

    # Calculating accruacy
    def accuracy(y_true, y_pred):
        acu = np.sum(y_true == y_pred)/ len(y_true)
        return acu
    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    accuracy_result = accuracy(y_test, y_pred)


    print(f'Logistic Regression Classification accuracy :  {accuracy(y_test, y_pred)}')



