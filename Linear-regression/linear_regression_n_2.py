from numpy import array, dot, transpose
from numpy.linalg import inv
import numpy as np
import scipy.io

dataset = scipy.io.loadmat('dataset1.mat')

x_train = dataset["X_trn"]
y_train = dataset["Y_trn"]
x_test = dataset["X_tst"]
y_test = dataset["Y_tst"]

x_train = np.power(x_train,[1,2])
x_test = np.power(x_test,[1,2])

def linear_regression_CF(x_train, y_train):
    X = np.array(x_train)
    ones = np.ones(len(X))
    X = np.column_stack((ones, X))
    y = np.array(y_train)

    Xt = X.T
    product = dot(Xt, X)
    theInverse = inv(product)
    w = dot(dot(theInverse, Xt), y)
    return w

def linear_regression_GD(x_train,y_train):
    learning_rate = 0.00001
    X = np.array(x_train)
    Y = np.array(y_train)
    W = np.array([[1], [1],[1]])
    ones = np.ones(len(X))
    X = np.column_stack((ones, X))
    y = np.array(y_train)
    Xt = X.T
    for i in range(100000):
        XWY = dot(X,W) - Y
        diff_W = dot(Xt,XWY)
        W_new = W - learning_rate * diff_W
        W = W_new
    return W

def mean_sqr_error(x_data,y_data,w):
    y_pred = []
    for i in x_data:
        y_i_pred = w[0] + w[1] * i[0] + w[2] * i[1]
        y_pred.append(y_i_pred)
    return ((y_data - y_pred) ** 2).mean()


W_CF = linear_regression_CF(x_train, y_train)
error_on_train_CF = mean_sqr_error(x_train, y_train, W_CF)
error_on_test_CF = mean_sqr_error(x_test, y_test, W_CF)

print("---------n = 2-------------- \n")
print("Closed Form solution:")
print("Weights CF: \n" + str(W_CF))
print("Error on training data: " + str(error_on_train_CF))
print("Error on test data: " + str(error_on_test_CF))

W_GD = linear_regression_GD(x_train,y_train)
error_on_train_GD = mean_sqr_error(x_train, y_train, W_GD)
error_on_test_GD = mean_sqr_error(x_test, y_test, W_GD)
print("\n\nGradient Descent solution:")
print("Weights GD: \n" + str(W_GD))
print("Error on training data: " + str(error_on_train_GD))
print("Error on test data: " + str(error_on_test_GD))

