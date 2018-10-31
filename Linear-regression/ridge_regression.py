from numpy import dot
import numpy as np
import scipy.io
dataset = scipy.io.loadmat('dataset2.mat')

x_train = dataset["X_trn"]
y_train = dataset["Y_trn"]
x_test = dataset["X_tst"]
y_test = dataset["Y_tst"]


def ridge_regression_CF(x_train, y_train):
    lam = 1000
    X = np.array(x_train)
    ones = np.ones(len(X))
    X = np.column_stack((ones, X))
    y = np.array(y_train)

    Xt = X.T
    lamI = lam * np.identity(len(Xt))
    Inv = np.linalg.inv(np.dot(Xt, X) + lamI)
    w = np.dot(np.dot(Inv, Xt), y)
    return w

def ridge_regression_GD(x_train,y_train):
    lam = 1000
    learning_rate = 0.000001
    X = np.array(x_train)
    Y = np.array(y_train)
    W = np.array([[1], [1]])
    ones = np.ones(len(X))
    X = np.column_stack((ones, X))
    y = np.array(y_train)
    Xt = X.T
    for i in range(1000000):
        diff_W = dot(Xt,dot(X,W) - Y) + 2 * lam * W
        W = W - learning_rate * diff_W
    return W

def mean_sqr_error(x_data,y_data,w):
    y_pred = []
    for i in x_data:
        components = w[0] + w[1] * i
        y_pred.append(components)
    return ((y_data - y_pred) ** 2).mean()

W_CF = ridge_regression_CF(x_train, y_train)
w = ridge_regression_GD(x_train, y_train)
error_on_train = mean_sqr_error(x_train, y_train,W_CF)
error_on_test = mean_sqr_error(x_test,y_test,W_CF)


print("Weights from CF: \n" + str(W_CF))
print("Error on training data: " + str(error_on_train))
print("Error on test data: " + str(error_on_test))

error_on_train = mean_sqr_error(x_train, y_train,w)
error_on_test = mean_sqr_error(x_test,y_test,w)
print("Weights from GD: \n" + str(w))
print("Error on training data: " + str(error_on_train))
print("Error on test data: " + str(error_on_test))
