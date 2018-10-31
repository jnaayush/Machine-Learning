import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def sigmoid(W, X):
    return 1.0 / (1 + np.exp(-np.dot(X, W.T)))


def logistic_regression_GD(X, Y, lr=.0001):
    W = np.matrix([0,0,0])
    for i in range(3000):
        temp = sigmoid(W, X) - Y.reshape(X.shape[0], -1)
        dJdw = np.dot(temp.T, X)
        W = W - (lr * dJdw)
    return W


def test_model(W, X):
    pred_prob = sigmoid(W, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    return np.squeeze(pred_value)


def plot(X, y, W):
    x_0 = X[np.where(y == 0.0)]
    x_1 = X[np.where(y == 1.0)]
    plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='b', label='y = 0')
    plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1')
    x1 = np.arange(0, 1, 0.1)
    x2 = -(W[0, 0] + W[0, 1] * x1) / W[0, 2]
    plt.plot(x1, x2, c='k', label='reg line')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.savefig('results/plot.png')
    plt.show()


if __name__ == "__main__":
    #dataset3 = scipy.io.loadmat('hw02_data/dataset3.mat')
    dataset4 = scipy.io.loadmat('hw02_data/dataset3.mat')

    x_train = dataset4["X_trn"]
    y_train = dataset4["Y_trn"]
    x_test = dataset4["X_tst"]
    y_test = dataset4["Y_tst"]

    X = np.hstack((np.matrix(np.ones(x_train.shape[0])).T, x_train))
    Y = y_train.flatten()

    X_test = np.hstack((np.matrix(np.ones(x_test.shape[0])).T, x_test))
    Y_test = y_test.flatten()

    W = logistic_regression_GD(X, Y)

    print("Estimated regression coefficients:", W)
    y_predicted = test_model(W, X)
    print("Correct classifications:", np.sum(Y == y_predicted))
    print("Incorrect Classifications:", len(Y) - np.sum(Y == y_predicted))
    plot(X, Y, W)

    y_predicted_test = test_model(W, X_test)
    print("Correct test classifications:", np.sum(Y_test == y_predicted_test))
    print("Incorrect test Classifications:", len(Y_test) - np.sum(Y_test == y_predicted_test))

    plot(X_test, Y_test, W)
