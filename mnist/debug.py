import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

data = pd.read_csv('data/mnist_train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_train = data.T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255


test_data = pd.read_csv('data/mnist_test.csv')
test_data = np.array(test_data)
m2, n2 = test_data.shape
np.random.shuffle(test_data)

data_test_t = test_data.T
Y_test = data_test_t[0]
X_test = data_test_t[1:n]
X_test = X_test / 255

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def deriv_ReLU(Z):
    return Z > 0

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# input: [1 2 3]
# return:
# [
#   [1 0 0],
#   [0 1 0],
#   [0 0 1] 
# ]
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    print("m: ", m)
    print("dZ2.dot(A1.T): ", m * dZ2.dot(A1.T))
    dW2 = 1 / m * dZ2.dot(A1.T)
    print('dZ2.shape', dZ2.shape)
    print('A1.shape', A1.shape)
    print('dW2.shape', dW2.shape)
    print('dW2', dW2)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    print("W2.shape", W2.shape)
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    #print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = load_weights()

    # for i in range(iterations):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
    W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    # if i % 10 == 0:
    #     print("Iteration: ", i)
    #     print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def save_weights(W1, b1, W2, b2):
    np.save("model/W1.npy", W1)
    np.save("model/b1.npy", b1)
    np.save("model/W2.npy", W2)
    np.save("model/b2.npy", b2)

def load_weights():
    if os.path.isfile("model/W1.npy"):
        W1 = np.load("model/W1.npy")
        b1 = np.load("model/b1.npy")
        W2 = np.load("model/W2.npy")
        b2 = np.load("model/b2.npy")
        return W1, b1, W2, b2
    
    return init_params()

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.05)

#save_weights(W1, b1, W2, b2)

#test_predictions = make_predictions(X_test, W1, b1, W2, b2)
#print("Accuracy: ", get_accuracy(test_predictions, Y_test))