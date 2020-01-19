import numpy as np
import time


def ReLU(z):
    return np.maximum(z, np.zeros(np.shape(z)))

def softmax(vector):
    return np.exp(vector) / np.sum(np.exp(vector))

def NLL(x):
    if x > 0 :
        return -np.log(x)
    else:
        return 0

def d_ReLU(drelu_x):
    drelu_x[drelu_x <= 0] = 0
    drelu_x[drelu_x > 0] = 1
    return drelu_x

class NeuralNet(object):

    def __init__(self, input_size, hidden_size, output_size, X_train, Y_train,
                 netSize , lr):
        self.params = {}
        self.netSize = netSize
        self.hiddenSize = hidden_size
        self.x = X_train
        self.y = Y_train
        self.output_size = output_size
        self.input_size = input_size
        self.lr = lr

        self.params['W1'] = np.random.randn(hidden_size,input_size)
        self.params['b1'] = np.random.randn(hidden_size)

        self.params['W2'] = np.random.randn(output_size,hidden_size)
        self.params['b2'] = np.random.randn(output_size)

    def forwardComp(self, cur_example, cur_label):
        Z1 = self.params['W1'].dot(cur_example) + self.params['b1']
        H1 = ReLU(Z1)
        H1 = H1/np.max(H1) # normalization
        Z2 = self.params['W2'].dot(H1) + self.params['b2']
        # should be ReLU again?!
        H2 = softmax(Z2)
        loss = NLL(H2[int(cur_label)])
        results = {"Z1": Z1, "H1": H1, "Z2": Z2, "H2": H2}

        return results

    def backprop(self, cur_example, y, forward_results):
        y_hat = forward_results["H2"]
        #set y as a vector, while the correct class index is 1 ( [0...1..0] )
        correct_class = int(y)
        y= np.zeros(self.output_size)
        y[correct_class] = 1

        ds_db2 = np.array(y_hat-y) # -(y - y_hat)
        ds_dw2 = ds_db2.reshape(-1,1).dot(forward_results["H1"].reshape(1,self.hiddenSize)) # -(y - y_hat) * h1

        ds_dh1 = self.params["W2"].T.dot(ds_db2) # (y - y_hat) * w2
        ds_dz1 = np.multiply(ds_dh1, d_ReLU(forward_results["Z1"]))  # -(y - y_hat)* w2 * relu'(z1)
        ds_dw1 = np.dot(np.array(ds_dz1).reshape(self.hiddenSize,1), np.array(cur_example).reshape(1,self.input_size))  # -(y - y_hat)* w2 * relu'(z1) * x
        ds_db1 = ds_dz1  # (-y - y_hat)* w2 * relu'(z1)

        self.params["W1"] = self.params["W1"] - self.lr * ds_dw1
        self.params["b1"] = self.params["b1"] - self.lr * ds_db1

        self.params["W2"] = self.params["W2"] - self.lr * ds_dw2
        self.params["b2"] = self.params["b2"] - self.lr * ds_db2

        # print("--params--")
        # print("-W1 : ")
        # print(self.params["W1"])
        # print("-b1 : ")
        # print(self.params["b1"])
        # print("-W2 : ")
        # print(self.params["W2"])
        # print("-b2 : ")
        # print(self.params["b2"])

    def train(self, epochs, train_x, train_y, ):
        for i in range(epochs):
            print("--- Epoch #{0} ---".format(i))
            loop_start = time.time()
            for x, y in zip(train_x, train_y):
                forward_results = self.forwardComp(x, y)
                self.backprop(x, y, forward_results)
                # time.sleep(0.3)
            loop_end = time.time()
            print("Epoch ended . Elapsed time is {1:.2f} sec ".format(i,loop_end-loop_start))
