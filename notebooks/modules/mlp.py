import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)

def dsigmoid(grad_a, act):
    return np.multiply(grad_a, act - np.square(act))


def dtanh(grad_a, act):
    return np.multiply(grad_a, 1 - np.square(act))


def softmax(x):
    eps = 1e-8
    out = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return out / (np.sum(out, axis=1).reshape(-1, 1) + eps)


def linear(x):
    return x


def cross_entropy(pred, y):
    return -(np.multiply(y, np.log(pred + 1e-4))).mean()


def squared_error(pred, y):
    return np.square(pred - y).mean() / 2


class MLP(object):

    def __init__(self, act_func, der_act_func, layers, epochs=20, regression=False, learning_rate=0.01, lmbda=1e-2):
        self.reg = 2 
        self.lmbda = lmbda
        self.gamma = 0.9
        self.eps = 1e-8
        self.epochs, self.batch_size = epochs, 32
        self.learning_rate = learning_rate
        self.layer_num = len(layers) - 1
        self.n_labels = layers[-1]
        self.regression = regression
        self.output = linear if self.regression else softmax
        self.loss = squared_error if self.regression else cross_entropy

        self.afunc = act_func
        self.dact = der_act_func
        self.optimize = self.sgd

        self.w, self.b = [np.empty] * \
            self.layer_num, [np.empty] * self.layer_num
        self.mom_w, self.cache_w = [np.empty] * \
            self.layer_num, [np.empty] * self.layer_num
        self.mom_b, self.cache_b = [np.empty] * \
            self.layer_num, [np.empty] * self.layer_num

        for i in range(self.layer_num):
            self.w[i] = np.random.randn(layers[i], layers[i + 1])
            self.b[i] = np.random.randn(1, layers[i + 1])
            self.mom_w[i] = np.zeros_like(self.w[i])
            self.cache_w[i] = np.zeros_like(self.w[i])
            self.mom_b[i] = np.zeros_like(self.b[i])
            self.cache_b[i] = np.zeros_like(self.b[i])

    def sgd(self, grad_w, grad_b):
        alpha = self.learning_rate / self.batch_size
        for i in range(self.layer_num):
            self.w[i] -= alpha * grad_w[i]
            self.b[i] -= alpha * grad_b[i]

    def regularization(self):
        if(self.reg == 0):
            return
        alpha = self.learning_rate * self.lmbda
        for i in range(self.layer_num):
            if(self.reg == 1):
                self.w[i] -= alpha * np.sign(self.w[i])
            elif(self.reg == 2):
                self.w[i] -= alpha * self.w[i]

    def predict(self, x):
        act = x
        for i in range(self.layer_num - 1):
            act = self.afunc(act.dot(self.w[i]) + self.b[i])
        return self.output(act.dot(self.w[self.layer_num - 1]) + self.b[self.layer_num - 1])

    def fit(self, x, labels):
        train_num = x.shape[0]
        l_num = self.layer_num
        bvec = np.ones((1, self.batch_size))

        if self.regression:
            y = labels
        else:
            y = np.zeros((train_num, self.n_labels))
            y[np.arange(train_num), labels] = 1

        for epoch in range(self.epochs):
            p = np.random.permutation(train_num // self.batch_size * self.batch_size).reshape(-1, self.batch_size)
            for b_idx in range(p.shape[0]):
                act = [np.empty] * (l_num + 1)
                act[0] = x[p[b_idx, :]]
                for i in range(1, l_num):
                    act[i] = self.afunc(
                        act[i - 1].dot(self.w[i - 1]) + self.b[i - 1])
                act[l_num] = self.output(
                    act[l_num - 1].dot(self.w[l_num - 1]) + self.b[l_num - 1])
                grad_a, grad_w, grad_b = [
                    np.empty] * (l_num + 1), [np.empty] * l_num, [np.empty] * l_num
                grad_a[l_num] = act[l_num] - y[p[b_idx, :]]
                grad_w[l_num - 1] = act[l_num - 1].T.dot(grad_a[l_num])
                grad_b[l_num - 1] = bvec.dot(grad_a[l_num])

                for i in reversed(range(1, l_num)):
                    grad_a[i] = grad_a[i + 1].dot(self.w[i].T)
                    grad_a[i] = self.dact(grad_a[i], act[i])
                    grad_w[i - 1] = act[i - 1].T.dot(grad_a[i])
                    grad_b[i - 1] = bvec.dot(grad_a[i])
                self.regularization()
                self.optimize(grad_w, grad_b)