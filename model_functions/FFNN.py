import numpy as np

def relu(x): return np.where(x > 0, x, 0.01 * x)
def relu_deriv(x): return np.where(x > 0, 1, 0.01)

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def softplus_deriv(x):
    return 1.0 / (1.0 + np.exp(-x))

def poisson_loss(y_true, y_pred, eps=1e-8):
    return np.mean(y_pred - y_true * np.log(y_pred + eps))

def poisson_grad(y_true, y_pred, eps=1e-8):
    return (1.0 - y_true / (y_pred + eps)) / y_true.size


class FFNN:
    def __init__(self, input_size, hidden_sizes, output_size=1, lr=0.0005, seed=31):
        np.random.seed(seed)
        self.lr = lr
        self.weights = []
        self.biases = []

        prev = input_size
        # hidden layers
        for h in hidden_sizes:
            W = np.random.randn(prev, h) * np.sqrt(2.0 / prev)
            b = np.zeros((1, h))
            self.weights.append(W)
            self.biases.append(b)
            prev = h
        # output layer
        W = np.random.randn(prev, output_size) * np.sqrt(2.0 / prev)
        b = np.zeros((1, output_size))
        self.weights.append(W)
        self.biases.append(b)

    def forward(self, X):
        activations = [X]
        preacts = []

        # hidden layers: LeakyReLU
        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            preacts.append(z)
            a = relu(z)
            activations.append(a)

        # output layer: linear -> softplus
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        preacts.append(z)
        a = softplus(z)
        activations.append(a)

        return activations, preacts

    def backward(self, activations, preacts, y_true):
        grads_w, grads_b = [], []
        y_pred = activations[-1]

        # dL/dλ̂
        dY = poisson_grad(y_true, y_pred)
        # dL/dz_L = dL/dλ̂ * dλ̂/dz_L
        dZ_L = dY * softplus_deriv(preacts[-1])

        # output layer gradients
        grad_w = activations[-2].T @ dZ_L
        grad_b = np.sum(dZ_L, axis=0, keepdims=True)
        grads_w.insert(0, grad_w)
        grads_b.insert(0, grad_b)

        # backprop through hidden layers
        dA = dZ_L @ self.weights[-1].T
        for i in reversed(range(len(self.weights) - 1)):
            dZ = dA * relu_deriv(preacts[i])
            grad_w = activations[i].T @ dZ
            grad_b = np.sum(dZ, axis=0, keepdims=True)
            grads_w.insert(0, grad_w)
            grads_b.insert(0, grad_b)
            if i != 0:
                dA = dZ @ self.weights[i].T

        # gradient clipping
        for g in grads_w:
            np.clip(g, -1.0, 1.0, out=g)
        for g in grads_b:
            np.clip(g, -1.0, 1.0, out=g)

        # update
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i]  -= self.lr * grads_b[i]

        return poisson_loss(y_true, y_pred)

    def fit(self, X, y, epochs=20, batch_size=1024):
        N = X.shape[0]
        for epoch in range(epochs):
            permutation = np.random.permutation(N)
            X = X[permutation]
            Y = y[permutation]

            for i in range(0, N, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = Y[i:i+batch_size]    # <-- fixed here

                activations, preacts = self.forward(X_batch)
                loss = self.backward(activations, preacts, y_batch)

            print(f"Epoch {epoch}, Loss={loss}")

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]


