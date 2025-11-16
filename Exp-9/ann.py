import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
import time

np.random.seed(42)
sns.set()

# Task 1: Data loading & preprocessing

data = load_breast_cancer()
X = data.data              
y = data.target            
print("X shape:", X.shape, "y shape:", y.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Task 2: Utilities (activations, derivatives, losses)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(A):
    # input is activation A = sigmoid(Z)
    return A * (1 - A)

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def compute_bce_loss(Y, Y_hat, eps=1e-15):
    m = Y.shape[1]
    Y_hat = np.clip(Y_hat, eps, 1 - eps)
    loss = - (1.0 / m) * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    return loss

def compute_mse_loss(Y, Y_hat):
    m = Y.shape[1]
    loss = (1.0 / m) * np.sum((Y_hat - Y) ** 2)
    return loss

# Task 3: MyANNClassifier (with NumPy-only)
class MyANNClassifier:
    def __init__(self, layer_dims, learning_rate=0.01, n_iterations=1000, loss='bce', verbose=False, seed=42):
        """
        layer_dims: list like [n_x, hidden1, hidden2, ..., 1]
        """
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1  # number of layers with parameters
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.loss = loss.lower()
        self.verbose = verbose
        self.seed = seed
        self.parameters_ = {}
        self.costs_ = []
        self._initialize_parameters()

    def _initialize_parameters(self):
        np.random.seed(self.seed)
        params = {}
        for l in range(1, len(self.layer_dims)):
            W = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            b = np.zeros((self.layer_dims[l], 1))
            params['W' + str(l)] = W
            params['b' + str(l)] = b
        self.parameters_ = params

    def _forward_propagation(self, X):
        """
        X: shape (n_x, m)
        returns: AL (1,m), cache (list of dicts with Z,A for each layer)
        """
        cache = {}
        A_prev = X
        cache['A0'] = A_prev

        for l in range(1, self.L + 1):
            W = self.parameters_['W' + str(l)]
            b = self.parameters_['b' + str(l)]
            Z = W.dot(A_prev) + b
            if l != self.L:  # hidden layers: ReLU
                A = relu(Z)
            else:             # output layer: Sigmoid
                A = sigmoid(Z)
            cache['Z' + str(l)] = Z
            cache['A' + str(l)] = A
            A_prev = A

        return A_prev, cache  # A_prev is AL

    def _backward_propagation(self, Y, AL, cache):
        """
        Y: (1,m)
        AL: (1,m)
        cache: dict with A0..AL and Z1..ZL
        returns grads: dict with dWl, dbl
        """
        m = Y.shape[1]
        grads = {}

        # dA for output layer
        if self.loss == 'bce':
            # dA = -(Y/AL - (1-Y)/(1-AL))
            dAL = - (np.divide(Y, np.clip(AL,1e-15,1)) - np.divide(1 - Y, np.clip(1 - AL,1e-15,1)))
        elif self.loss == 'mse':
            dAL = 2 * (AL - Y)
        else:
            raise ValueError("loss must be 'bce' or 'mse'")

        # Output layer L
        AL_act = cache['A' + str(self.L)]
        ZL = cache['Z' + str(self.L)]
        A_prev = cache['A' + str(self.L - 1)]

        dZL = dAL * sigmoid_derivative(AL_act)   # (1,m)
        dWL = (1.0 / m) * dZL.dot(A_prev.T)       # (1, size_prev)
        dbL = (1.0 / m) * np.sum(dZL, axis=1, keepdims=True)

        grads['dW' + str(self.L)] = dWL
        grads['db' + str(self.L)] = dbL

        dA_prev = self.parameters_['W' + str(self.L)].T.dot(dZL)  # (size_prev, m)

        # Hidden layers L-1 .. 1
        for l in reversed(range(1, self.L)):
            Z_curr = cache['Z' + str(l)]
            A_prev = cache['A' + str(l - 1)]
            dZ = dA_prev * relu_derivative(Z_curr)
            dW = (1.0 / m) * dZ.dot(A_prev.T)
            db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)

            grads['dW' + str(l)] = dW
            grads['db' + str(l)] = db

            if l > 1:
                dA_prev = self.parameters_['W' + str(l)].T.dot(dZ)

        return grads

    def _update_parameters(self, grads):
        for l in range(1, self.L + 1):
            self.parameters_['W' + str(l)] -= self.learning_rate * grads['dW' + str(l)]
            self.parameters_['b' + str(l)] -= self.learning_rate * grads['db' + str(l)]

    def fit(self, X, y):
        """
        X: (m, n_features)
        y: (m,) or (m,1)
        """
        X = np.array(X)
        y = np.array(y).reshape(-1)
        m = X.shape[0]
        # transpose for internal shape (n_x, m)
        X_cols = X.T                  # (n_x, m)
        Y_cols = y.reshape(1, m)      # (1, m)

        self.costs_ = []

        for i in range(self.n_iterations):
            AL, cache = self._forward_propagation(X_cols)
            # compute loss
            if self.loss == 'bce':
                cost = compute_bce_loss(Y_cols, AL)
            else:
                cost = compute_mse_loss(Y_cols, AL)

            grads = self._backward_propagation(Y_cols, AL, cache)
            self._update_parameters(grads)

            self.costs_.append(cost)

            if self.verbose and (i % max(1, self.n_iterations // 10) == 0):
                print(f"Iteration {i}/{self.n_iterations} - loss: {cost:.6f}")

        return self

    def predict_proba(self, X):
        X_cols = np.array(X).T
        AL, _ = self._forward_propagation(X_cols)
        return AL  # shape (1, m)

    def predict(self, X):
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)
        return preds.reshape(-1)

# Task 4: Training & experimentation


def train_and_report(model, X_train, y_train, X_val, y_val, name="Model"):
    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()
    preds = model.predict(X_val)
    print(f"\n{name} - time: {t1 - t0:.2f}s")
    print(classification_report(y_val, preds, digits=4))
    return model, preds

# Model 1: BCE loss, 1 hidden layer e.g., [30,10,1]
n_x = X_train_scaled.shape[1]
layer_dims_1 = [n_x, 10, 1]
model_bce = MyANNClassifier(layer_dims=layer_dims_1, learning_rate=0.001, n_iterations=5000, loss='bce', verbose=True, seed=1)
model_bce, preds_bce = train_and_report(model_bce, X_train_scaled, y_train, X_val_scaled, y_val, name="MyANN (BCE, 1 hidden)")

# Model 2: MSE loss, same architecture
model_mse = MyANNClassifier(layer_dims=layer_dims_1, learning_rate=0.001, n_iterations=5000, loss='mse', verbose=True, seed=1)
model_mse, preds_mse = train_and_report(model_mse, X_train_scaled, y_train, X_val_scaled, y_val, name="MyANN (MSE, 1 hidden)")

# Model 3: Deeper architecture [30, 10, 5, 1]
layer_dims_3 = [n_x, 10, 5, 1]
model_deep = MyANNClassifier(layer_dims=layer_dims_3, learning_rate=0.001, n_iterations=5000, loss='bce', verbose=True, seed=1)
model_deep, preds_deep = train_and_report(model_deep, X_train_scaled, y_train, X_val_scaled, y_val, name="MyANN (BCE, 2 hidden)")


# Task 5: Compare with sklearn MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam',
                    learning_rate_init=0.001, max_iter=1000, random_state=42)
t0 = time.time()
mlp.fit(X_train_scaled, y_train)
t1 = time.time()
preds_mlp = mlp.predict(X_val_scaled)
print("\nSklearn MLPClassifier - time: {:.2f}s".format(t1 - t0))
print(classification_report(y_val, preds_mlp, digits=4))


# Loss curves for Model1 (BCE) vs Model2 (MSE)

plt.figure(figsize=(8,5))
iters = range(len(model_bce.costs_))
plt.plot(iters, model_bce.costs_, label='BCE (1 hidden)')
plt.plot(iters, model_mse.costs_, label='MSE (1 hidden)')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss curves")
plt.legend()
plt.show()

# Summary table (Precision/Recall/F1 for class 1)

from sklearn.metrics import precision_recall_fscore_support

def prf(y_true, y_pred):
    p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    # class 1 metrics:
    return {'precision_class1': p[1], 'recall_class1': r[1], 'f1_class1': f[1]}

summary = {
    'MyANN_BCE_1hidden': prf(y_val, preds_bce),
    'MyANN_MSE_1hidden': prf(y_val, preds_mse),
    'MyANN_BCE_2hidden': prf(y_val, preds_deep),
    'Sklearn_MLP': prf(y_val, preds_mlp)
}
import pandas as pd
summary_df = pd.DataFrame(summary).T
print("\nSummary (class 1 metrics):\n", summary_df)



