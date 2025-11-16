This experiment implements a feed-forward neural network using only NumPy, covering the full workflow from parameter initialization to prediction. The model learns using forward propagation, loss computation, and backpropagation, followed by gradient descent updates.

A neural network consists of layers of neurons, where each neuron performs a weighted sum of its inputs and applies a non-linear activation function. ReLU is used in hidden layers for efficient gradient flow, while Sigmoid is applied in the output layer for binary classification.

During forward propagation, the network computes predictions. A loss function (Binary Cross-Entropy or Mean Squared Error) measures the error between predictions and true labels. Backpropagation then computes gradients of this loss with respect to every weight and bias using the chain rule. Gradient descent updates parameters in the direction that reduces the loss.

The model is trained on the Breast Cancer Wisconsin dataset and compared with scikit-learn’s MLPClassifier, which uses more advanced optimizers like Adam. This experiment helps understand how neural networks work internally and how modern frameworks optimize them.
