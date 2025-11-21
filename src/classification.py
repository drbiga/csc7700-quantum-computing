import time

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from src.datasets import load_cancer_train_test

num_qubits = 5
dev = qml.device('default.qubit', wires=num_qubits)
x_train, x_test, y_train, y_test = load_cancer_train_test()
y_train = y_train.apply(lambda x: x * 2 - 1).to_numpy()
y_test = y_test.apply(lambda x: x * 2 - 1).to_numpy()
x_train = x_train.to_numpy()
x_min = x_train.min(axis=0)
x_max = x_train.max(axis=0)
x_train = (x_train - x_min)/ (x_max-x_min + 1e-8) * np.pi
x_train = np.pad(x_train, ((0, 0), (0, 2)), mode='constant', constant_values=0)
x_test = x_test.to_numpy()
x_test = (x_test - x_min)/ (x_max-x_min + 1e-8) * np.pi
x_test = np.pad(x_test, ((0, 0), (0, 2)), mode='constant', constant_values=0)

def layer(layer_weights):
    k = layer_weights.shape[0]
    for wire in range(k):
        qml.Rot(*layer_weights[wire], wires=wire)

    for wire in range(k):
        qml.CNOT([wire,(wire+1)%k])

@qml.qnode(dev)
def circuit(weights, x):
    qml.AmplitudeEmbedding(x, wires=range(num_qubits), normalize=True)

    for layer_weights in weights:
        layer(layer_weights)

    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return np.mean((labels - qml.math.stack(predictions)) ** 2)

def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc

def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)

np.random.seed(42)
num_layers = 3
weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)
opt = NesterovMomentumOptimizer(stepsize=0.05)
batch_size = 30

weights = weights_init
bias = bias_init
indexs = np.arange(len(x_train))
for epoch in range(10):
    print(f"Training Epoch {epoch+1}:")
    np.random.shuffle(indexs)
    for batch in range(int(np.ceil(len(indexs)/batch_size))):
        start = time.time()
        # Update the weights by one optimizer step, using only a limited batch of data
        batch_index = indexs[batch*batch_size:min((batch+1)*batch_size,len(indexs))]
        X_batch = x_train[batch_index]
        Y_batch = y_train[batch_index]
        weights, bias = opt.step(cost, weights, bias, X=X_batch, Y=Y_batch)

        # Compute accuracy
        predictions = [np.sign(variational_classifier(weights, bias, x)) for x in x_train]

        current_cost = cost(weights, bias, x_train, y_train)
        acc = accuracy(y_train, predictions)

        print(f"\tIter: {batch+1:3d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")

    predictions_test = [np.sign(variational_classifier(weights, bias, x)) for x in x_test]
    acc = accuracy(y_test, predictions_test)
    print(f"Test Accuracy: {acc*100:0.4f}%")