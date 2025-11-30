import time
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
from datasets import load_mnist_data_pennylane

# Load your MNIST data (64 features)
x_train, x_test, y_train, y_test = load_mnist_data_pennylane()

# Convert to numpy and one-hot encode labels
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

def one_hot_encode(y, num_classes=10):
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded

y_train_encoded = one_hot_encode(y_train_np)
y_test_encoded = one_hot_encode(y_test_np)

# Normalize features to [0, π]
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

x_min = x_train.min()
x_max = x_train.max()
x_train = (x_train - x_min) / (x_max - x_min + 1e-8) * np.pi
x_test = (x_test - x_min) / (x_max - x_min + 1e-8) * np.pi

# 64 features = 2^6, so we need 6 qubits
num_qubits = 6
num_quantum_outputs = 6  # Number of qubits we measure
num_classes = 10

dev = qml.device('default.qubit', wires=num_qubits)

def layer(layer_weights):
    k = layer_weights.shape[0]
    for wire in range(k):
        qml.Rot(*layer_weights[wire], wires=wire)
    
    for wire in range(k):
        qml.CNOT(wires=[wire, (wire + 1) % k])

@qml.qnode(dev)
def circuit(weights, x):
    qml.AmplitudeEmbedding(x, wires=range(num_qubits), normalize=True)
    
    for layer_weights in weights:
        layer(layer_weights)
    
    # Measure 4 qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(num_quantum_outputs)]

def variational_classifier(weights, bias, W_out, x):
    """
    Map quantum outputs (4 values) to class scores (10 values)
    """
    quantum_output = np.array(circuit(weights, x))  # Shape: (4,)
    # Linear transformation: (4,) @ (4, 10) + (10,) = (10,)
    class_scores = quantum_output @ W_out + bias
    return class_scores

def softmax(x):
    """Numerical stable softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def cross_entropy_loss(labels, predictions):
    """Cross-entropy loss for multi-class classification"""
    probs = np.array([softmax(pred) for pred in predictions])
    probs = np.clip(probs, 1e-10, 1.0)
    loss = -np.mean(np.sum(labels * np.log(probs), axis=1))
    return loss

def accuracy(labels, predictions):
    """Multi-class accuracy"""
    pred_classes = np.array([np.argmax(softmax(pred)) for pred in predictions])
    true_classes = np.argmax(labels, axis=1)
    acc = np.mean(pred_classes == true_classes)
    return acc

def cost(weights, bias, W_out, X, Y):
    predictions = [variational_classifier(weights, bias, W_out, x) for x in X]
    return cross_entropy_loss(Y, predictions)

# Initialize parameters
np.random.seed(42)
num_layers = 6
weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3)
weights_init = np.array(weights_init, requires_grad=True)

bias_init = np.zeros(num_classes, requires_grad=True)  # 10 class biases
W_out_init = np.array(0.01 * np.random.randn(num_quantum_outputs, num_classes), requires_grad=True)  # 6x10 weight matrix

opt = AdamOptimizer(stepsize=0.01)
batch_size = 20

weights = weights_init
bias = bias_init
W_out = W_out_init
indexs = np.arange(len(x_train))

for epoch in range(10):
    print(f"\nTraining Epoch {epoch+1}:")
    np.random.shuffle(indexs)
    
    for batch in range(int(np.ceil(len(indexs)/batch_size))):
        start = time.time()
        
        # Update weights
        batch_index = indexs[batch*batch_size:min((batch+1)*batch_size, len(indexs))]
        X_batch = x_train[batch_index]
        Y_batch = y_train_encoded[batch_index]
        
        weights, bias, W_out = opt.step(cost, weights, bias, W_out, X=X_batch, Y=Y_batch)
        
        # Evaluate every 5 batches
        if batch % 5 == 0:
            sample_indices = np.random.choice(len(x_train), size=100, replace=False)
            predictions = [variational_classifier(weights, bias, W_out, x) for x in x_train[sample_indices]]
            current_cost = cross_entropy_loss(y_train_encoded[sample_indices], predictions)
            acc = accuracy(y_train_encoded[sample_indices], predictions)
            
            print(f"\tBatch {batch+1:3d} | Cost: {current_cost:0.7f} | Train Acc: {acc:0.4f} | Time: {time.time()-start:.2f}s")
    
    # Test accuracy at end of epoch
    predictions_test = [variational_classifier(weights, bias, W_out, x) for x in x_test]
    acc = accuracy(y_test_encoded, predictions_test)
    print(f"Epoch {epoch+1} Test Accuracy: {acc*100:0.2f}%")
