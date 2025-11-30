from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import Estimator

import torch
from torch import no_grad
import torch.optim as optim
from torch.nn import Module, Linear, CrossEntropyLoss  
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from datasets import load_cancer_train_test
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from qiskit.quantum_info import SparsePauliOp


NUM_QUBITS = 3
NUM_COMPONENTS = NUM_QUBITS


def load_and_preprocess_data():
    """Load and preprocess breast cancer dataset"""
    X_train, X_test, y_train, y_test = load_cancer_train_test()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=NUM_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    X_min = X_train_pca.min()
    X_max = X_train_pca.max()
    
    X_train_pca = (X_train_pca - X_min) / (X_max - X_min + 1e-8) * (2 * np.pi)
    X_test_pca = (X_test_pca - X_min) / (X_max - X_min + 1e-8) * (2 * np.pi)

    X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define and create QNN
def create_qnn(num_qubits=3):
    feature_map = ZZFeatureMap(num_qubits, reps=2)
    ansatz = RealAmplitudes(num_qubits, reps=3)

    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    estimator = Estimator()
    
    observables = [
        SparsePauliOp("ZII"),  # Measure qubit 0
        SparsePauliOp("IZI"),  # Measure qubit 1
        SparsePauliOp("IIZ"),  # Measure qubit 2
        SparsePauliOp("ZZI"),  # 2-qubit correlation
        SparsePauliOp("IZZ"),  # 2-qubit correlation
        SparsePauliOp("ZIZ"),  # 2-qubit correlation
    ]

    qnn = EstimatorQNN(
        circuit=qc,
        estimator=estimator,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        observables=observables,
        input_gradients=True
    )

    return qnn


qnn4 = create_qnn(num_qubits=NUM_QUBITS)


# Define PyTorch model with QNN
class Net(Module):
    def __init__(self, qnn):
        super().__init__()
        self.qnn = TorchConnector(qnn)
        self.post_qnn = Linear(6, 2)

    def forward(self, x):
        x = self.qnn(x)
        x = self.post_qnn(x)
        return x


model4 = Net(qnn4)


def plot_loss_convergence(loss_list, acc_list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(loss_list)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    
    ax2.plot(acc_list)
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_progress.png")
    plt.show()


optimizer = optim.Adam(model4.parameters(), lr=0.05)
loss_func = CrossEntropyLoss()


# Training loop
epochs = 10
loss_list = []
acc_list = []
model4.train()

print("\nStarting training...")
print("=" * 60)

for epoch in range(epochs):
    total_loss = []
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        output = model4(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss.append(loss.item())
        
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    avg_loss = sum(total_loss) / len(total_loss)
    train_acc = 100.0 * correct / total
    loss_list.append(avg_loss)
    acc_list.append(train_acc)
    
    print(f"Epoch [{epoch+1:2d}/{epochs}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")


plot_loss_convergence(loss_list, acc_list)


# Evaluation
model4.eval()
with no_grad():
    correct = 0
    total_loss = []
    all_preds = []
    all_targets = []

    for batch_idx, (data, target) in enumerate(test_loader):
        output = model4(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        all_preds.extend(pred.squeeze().tolist())
        all_targets.extend(target.tolist())

        loss = loss_func(output, target)
        total_loss.append(loss.item())

    test_acc = 100.0 * correct / len(test_loader.dataset)
    test_loss = sum(total_loss) / len(total_loss)
    
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.1f}%")
    
    
# Classical Neural Network for Comparison
class ClassicalNet(Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.fc1 = Linear(input_dim, 32)
        self.fc2 = Linear(32, 16)
        self.fc3 = Linear(16, 8)
        self.fc4 = Linear(8, 2)
        self.dropout = torch.nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Create and train classical model
classical_model = ClassicalNet(input_dim=NUM_COMPONENTS)
classical_optimizer = optim.Adam(classical_model.parameters(), lr=0.001)
loss_func = CrossEntropyLoss()

print("\n" + "=" * 60)
print("Training Classical Neural Network")
print("=" * 60)

classical_loss_list = []
classical_acc_list = []
classical_model.train()

for epoch in range(epochs):
    total_loss = []
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        classical_optimizer.zero_grad(set_to_none=True)
        output = classical_model(data)
        loss = loss_func(output, target)
        loss.backward()
        classical_optimizer.step()
        
        total_loss.append(loss.item())
        
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    avg_loss = sum(total_loss) / len(total_loss)
    train_acc = 100.0 * correct / total
    classical_loss_list.append(avg_loss)
    classical_acc_list.append(train_acc)
    
    print(f"Epoch [{epoch+1:2d}/{epochs}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

# Evaluate classical model
classical_model.eval()
with no_grad():
    correct = 0
    total_loss = []
    
    for batch_idx, (data, target) in enumerate(test_loader):
        output = classical_model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss = loss_func(output, target)
        total_loss.append(loss.item())
    
    classical_test_acc = 100.0 * correct / len(test_loader.dataset)
    classical_test_loss = sum(total_loss) / len(total_loss)
    
    print(f"\nClassical Model Results:")
    print(f"Test Loss:     {classical_test_loss:.4f}")
    print(f"Test Accuracy: {classical_test_acc:.1f}%")
