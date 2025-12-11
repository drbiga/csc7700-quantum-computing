from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

import torch
from torch import no_grad
import torch.optim as optim
from torch.nn import Module, Linear, NLLLoss
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_mnist_data
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_score, recall_score, f1_score


NUM_QUBITS = 3
NUM_COMPONENTS = NUM_QUBITS

def preprocessing_data(
    x: pd.DataFrame, y: pd.Series
):
    X = x.astype(np.float32).to_numpy()
    y = y.astype(np.int64).to_numpy()
    
    X /= 16.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    x_train_flat = X_train.reshape(X_train.shape[0], -1)
    x_test_flat = X_test.reshape(X_test.shape[0], -1)

    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(x_train_flat)
    X_test_flat = scaler.transform(x_test_flat)

    pca = PCA(n_components=NUM_COMPONENTS)
    X_train = pca.fit_transform(X_train_flat)
    X_test = pca.transform(X_test_flat)

    X_min = X_train.min()
    X_max = X_train.max()
    
    X_train = (X_train - X_min) / (X_max - X_min + 1e-8) * (2 * np.pi)
    X_test = (X_test - X_min) / (X_max - X_min + 1e-8) * (2 * np.pi)
        
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    return X_train, X_test, y_train, y_test


x, y = load_mnist_data()
X_train, X_test, y_train, y_test = preprocessing_data(x, y)

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


# Define torch NN module
class Net(Module):
    def __init__(self, qnn, input_dim=4, num_qubits=4):
        super().__init__()
        
        self.fc1 = Linear(input_dim, 16)
        self.fc2 = Linear(16, num_qubits)
        
        self.qnn = TorchConnector(qnn)
        
        self.fc3 = Linear(6, 32)
        self.fc4 = Linear(32, 10)

        self.dropout = torch.nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = self.qnn(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return F.log_softmax(x, dim=1)


model4 = Net(qnn4, input_dim=NUM_COMPONENTS, num_qubits=NUM_QUBITS)


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


def save_model(model):
    torch.save(model.state_dict(), "model4.pt")
    print("Model saved to model4.pt")


# Define model, optimizer, and loss function
optimizer = optim.Adam(model4.parameters(), lr=0.005)
loss_func = NLLLoss()


# Start training
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

print("=" * 60)
print("Training completed!\n")


plot_loss_convergence(loss_list, acc_list)
save_model(model4)

# Evaluation
print("Evaluating on test set...")
model5 = Net(qnn4, input_dim=NUM_COMPONENTS, num_qubits=NUM_QUBITS)
state = torch.load("model4.pt", weights_only=True)
model5.load_state_dict(state)

model5.eval()
with no_grad():
    correct = 0
    total_loss = []
    all_preds = []
    all_targets = []

    for batch_idx, (data, target) in enumerate(test_loader):
        output = model5(data)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        all_preds.extend(pred.squeeze().tolist())
        all_targets.extend(target.tolist())

        loss = loss_func(output, target)
        total_loss.append(loss.item())

    test_acc = 100.0 * correct / len(test_loader.dataset)
    test_loss = sum(total_loss) / len(total_loss)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.1f}%")
    print("=" * 60)
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    for digit in range(10):
        mask = all_targets == digit
        if mask.sum() > 0:
            digit_acc = 100.0 * (all_preds[mask] == digit).sum() / mask.sum()
            print(f"  Digit {digit}: {digit_acc:.1f}% ({mask.sum()} samples)")

    # Overall Precision, Recall, F1
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')

    print("\nOverall Evaluation Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

