from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import numpy as np
import time

from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.optimizers import SPSA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import StatevectorSampler as Sampler

from datasets import load_cancer_train_test_qiskit

def preprocessing_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=4)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca


def evaluate_classical(X_train, X_test, y_train, y_test):
    X_train_pca, X_test_pca = preprocessing_data(X_train, X_test)

    model = SVC(kernel='rbf')
    model.fit(X_train_pca, y_train)
    return model.score(X_test_pca, y_test)


def draw_circuits_once():
    
    num_features = 4
    
    feature_map = ZZFeatureMap(num_features, reps=1, entanglement='linear')
    feature_map.decompose().draw('mpl')
    plt.title('Feature Map (ZZFeatureMap)')
    plt.savefig('feature_map.png', dpi=150, bbox_inches='tight')
    plt.show()

    ansatz = RealAmplitudes(num_features, reps=2, entanglement='linear')
    ansatz.decompose().draw('mpl')
    plt.title('Ansatz (RealAmplitudes)')
    plt.savefig('ansatz.png', dpi=150, bbox_inches='tight')
    plt.show()


def evaluate_quantum(X_train, X_test, y_train, y_test):
    X_train_pca, X_test_pca = preprocessing_data(X_train, X_test)

    num_features = X_train_pca.shape[1]
    num_qubits = num_features

    feature_map = ZZFeatureMap(num_features, reps=1, entanglement='linear')
    feature_map.decompose().draw('mpl')

    ansatz = RealAmplitudes(num_qubits, reps=2, entanglement='linear')
    ansatz.decompose().draw('mpl')

    sampler = Sampler()

    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=SPSA(maxiter=100),
        callback=None
    )

    vqc.fit(X_train_pca, y_train)
    return vqc.score(X_test_pca, y_test)


def evaluate_both():
    X_df, y_df = load_cancer_train_test_qiskit()
    X = X_df.values
    y = y_df.values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    classical_acc = []
    quantum_acc = []
    
    classical_times = []
    quantum_times = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/5")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Classical evaluation with timing
        start = time.time()
        acc_classical = evaluate_classical(X_train, X_test, y_train, y_test)
        classical_time = time.time() - start
        classical_acc.append(acc_classical)
        classical_times.append(classical_time)
        print(f"Classical Accuracy: {acc_classical:.4f} (Time: {classical_time:.2f}s)")

        # Quantum evaluation with timing
        start = time.time()
        acc_quantum = evaluate_quantum(X_train, X_test, y_train, y_test)
        quantum_time = time.time() - start
        quantum_acc.append(acc_quantum)
        quantum_times.append(quantum_time)
        print(f"Quantum Accuracy:   {acc_quantum:.4f} (Time: {quantum_time:.2f}s)")

    print("FINAL RESULTS")
    
    print(f"\nClassical SVM:")
    print(f"  Mean Accuracy:     {np.mean(classical_acc):.4f}")
    print(f"  Min/Max Accuracy:  {np.min(classical_acc):.4f} / {np.max(classical_acc):.4f}")
    print(f"  Avg Time per Fold: {np.mean(classical_times):.2f}s")
    
    print(f"\nQuantum VQC:")
    print(f"  Mean Accuracy:     {np.mean(quantum_acc):.4f}")
    print(f"  Min/Max Accuracy:  {np.min(quantum_acc):.4f} / {np.max(quantum_acc):.4f}")
    print(f"  Avg Time per Fold: {np.mean(quantum_times):.2f}s")


if __name__ == "__main__":
    draw_circuits_once()
    evaluate_both()
