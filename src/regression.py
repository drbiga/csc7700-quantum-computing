import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
import jax
from jax import numpy as jnp
import optax

import matplotlib.pyplot as plt
import seaborn as sns


from datasets import load_housing_train_test


@dataclass
class RegressionResults:
    score_mean: float
    score_std: float


def evaluate_classical() -> RegressionResults:
    X_train, X_test, y_train, y_test = load_housing_train_test()
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    print("Evaluation of classical model - R2 =", r2)


def evaluate_quantum():
    X_train, X_test, y_train, y_test = load_housing_train_test()

    med_inc_scaler = MinMaxScaler((0, 2 * np.pi))
    scaled_X_train = med_inc_scaler.fit_transform(X_train)
    scaled_X_test = med_inc_scaler.transform(X_test)

    house_val_scaler = MinMaxScaler((-1, 1))
    scaled_y_train = house_val_scaler.fit_transform(pd.DataFrame(y_train)).reshape(
        -1, 1
    )
    scaled_y_test = house_val_scaler.transform(pd.DataFrame(y_test)).reshape(-1, 1)

    pnp.random.seed(42)

    dev = qml.device("default.qubit", wires=2)

    def S(x):
        qml.AngleEmbedding(x, wires=[0, 1], rotation="Z")

    def W(params):
        qml.StronglyEntanglingLayers(params, wires=[0, 1])

    @qml.qnode(dev, interface="jax")
    def quantum_neural_network(params, x):
        layers = len(params[:, 0, 0]) - 1
        n_wires = len(params[0, :, 0])
        n_params_rot = len(params[0, 0, :])
        for i in range(layers):
            W(params[i, :, :].reshape(1, n_wires, n_params_rot))
            S(x)
        W(params[-1, :, :].reshape(1, n_wires, n_params_rot))

        return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

    @jax.jit
    def mse(params, x, targets):
        # We compute the mean square error between the target function and the quantum circuit to quantify the quality of our estimator
        return (quantum_neural_network(params, x) - jnp.array(targets)) ** 2

    opt = optax.adam(learning_rate=0.05)
    max_steps = 300

    @jax.jit
    def update_step_jit(i, args):
        # We loop over this function to optimize the trainable parameters
        params, opt_state, data, targets, print_training = args
        loss_val, grads = jax.value_and_grad(loss_fn)(params, data, targets)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        def print_fn():
            jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)

        # if print_training=True, print the loss every 50 steps
        jax.lax.cond((jnp.mod(i, 50) == 0) & print_training, print_fn, lambda: None)
        return (params, opt_state, data, targets, print_training)

    @jax.jit
    def optimization_jit(params, data, targets, print_training=False):
        opt_state = opt.init(params)
        args = (params, opt_state, jnp.asarray(data), targets, print_training)
        # We loop over update_step_jit max_steps iterations to optimize the parameters
        (params, opt_state, _, _, _) = jax.lax.fori_loop(
            0, max_steps + 1, update_step_jit, args
        )
        return params

    @jax.jit
    def loss_fn(params, x, targets):
        # We define the loss function to feed our optimizer
        mse_pred = jax.vmap(mse, in_axes=(None, 0, 0))(params, x, targets)
        loss = jnp.mean(mse_pred)
        return loss

    wires = 2
    layers = 10
    params_shape = qml.StronglyEntanglingLayers.shape(
        n_layers=layers + 1, n_wires=wires
    )
    params = pnp.random.default_rng().random(size=params_shape)
    best_params = optimization_jit(
        params,
        scaled_X_train.reshape(-1, 1),
        jnp.array(scaled_y_train.reshape(-1, 1)),
        print_training=True,
    )

    y_hat_test = jax.vmap(quantum_neural_network, in_axes=(None, 0))(
        best_params, scaled_X_test
    )

    sns.scatterplot(
        x=scaled_X_test.reshape(-1), y=scaled_y_test.reshape(-1), label="True"
    )
    sns.scatterplot(x=scaled_X_test.reshape(-1), y=y_hat_test, label="Predicted")
    plt.legend()
    plt.show()

    r2 = r2_score(scaled_y_test, y_hat_test)
    print("Evaluation of quantum model - R2 =", r2)


def main():
    evaluate_quantum()


if __name__ == "__main__":
    main()
