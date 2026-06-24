import numpy as np
import jax
from jax import numpy as jnp
lr = 0.01

def model(inputs, W, b):
    return jax.nn.relu(jnp.matmul(inputs, W) + b)

def mean_squared_error(predictions, targets):
    return jnp.mean(jnp.square((predictions - targets)))

def compute_loss(state, inputs, targets):
    W, b = state
    predictions = model(inputs, W, b)
    loss = mean_squared_error(predictions, targets)
    return loss


def training_step(inputs, targets, W, b):
    grad_fn = jax.value_and_grad(compute_loss)
    loss, grads = grad_fn((W, b), inputs, targets)
    W -= lr * grads[0]
    b -= lr * grads[1]
    return loss, W, b


def training_loop(inputs, targets, W, b):
    for step in range(40):
        loss, W, b = training_step(inputs, targets, W, b)
        print(f"Loss at step {step} : {loss: .4f}")

if __name__ == "__main__":
    number_samples_per_class = 1000
    negative_samples = np.random.multivariate_normal(mean=[0, 3], cov=[[1, 0.5],[0.5, 1]], size=number_samples_per_class)
    negative_targets = np.zeros((number_samples_per_class, 1), dtype=np.float32)
    positive_samples = np.random.multivariate_normal(mean=[3, 0], cov=[[1, 0.5],[0.5, 1]], size=number_samples_per_class)
    positive_targets = np.ones((number_samples_per_class, 1), dtype=np.float32)

    inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
    targets = np.vstack((negative_targets, positive_targets)).astype(np.float32)

    input_dim = 2
    output_dim = 1

    W = jnp.array(np.random.uniform(size=(input_dim, output_dim)))
    b = jnp.array(np.zeros(shape=(output_dim,)))

    training_loop(inputs, targets, W, b)