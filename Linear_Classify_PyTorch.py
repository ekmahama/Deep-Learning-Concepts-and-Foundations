import numpy as np
import tensorflow as tf
import torch

input_dim = 2
output_dim = 1
lr = 0.001
W = torch.rand(size=(input_dim, output_dim), requires_grad=True)
b = torch.zeros(size=(output_dim,), requires_grad=True)

def model(inputs, W, b):
    return torch.matmul(inputs, W) + b

def mean_squared_error(targets, predictions):
    per_sample_error = torch.square(targets - predictions)
    return torch.mean(per_sample_error)

def batch_generator(inputs, targets, batch_size):
    data = tf.data.Dataset.from_tensor_slices((inputs, targets))
    data = data.shuffle(buffer_size=len(inputs))
    data = data.batch(batch_size)
    for  batch_inputs, batch_targets in data:
        yield  batch_inputs, batch_targets

def training_step(inputs, targets, W, b):
    if hasattr(inputs, "numpy"):
        inputs = inputs.numpy()
    if hasattr(targets, "numpy"):
        targets = targets.numpy()

    inputs = torch.from_numpy(np.asarray(inputs, dtype=np.float32))
    targets = torch.from_numpy(np.asarray(targets, dtype=np.float32))
    predictions = model(inputs, W, b)
    loss = mean_squared_error(targets, predictions)
    loss.backward()
    grad_loss_wrt_W, grad_loss_wrt_b = W.grad, b.grad
    with torch.no_grad():
        W -= lr * grad_loss_wrt_W
        b -= lr * grad_loss_wrt_b
    W.grad = None
    b.grad = None
    return loss

def training_loop(inputs, targets, num_epoch=5, batch_size = 128):
    for step in range(40):
        for epoch in range(num_epoch):
            batch_num = 0
            for batch_inputs, batch_targets in batch_generator(inputs, targets, batch_size):
                loss = training_step(batch_inputs, batch_targets, W, b)
                if batch_num % 100 == 0:
                    print(f"Epoch {epoch}, loss : {loss.item():4.4f}")
                batch_num += 1

        
if __name__ == "__main__":
    number_samples_per_class = 1000
    negative_samples = np.random.multivariate_normal(mean=[0, 3], cov=[[1, 0.5],[0.5, 1]], size=number_samples_per_class)
    negative_targets = np.zeros((number_samples_per_class, 1), dtype=np.float32)
    positive_samples = np.random.multivariate_normal(mean=[3, 0], cov=[[1, 0.5],[0.5, 1]], size=number_samples_per_class)
    positive_targets = np.ones((number_samples_per_class, 1), dtype=np.float32)

    inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
    targets = np.vstack((negative_targets, positive_targets)).astype(np.float32)

    training_loop(inputs, targets, 10, 128)