import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

input_dim, output_dim = 2, 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))
learning_rate = 1e-3

def model(inputs):
    return tf.matmul(inputs, W) + b

def mean_squared_error(targets, predictions):
    per_sample_mean = tf.square((targets - predictions))
    return tf.reduce_mean(per_sample_mean)

@tf.function(jit_compile=True)
def training_step(model, batch_inputs, batch_targets):
    with tf.GradientTape() as tape:
        predictions = model(batch_inputs)
        loss = mean_squared_error(batch_targets, predictions)
    grad_loss_wrt_W , grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss

def batch_generator(inputs, targets, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.shuffle(buffer_size=len(inputs))
    dataset = dataset.batch(batch_size)
    for batch_inputs, batch_targets in dataset:
        yield batch_inputs, batch_targets

def training_loop(inputs, targets, model, num_epochs = 10): 
    for epoch in range(num_epochs):
        batch_num = 0
        for batch_inputs, batch_targets in batch_generator(inputs, targets, batch_size=128):
            loss = training_step(model, batch_inputs, batch_targets)
            if batch_num % 100 == 0:
                print(f"Epoch {epoch}, loss : {loss:4.4f}")
            batch_num += 1
        

if __name__ == "__main__":
    number_samples_per_class = 10000
    negative_samples = np.random.multivariate_normal(mean=[0, 3], cov=[[1, 0.5],[0.5, 1]], size=number_samples_per_class)
    negative_targets = np.zeros((number_samples_per_class, 1), dtype=np.float32)
    positive_samples = np.random.multivariate_normal(mean=[3, 0], cov=[[1, 0.5],[0.5, 1]], size=number_samples_per_class)
    positive_targets = np.ones((number_samples_per_class, 1), dtype=np.float32) 
    inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
    targets = np.vstack((negative_targets, positive_targets)).astype(np.float32) 

    training_loop(inputs, targets, model, 20)

    predictions = model(inputs)
    x = np.linspace(-1, 4, 100)
    y = - W[0] / W[1] * x + (0.5 - b) / W[1]
    plt.plot(x, y, "-r")
    plt.scatter(inputs[:,0], inputs[:,1], c=predictions[:,0] > 0.5)
    plt.show()
