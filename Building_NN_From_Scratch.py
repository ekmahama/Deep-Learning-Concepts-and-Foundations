import keras
from keras import ops
from keras.datasets import mnist
import math
import tensorflow as tf
import numpy as np
from numpy.typing import NDArray
from typing import Callable


class NaiveDense:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Callable[[tf.Tensor], tf.Tensor] | None = None,
    ) -> None:
        self.activation = activation
        self.W = keras.Variable(
            shape=(input_size, output_size), initializer = "uniform"
        )
        self.b = keras.Variable(
            shape=(output_size,), initializer = "zeros"
        )

    def __call__(self, inputs: tf.Tensor | NDArray[np.float32]) -> tf.Tensor:
        x =  ops.matmul(inputs, self.W) + self.b
        if self.activation:
            x = self.activation(x)
        return x
    
    @property
    def weights(self) -> list[keras.Variable]:
        return [self.W, self.b]


class NaiveSequential:
    def __init__(self, layers: list[NaiveDense]) -> None:
        self.layers = layers

    def __call__(self, inputs: tf.Tensor | NDArray[np.float32]) -> tf.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self) -> list[keras.Variable]:
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights
    

class BatchGenerator:
    def __init__(
        self,
        features: tf.Tensor | NDArray[np.float32],
        labels: tf.Tensor | NDArray[np.int64] | NDArray[np.uint8],
        batch_size: int = 128,
    ) -> None:
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.index = 0
        self.number_of_batches = math.ceil(len(features) / batch_size)

    def next(
        self,
    ) -> tuple[tf.Tensor | NDArray[np.float32], tf.Tensor | NDArray[np.int64] | NDArray[np.uint8]]:
        features = self.features[self.index: self.index + self.batch_size]
        labels = self.labels[self.index: self.index + self.batch_size]
        self.index += self.batch_size
        return features, labels


def one_training_step(
    model: NaiveSequential,
    features_batch: tf.Tensor | NDArray[np.float32],
    labels_batch: tf.Tensor | NDArray[np.int64] | NDArray[np.uint8],
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        predictions = model(features_batch)
        loss = ops.sparse_categorical_crossentropy(labels_batch, predictions)
        average_loss = ops.mean(loss)
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss


def update_weights(
    gradients: list[tf.Tensor | tf.IndexedSlices | None],
    weights: list[keras.Variable],
) -> None:
    lr = 1e-3
    for g, w in zip(gradients, weights):
        if g is not None:
            w.assign_sub(lr * g)

def update_weights_v2(
    gradients: list[tf.Tensor | tf.IndexedSlices | None],
    weights: list[keras.Variable],
) -> None:
    learning_rate = 1e-3
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    optimizer.apply_gradients(zip(gradients, weights))


def fit(
    model: NaiveSequential,
    features: tf.Tensor | NDArray[np.float32],
    labels: tf.Tensor | NDArray[np.int64] | NDArray[np.uint8],
    epochs: int,
    batch_size: int = 128,
) -> None:
    for epoch_counter in range(epochs):
        print(f"Epoch: {epoch_counter}")
        batch_generator = BatchGenerator(features, labels, batch_size)
        for batch_counter in range(batch_generator.number_of_batches):
            features_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, features_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")

            
if __name__=="__main__":
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255


    model = NaiveSequential(
        [
            NaiveDense(input_size = train_images.shape[1], output_size = 512, activation = ops.relu),
            NaiveDense(input_size = 512, output_size = 10, activation = ops.softmax)

        ]
    )

    fit(model, train_images, train_labels, epochs=10, batch_size=128)


    predictions = model(test_images)
    predicted_labels = ops.argmax(predictions, axis=1)
    matches = predicted_labels == test_labels
    print(f"accuracy: {ops.mean(matches):.2f}")