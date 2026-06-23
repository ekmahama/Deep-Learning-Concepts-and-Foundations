import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

input_dim = 2
output_dim = 1
lr = 0.001


class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.W = torch.nn.Parameter(torch.rand(size=(input_dim, output_dim)))
        self.b = torch.nn.Parameter(torch.zeros(size=(output_dim,)))

    def forward(self, inputs):
        return torch.matmul(inputs, self.W) + self.b

def mean_squared_error(targets, predictions):
    per_sample_error = torch.square(targets - predictions)
    return torch.mean(per_sample_error)

def batch_generator(inputs, targets, batch_size):
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch_inputs, batch_targets in dataloader:
        yield batch_inputs, batch_targets

def training_step(model, optimizer, inputs, targets):
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = mean_squared_error(targets, predictions)
    loss.backward()
    optimizer.step()

    return loss

def training_loop(model, optimizer, inputs, targets, num_epoch=5, batch_size = 200):
    for step in range(30):
        for epoch in range(num_epoch):
            batch_num = 0
            for batch_inputs, batch_targets in batch_generator(inputs, targets, batch_size):
                loss = training_step(model, optimizer, batch_inputs, batch_targets)
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

    model = LinearModel(input_dim, output_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    torch_inputs = torch.tensor(inputs)
    torch_targets = torch.tensor(targets)


    training_loop(model, optimizer, torch_inputs, torch_targets, 10, 128)