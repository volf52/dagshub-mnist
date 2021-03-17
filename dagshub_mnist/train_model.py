import json
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from dagshub_mnist.torch_model import Net


def trainNN(
    model: nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer,
    epoch: int,
):
    log_interval = 100
    model.train()

    data: torch.Tensor
    target: torch.Tensor
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        yhat = model(data)
        loss = F.nll_loss(yhat, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def train_model():
    start_time = time.time()

    print("Setting up training params...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 3
    learning_rate = 0.01
    momentum = 0.5

    print("Loading data...")
    train_data: np.ndarray = np.load("./data/processed_train_data.npy")

    labels = torch.tensor(train_data[:, 0]).long()
    data = torch.tensor(train_data[:, 1:].reshape([-1, 1, 28, 28])).float()
    torch_train_data = torch.utils.data.TensorDataset(data, labels)
    train_loader = torch.utils.data.DataLoader(
        torch_train_data, batch_size=batch_size, shuffle=True
    )

    print("Training model...")
    model = Net().float().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(1, epochs + 1):
        trainNN(model, device, train_loader, optimizer, epoch)

    end_time = time.time()

    print("Saving model and training time metric...")
    with open("./artifacts/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("./metrics/train_metric.json", "w") as f:
        json.dump({"train_time": end_time - start_time}, f)

    print("Done...")


if __name__ == "__main__":
    train_model()
