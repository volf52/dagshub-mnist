import json
import pickle

import numpy as np
import torch
from sklearn.metrics import accuracy_score


def eval_model():
    print("Loading data and model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data: np.ndarray = np.load("./data/processed_test_data.npy")

    with open("./artifacts/model.pkl", "rb") as f:
        model: torch.nn.Module = pickle.load(f)

    model = model.to(device)
    model.eval()
    labels = test_data[:, 0]
    data = torch.tensor(test_data[:, 1:].reshape([-1, 1, 28, 28])).float().to(device)

    print("Running model on test data...")
    preds = model(data).max(1, keepdim=True)[1].cpu().data.numpy()

    print("Calculating metrics...")
    metrics = {"acc": accuracy_score(labels, preds)}

    with open("./metrics/eval.json", "w") as f:
        json.dump(metrics, f)
    print("Done")


if __name__ == "__main__":
    eval_model()
