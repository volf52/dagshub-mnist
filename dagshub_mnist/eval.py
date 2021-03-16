import json
import pickle

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsOneClassifier


def eval_model():
    print("Loading data and model...")
    test_data: np.ndarray = np.load("./data/processed_test_data.npy")

    with open("./artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)

    labels = test_data[:, 0]
    data = test_data[:, 1:]

    print("Running model on test data...")
    preds = model.predict(data)

    print("Calculating metrics...")
    metrics = {"acc": accuracy_score(labels, preds)}

    with open("./metrics/eval.json", "w") as f:
        json.dump(metrics, f)
    print("Done")


if __name__ == "__main__":
    eval_model()
