import json
import pickle
import time

import numpy as np
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC


def train_model():
    start_time = time.time()

    print("Loading data...")
    train_data: np.ndarray = np.load("./data/processed_train_data.npy")
    print("Choosing smaller sample to shorten training time...")
    np.random.seed(32)

    num_samples = 5000
    choice = np.random.choice(train_data.shape[0], num_samples, replace=False)
    train_data = train_data[choice, :]

    labels = train_data[:, 0]
    data = train_data[:, 1:]

    print("Training model...")
    model = OneVsOneClassifier(SVC(kernel="linear"), n_jobs=6)
    model.fit(data, labels)
    end_time = time.time()

    print("Saving model and training time metric...")
    with open("./artifacts/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("./metrics/train_metric.json", "w") as f:
        json.dump({"train_time": end_time - start_time}, f)

    print("Done...")


if __name__ == "__main__":
    train_model()
