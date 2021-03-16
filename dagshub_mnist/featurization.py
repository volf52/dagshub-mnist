"""
Create feature CSVs for train and test datasets
"""

import json

import numpy as np
import pandas as pd


def featurization():
    print("Loading datasets...")
    train_data = pd.read_csv("./data/train_data.csv", header=None, dtype=float)
    test_data = pd.read_csv("./data/train_data.csv", header=None, dtype=float)

    # Normalize the train data
    print("Normalizing data...")
    train_mean = train_data.values[:, 1:].mean()
    train_std = train_data.values[:, 1:].std()

    # Normalize train and test data according to train data distrib
    train_data.values[:, 1:] -= train_mean
    train_data.values[:, 1:] /= train_std

    test_data.values[:, 1:] -= train_mean
    test_data.values[:, 1:] /= train_std

    print("Saving processed dataset and normalization parameters...")
    np.save("./data/processed_train_data", train_data)
    np.save("./data/processed_test_data", test_data)

    with open("./data/norm_params.json", "w") as f:
        json.dump({"mean": train_mean, "std": train_std}, f)

    print("Done.")


if __name__ == "__main__":
    featurization()
