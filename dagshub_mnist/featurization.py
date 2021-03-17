"""
Create feature CSVs for train and test datasets
"""

import base64
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def featurization():
    print("Loading datasets...")
    train_data = pd.read_csv("./data/train_data.csv", header=None, dtype=float).values
    test_data = pd.read_csv("./data/train_data.csv", header=None, dtype=float).values

    print("Creating PCA object...")
    pca = PCA(n_components=15, whiten=True)
    pca.fit(train_data[:, 1:])

    train_lbls = train_data[:, 0].reshape(-1, 1)
    test_lbls = test_data[:, 0].reshape(-1, 1)

    train_data = np.concatenate([train_lbls, pca.transform(train_data[:, 1:])], axis=1)
    test_data = np.concatenate([test_lbls, pca.transform(test_data[:, 1:])], axis=1)

    print("Saving processed datasets and norm parameters...")
    np.save("./data/processed_train_data", train_data)
    np.save("./data/processed_test_data", test_data)

    with open("./data/norm_params.json", "w") as f:
        pca_as_string = base64.encodebytes(pickle.dumps(pca)).decode("utf-8")
        json.dump({"pca": pca_as_string}, f)

    print("Done.")


if __name__ == "__main__":
    featurization()
