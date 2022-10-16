from collections import Counter
from typing import AnyStr

import numpy as np


class KNN:
    ## TODO: maybe look into see if KNN can inherit from base.BaseModel somehow
    # (even though KNN is a non-parametric model)
    def __init__(self, k: int, metric="euclidean"):
        self.k = k
        self.metric = metric

    def compute_distances(self, x_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """
        In order to answer part 2, I've included code to compute
        the distance using either the Manhattan or Euclidean formulations.
        """
        euclidean_dist = lambda x, y: np.linalg.norm(x - y, axis=1)
        manhattan_dist = lambda x, y: np.sum(np.abs(x - y))

        if self.metric == "euclidean":
            return euclidean_dist(x_test, X_train)
        return manhattan_dist(x_test, X_train)

    def most_common_class(self, class_dist: Counter) -> AnyStr:
        """linear search to find the first label w/ the highest frequency"""
        most_common_count = max(list(class_dist.values()))
        for label, label_occurences in class_dist.items():
            if label_occurences == most_common_count:
                return label

    def fit(self):
        raise RuntimeWarning("Sorry, calling KNN.fit() doesn't actually do anything.")

    def predict(self, X_train, y_train, X_test):
        y_pred = np.zeros(X_test.shape[0], dtype=object)
        for test_sample_index in range(X_test.shape[0]):
            ix = test_sample_index
            # find the dist to all training pts
            distances_for_1_test_pt = self.compute_distances(
                X_test.values[ix][:].reshape(1, -1), X_train
            )
            # sort the distances --> choose the closest k
            nearest_neighbors = y_train.values[np.argsort(distances_for_1_test_pt)][
                : self.k
            ]
            # assign a class to the test point
            nearest_neighbors = np.squeeze(nearest_neighbors).tolist()
            y_pred[ix] = self.most_common_class(Counter(nearest_neighbors))

        # all done!
        return y_pred
