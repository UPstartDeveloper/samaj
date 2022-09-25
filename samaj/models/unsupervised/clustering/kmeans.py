import numpy as np


class KMeans:
    def __init__(self, k: int):
        """
        My implementation of the k-means algorithm in Python. 
        Relies heavily on NumPy and a lecture I found on MIT OCW:
            https://www.youtube.com/watch?v=esmzYhuFnds
        """
        self.centroid_coords_cov = dict()
        self.num_clusters = k

    def train(self, X: np.array) -> dict:
        """
        Assumes k is a positive integer.

        Returns: dict: contains the dict and covariance of each centroid
        """
        # A: init k random centroids from existing data
        rng = np.random.default_rng()
        centroids = rng.choice(X, size=(self.num_clusters,))
        centroids_assigned_pts = dict(zip(
            # scalars mapped to 2D arrays
            range(self.num_clusters), [[] for _ in range(self.num_clusters)]
        ))
        # B: converge on optimal centroids
        keep_going = True
        
        while keep_going is True:
            # 1: assign each point to a centroid
            for sample in X:
                centroid_assignment = np.argmin([
                    np.linalg.norm(sample - centroids, axis=1)  # Euclidean distance
                ])
                centroids_assigned_pts[centroid_assignment].append(sample)
            # 2: update centroid placements themselves
            cap = centroids_assigned_pts  # just an abbreviation
            new_centroids = np.array([
                np.mean(np.array(cap[centroid_label]), axis=0)
                for centroid_label in centroids_assigned_pts.keys()
            ])
            # 3: decide if we should continue
            if np.equal(centroids, new_centroids).all():
                keep_going = False
            centroids = new_centroids[:]

        # C: collect the results
        for centroid_label in centroids_assigned_pts:
            self.centroid_coords_cov[centroid_label] = (
                centroids[centroid_label],
                np.cov(self.centroids_assigned_pts[centroid_label], rowvar=False)
            )
        return self.centroid_coords_cov

    def predict(self, X: np.array):
        centroids = [coords for coords, _ in self.centroid_coords_cov.values()]
        centroid_assignments = [
            np.argmin([
                np.linalg.norm(sample - centroids, axis=1)  # Euclidean distance
            ]) for sample in X
        ]

        return centroid_assignments
