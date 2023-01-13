import numpy as np
from sklearn.neighbors import radius_neighbors_graph


class SpectralClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters
        self.max_iterations = 1000
        self.num_examples = X.shape[0]
        self.num_features = X.shape[0]

    # Get eigenvectors and eigenvalues
    def get_eigs_from_laplacian(self, X):

        # Compute graph Laplacian from data
        W = radius_neighbors_graph(X, 0.4, mode="distance", include_self=True).toarray()
        D = np.diag(np.sum(np.array(W), axis=1))
        L = D - W

        return np.linalg.eig(L)

    # Randomly initialize centroids
    def initialize_random_centroids(self, X):
        # Matrix of zeros
        centroids = np.zeros((self.K, self.num_features))

        for k in range(self.K):
            centroid = X[np.random.choice(range(self.num_examples))]
            centroids[k] = centroid

        return centroids

    # Create cluster function
    def create_cluster(self, X, centroids):
        clusters = [[] for _ in range(self.K)]

        for point_idx, point in enumerate(X):
            # Get closest centroid using Euclidean distance
            closest_centroid = np.argmin(
                np.sqrt(np.sum((point - centroids) ** 2, axis=1))
            )
            clusters[closest_centroid].append(point_idx)

        return clusters

    # Get new centroids
    def calculate_new_centroids(self, cluster, X):
        # Matrix of zeros
        centroids = np.zeros((self.K, self.num_features))
        for idx, cluster in enumerate(cluster):
            # New centroid is mean of cluster
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = new_centroid
        return centroids

    # Predict/assign cluster label
    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred

    # Fit data
    def fit(self, X):
        eigvals, eigvecs = self.get_eigs_from_laplacian(X)

        # For K=2, spectral clustering is done through Fiedler vector.
        if self.K == 2:
            # Get index of second smallest eigenvalue
            fiedler_index = np.where(eigvals == np.partition(eigvals, 1)[1])[0][0]

            # Use Fiedler vector to cluster
            y_pred = eigvecs[:, fiedler_index].copy()
            y_pred[y_pred < 0] = 0
            y_pred[y_pred > 0] = 1
        # For K>2, clustering is done through k-means with eigenvectors as features
        elif self.K > 2:
            centroids = self.initialize_random_centroids(eigvecs)

            for _ in range(self.max_iterations):
                clusters = self.create_cluster(eigvecs, centroids)
                previous_centroids = centroids
                centroids = self.calculate_new_centroids(clusters, eigvecs)
                diff = centroids - previous_centroids
                if not diff.any():
                    break

            y_pred = self.predict_cluster(clusters, eigvecs)

        return y_pred
