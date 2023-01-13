import numpy as np
from sklearn.neighbors import radius_neighbors_graph 

class SpectralClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters # cluster number
        self.max_iterations = 1000 # max iteration. don't want to run inf time
        self.num_examples = X.shape[0] 
        self.num_features = X.shape[0] # num of examples, num of features
    
    # Get eigenvectors and eigenvalues
    def get_eigs_from_laplacian(self, X):

        # Compute graph Laplacian from data
        W = radius_neighbors_graph(X, 0.4, mode='distance', include_self=True).toarray()
        D = np.diag(np.sum(np.array(W), axis=1))
        L = D - W
        
        return np.linalg.eig(L)
    
    # randomly initialize centroids
    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.num_features)) # row , column full with zero 
        
        for k in range(self.K): # iterations of 
            centroid = X[np.random.choice(range(self.num_examples))] # random centroids
            centroids[k] = centroid
            
        return centroids # return random centroids
    
    # create cluster Function
    def create_cluster(self, X, centroids):
        clusters = [[] for _ in range(self.K)]
        
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin(
                np.sqrt(np.sum((point-centroids)**2, axis=1))
            ) # closest centroid using euler distance equation(calculate distance of every point from centroid)
            clusters[closest_centroid].append(point_idx)
            
        return clusters 
    
    # new centroids
    def calculate_new_centroids(self, cluster, X):
        centroids = np.zeros((self.K, self.num_features)) # row , column full with zero
        for idx, cluster in enumerate(cluster):
            new_centroid = np.mean(X[cluster], axis=0) # find the value for new centroids
            centroids[idx] = new_centroid
        return centroids
    
    # prediction
    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples) # row1 fillup with zero
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred
        
    # Fit data
    def fit(self, X):
        eigvals, eigvecs = self.get_eigs_from_laplacian(X)
        
        #
        #idx = eigvals.argsort()[::-1]   
        #eigvals = eigvals[idx]
        #eigvecs = eigvecs[:,idx]
        
        # For now, spectral clustering is done through Fiedler vector.
        if self.K == 2:
            # Get index of second smallest eigenvalue
            fiedler_index = np.where(eigvals == np.partition(eigvals, 1)[1])[0][0]

            # Use Fiedler vector to cluster
            y_pred = eigvecs[:, fiedler_index].copy()
            y_pred[y_pred < 0] = 0
            y_pred[y_pred > 0] = 1
        
        elif self.K > 2:
            #eigvecs = eigvecs[:, :self.K]
            centroids = self.initialize_random_centroids(eigvecs)
            
            for _ in range(self.max_iterations):
                clusters = self.create_cluster(eigvecs, centroids) # create cluster
                previous_centroids = centroids
                centroids = self.calculate_new_centroids(clusters, eigvecs) # calculate new centroids
                diff = centroids - previous_centroids # calculate difference
                if not diff.any():
                    break    
                    
            y_pred = self.predict_cluster(clusters, eigvecs)
                    
        return y_pred
