import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, max_iter=300, tol=1e-9, max_k=10):
        self.n_clusters = None
        self.max_iter = max_iter
        self.tol = tol
        self.max_k = max_k
        self.centroids = None

    def best_k(self, X):
        """
        Computes the best number of clusters using the elbow method

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            max_k (int): maximum number of clusters to try

        Returns:
            A tuple (best_k, distortions) where:
                - best_k (int): the best number of clusters
                - distortions (array<max_k>): the distortion for each k
        """

        print("Computing the best number of clusters...")

        X = X.copy()
        silhouetttes = []
        for k in range(2, self.max_k+1):
            self.fit_one(X, k)
            z = self.predict(X)
            silhouetttes.append(euclidean_silhouette(X, z))
        
        best_k = np.argmax(silhouetttes) + 2
        print("Best number of clusters: {}".format(best_k))
        return best_k


    def fit(self, X, n_clusters=None):
        """
        Fit several K-means models and choose the best one, using fit_one()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        
        """
        X = X.copy()
        if n_clusters is None:
            self.n_clusters = self.best_k(X)
        else:
            self.n_clusters = n_clusters

        # 3D array to store the centroids of each model
        centroids = [] 
        # 1D array to store the distortion of each model
        distortions = []
        for i in range(10):
            model = KMeans()
            model.fit_one(X, self.n_clusters)
            z = model.predict(X)
            centroids.append(model.get_centroids())
            distortions.append(euclidean_distortion(X, z))

        # Choose the model with the lowest distortion
        self.centroids = centroids[np.argmin(distortions)]

    
    
    def fit_one(self, X, n_clusters):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        X = X.copy()
        #Standardize the data
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        self.n_clusters = n_clusters   
        self.centroids = self.kmeans_plusplus_init(X)
        #self.centroids = self.random_init(X)
       
        
        num_iter = 0
        X["cluster_label"] = 0

        for i in range(self.max_iter):
            X["cluster_label"] = X.apply(lambda x: np.argmin(np.linalg.norm(np.array([x[:-1]]*self.n_clusters) - self.centroids, axis=1)**2), axis=1)
            new_centroids = X.groupby("cluster_label").mean().values
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids
            num_iter += 1
        
        #De-standardize the data
        X.iloc[:,:-1] = X.iloc[:,:-1] * std + mean
        self.centroids = self.centroids * np.array(std) + np.array(mean)

        return X, num_iter
    
    def kmeans_plusplus_init(self, X):
        # Initialize centroids using the K-means++ algorithm
        centroids = []
        centroids.append(X.sample().values[0])
        for i in range(1, self.n_clusters):
            distances = X.apply(lambda x: np.min(np.linalg.norm(np.array([x]*len(centroids)) - np.array(centroids), axis=1)), axis=1)
            prob = distances / distances.sum()

            centroids.append(X.sample(weights=prob**2).values[0])
    
        return np.array(centroids)
    
    def random_init(self, X):
        # Initialize centroids randomly
        centroids = []
        for i in range(self.n_clusters):
            centroids.append(X.sample().values[0])
        
        return np.array(centroids)

    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        X = X.copy()
        #Standardize the data
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        standardized_centroids = (self.centroids - np.array(mean)) / np.array(std)
        X["cluster_label"] = 0

        #Predict labels for each point in X using the the centroids
        X["cluster_label"] = X.apply(lambda x: np.argmin(np.linalg.norm(np.array([x[:-1]]*self.n_clusters) - standardized_centroids, axis=1)), axis=1)
        return X["cluster_label"].values
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids
    
    
    
    
# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += np.sum(((Xc - mu) ** 2).sum(axis=1))
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  