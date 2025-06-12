# Pricipal Component Analysis (PCA) implementation

"""
Steps for the PCA algorithm:
1. Subtract the mean from X
2. Caclulate the Cov(X,X)
3. Calculate the eigenvalues and eigenvectors of the covariance matrix
4. Sort the eigenvectors according to their eigenvalues in decreasing order
5. Choose first k eigenvectors and that will be the new k dimensions
6. Transform the original n-dimensional data points into k dimensions (= Projection with dot product)
"""

import numpy as np

class PrincipalComponentAnalysis:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        # Step 1: Substract the mean from X
        self.mean = np.mean(X, axis = 0) # axis = 0 means row-wise mean and axis = 1 means column-wise mean
        X = X - self.mean
        # Step 2: Calculate the Cov(X, X)
        covariance_matrix = np.cov(X.T)
        # Step 3: Calculate the eigenvalues and eigenvectors of the covariance matrix
        eigenvectors, eigenvalues = np.linalg.eig(covariance_matrix)
        eigenvectors = eigenvectors.T # Transpose v = [:, i] column vector
        # Step 4: Sort the eigenvectors according to their eigenvalues in decreasing order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[sorted_indices]
        eigenvalues = eigenvalues[sorted_indices]
        # Step 5: Choose first k eigenvectors and that will be the new k dimensions
        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        # Step 6: Transform the original n-dimensional data points into k dimensions
        X = X - self.mean
        return np.dot(X, self.components.T)
    
# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn import datasets

    # data = datasets.load_digits()
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Project the data onto the 2 primary principal components
    pca = PrincipalComponentAnalysis(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()