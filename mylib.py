import numpy as np


# Generate n random points in R^d, each one drawn independently and uniformly at random in the interval [0.0, 1.0).
# Should return the points as an n-by-D numpy array
def genpoints(n, D):
    return np.random.random((n, D))


# Generate c clusters, each of n random points in R^d. The points in the i-th cluster should be drawn uniformly at
# random in the interval [i, i + 1.0). Should return the points as an nc-by-D Numpy array
def genclusters(c, n, D):
    M = genpoints(c*n,D)
    for i in range(c):
        M[i*n : (1 + i)*n] += i
    return M


# EXERCISE: write a function euc(x,y) that computes the Euclidean distance
# between two points x and y in R^n, given as numpy arrays of n elements.
def euc(x,y):
    return np.sqrt(np.sum((np.array(x)-np.array(y))**2))


# Return the symmetric Euclidean distance matrix between all pairs of rows in the Numpy
def alldist(X):
    n = X.shape[0]
    D = np.zeros((n,n))
    for i in range(n-1):
        for j in range(i+1, n):
            D[i,j] = euc(X.iloc[i], X.iloc[j])
    return D+D.transpose()


# Generate the D-by-d random matrix used by Achlioptas' algorithm. The i-th column of the matrix
# specifies the weights with which the coordinates of a D-dimensional vector are combined to produce
# the i-th entry of its d-dimensional transform. Each entry is chosen independently and uniformly at
# random in {-1,1}
def achmat(D,d):
    return np.where((np.random.randn(D,d) < 0),-1,1)


# Perform dimensionality reduction on a set of points. Takes in input a set of n points in
# R^D , in the form of an n-by-D matrix X, and reduces each point to have dimensionality
# d. It returns the n-by-d NumPy array containing the n reduced points as rows. Recall
# that the linear map that reduces point x is f(x) = d^−0.5 * X^T * A, where A is the matrix
# obtained with achmat(), and T means transpose.
def reduce(X,d):
    A = achmat(X.shape[1],d)
    return np.dot(X,A) / np.sqrt(d)


# Compute the distance distortion between all pairs of points, according to the distances
# in the matrices dm1 and dm2. Recall that the distance distortion between point i and
# point j is simply dm2(i,j)/dm1(i,j). The result is a NumPy array of length n(n − 1)/2
# containing, for each pair of points (i, j) with 1 <= i < j <= n, the ratio between their
# distance according to dm1 and their distance according to dm2. (Note that the distance
# between a point and itself is not considered).
def distortion(dm1, dm2):
    A = dm2/dm1
    return A[np.triu_indices(len(A),k=1)]
