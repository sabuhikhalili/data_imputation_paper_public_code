import numpy as np
from utilities import svd_flip

def apc(X, kmax):
    # Error checking
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(kmax, int):
        raise ValueError("'kmax' must be an integer.")
    if kmax > min(X.shape):
        raise ValueError("'kmax' must be smaller than the size of X.")
    
    # Create output object
    out = dict()
    out['X'] = X
    out['kmax'] = kmax

    # Perform SVD on the covariance matrix
    U, d, VT = np.linalg.svd(X, full_matrices=False)
    U, VT = svd_flip(U, VT)

    T, N = X.shape
    # need to transpose to apply the functions in the same way with R
    V = VT.T 
    D = np.diag(d)
    D = D / (np.sqrt(N * T))
    Dr = D[:kmax, :kmax]

    
    # Get variance explained by singular values
    explained_variance = (d**2) / (T - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var

    out['Fhat'] = np.sqrt(T) * U[:, :kmax]
    out['Lamhat'] = np.sqrt(N) * V[:, :kmax] @ Dr
    out['Chat'] = out['Fhat'] @ out['Lamhat'].T
    out['d0'] = d
    out['d'] = d[:kmax]
    out['Dhat'] = np.diag(out['d'])
    out['ehat'] = out['X'] - out['Chat']
    out['explained_variance'] = explained_variance
    out['explained_variance_ratio'] = explained_variance_ratio

    return out

def pca(X, kmax, standardize=False):
    # Create output object
    out = dict()
    out['X'] = X.copy()
    out['kmax'] = kmax

    T, N = X.shape
    
    # Center the data by subtracting the mean
    if standardize:
        mean = np.mean(X, axis=0)
        X = X - mean

    # Perform SVD on the covariance matrix
    U, d, VT = np.linalg.svd(X, full_matrices=False)
    U, VT = svd_flip(U, VT)



    D = np.sqrt((d**2) / (T - 1))
    # D = d/np.sqrt(T-1)
    # D = D / (np.sqrt(N * T))
    D = np.diag(D)
    Dr = D[:kmax, :kmax]

    V = VT.T 


    # Get variance explained by singular values
    explained_variance = (d**2) / (T - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var

    out['Fhat'] =  np.sqrt(T-1)*U[:, :kmax] # standardized principal components scaled to unit norm
    # out['Lamhat'] = V[:, :kmax] * (d[:kmax] / np.sqrt(T - 1))
    out['Lamhat'] = V[:, :kmax] @ Dr
    # out['Lamhat'] = np.sqrt(N) * V[:, :kmax] @ Dr
    out['Chat'] = out['Fhat'] @ out['Lamhat'].T
    out['explained_variance'] = explained_variance
    out['explained_variance_ratio'] = explained_variance_ratio
    out['V'] = V
    out['d'] = d
    out['Dr'] = Dr


    return out

