# Code is coverted by Sabuhi Khalili from R to Python using https://rdrr.io/github/cykbennie/fbi/src/R/tw_apc.R
#' @author Yankang (Bennie) Chen <yankang.chen@@yale.edu>
#' @author Serena Ng <serena.ng@@columbia.edu>
#' @author Jushan Bai <jushan.bai@@columbia.edu>
#'
#' @references
#' Ercument Cahan, Jushan Bai, and Serena Ng (2021),
#' \emph{Factor-Based Imputation of Missing Values and Covariances in Panel Data of Large Dimensions}.
#' \url{https://arxiv.org/abs/2103.03045}


import numpy as np
from numpy.linalg import lstsq
from pca import apc

def tw_apc(X, kmax, center=True, standardize=True, re_estimate=True):
    """
    Tall-Wide Imputation of Missing Value in Panel Data

    tw_apc imputes the missing values in a given panel data using the method of "Tall-Wide".

    Args:
        X (ndarray): A matrix of size T by N with missing values.
        kmax (int): Integer, indicating the maximum number of factors.
        center (bool): Logical, indicating whether or not X should be demeaned.
        standardize (bool): Logical, indicating whether or not X should be scaled.
        re_estimate (bool): Logical, indicating whether or not output factors,
            'Fhat', 'Lamhat', 'Dhat', and 'Chat', should be re-estimated from the imputed data.

    Returns:
        dict: A dictionary with the following elements:
            - Fhat (ndarray): Estimated F.
            - Lamhat (ndarray): Estimated Lambda.
            - Dhat (ndarray): Estimated D.
            - Chat (ndarray): Equals Fhat x Lamhat'.
            - ehat (ndarray): Equals Xhat - Chat.
            - data (ndarray): X with missing data imputed.
            - X (ndarray): The original data with missing values.
            - kmax (int): The maximum number of factors.
            - center (bool): Logical, indicating whether or not X was demeaned in the algorithm.
            - standardize (bool): Logical, indicating whether or not X was scaled in the algorithm.
            - re_estimate (bool): Logical, indicating whether or not output factors,
                'Fhat', 'Lamhat', 'Dhat', and 'Chat', were re-estimated from the imputed data.

    Author:
        Yankang (Bennie) Chen <yankang.chen@@yale.edu>
        Serena Ng <serena.ng@@columbia.edu>
        Jushan Bai <jushan.bai@@columbia.edu>

    References:
        Jushan Bai and Serena Ng (2021), Matrix Completion, Counterfactuals, and Factor Analysis of Missing Data.
        URL: https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1967163
    """

     # Error checking
    if not isinstance(center, bool):
        raise ValueError("'center' must be logical.")
    if not isinstance(standardize, bool):
        raise ValueError("'standardize' must be logical.")
    if not isinstance(re_estimate, bool):
        raise ValueError("'re_estimate' must be logical.")
    if (not center) and standardize:
        raise ValueError("The option 'center=False, standardize=True' is not available.")
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(kmax, int):
        raise ValueError("'kmax' must be an integer.")
    if kmax > min(X.shape):
        raise ValueError("'kmax' must be smaller than the size of X.")

    # Create output dictionary
    out = {}

    out['X'] = X
    out['kmax'] = kmax
    out['center'] = center
    out['standardize'] = standardize
    out['re_estimate'] = re_estimate

    T, N = X.shape
    missing = np.isnan(X)
    goodT = np.sum(np.isnan(X), axis=1) == 0
    goodN = np.sum(np.isnan(X), axis=0) == 0
    T1 = np.sum(goodT)
    N1 = np.sum(goodN)
    mu1 = np.tile(np.nanmean(X, axis=0), (T, 1))
    sd1 = np.tile(np.nanstd(X, axis=0), (T, 1))

    if center and standardize:
        # demean and standardize
        XT = (X[:, goodN] - mu1[:, goodN]) / sd1[:, goodN]
        XN = (X[goodT, :] - mu1[goodT, :]) / sd1[goodT, :]

        bnXT = apc(XT, kmax)
        bnXN = apc(XN, kmax)

        HH = lstsq(bnXN['Lamhat'][:N1, :kmax], bnXT['Lamhat'][:N1, :kmax], rcond=None)[0]
        Lamhat = np.dot(bnXN['Lamhat'], HH)
        Fhat = bnXT['Fhat']
        Dhat = np.diag(bnXT['d'])
        Chat = np.dot(bnXT['Fhat'], Lamhat.T)
        Xhat = X.copy()  # estimated data
        Xhat[missing] = Chat[missing] * sd1[missing] + mu1[missing]

        if re_estimate:
            mu_hat = np.tile(np.nanmean(Xhat, axis=0), (T, 1))
            sd_hat = np.tile(np.nanstd(Xhat, axis=0), (T, 1))
            Xhat_scaled = (Xhat - mu_hat) / sd_hat

            reest = apc(Xhat_scaled, kmax)
            out['Fhat'] = reest['Fhat']
            out['Lamhat'] = reest['Lamhat']
            out['Dhat'] = reest['d']
            out['Chat'] = reest['Chat'] * sd_hat + mu_hat
            data = X.copy()
            data[missing] = out['Chat'][missing]
            out['data'] = data
        else:
            out['data'] = Xhat
            out['Fhat'] = Fhat
            out['Lamhat'] = Lamhat
            out['Dhat'] = Dhat
            out['Chat'] = Chat * sd1 + mu1
    elif center and not standardize:
        # demean and not standardize
        XT = (X[:, goodN] - mu1[:, goodN]) 
        XN = (X[goodT, :] - mu1[goodT, :])

        bnXT = apc(XT, kmax)
        bnXN = apc(XN, kmax)

        HH = lstsq(bnXN['Lamhat'][:N1, :kmax], bnXT['Lamhat'][:N1, :kmax], rcond=None)[0]
        Lamhat = np.dot(bnXN['Lamhat'], HH)
        Fhat = bnXT['Fhat']
        Dhat = np.diag(bnXT['d'])
        Chat = np.dot(bnXT['Fhat'], Lamhat.T)
        Xhat = X.copy()  # estimated data
        Xhat[missing] = Chat[missing] + mu1[missing]

        if re_estimate:
            mu_hat = np.tile(np.nanmean(Xhat, axis=0), (T, 1))
            Xhat_scaled = (Xhat - mu_hat)

            reest = apc(Xhat_scaled, kmax)
            out['Fhat'] = reest['Fhat']
            out['Lamhat'] = reest['Lamhat']
            out['Dhat'] =  reest['Dhat']
            out['Chat'] = reest['Chat'] + mu_hat
            data = X.copy()
            data[missing] = out['Chat'][missing]
            out['data'] = data
        else:
            out['data'] = Xhat
            out['Fhat'] = Fhat
            out['Lamhat'] = Lamhat
            out['Dhat'] = Dhat
            out['Chat'] = Chat + mu1
    else:
        # no demean and not standardize
        XT = X[:, goodN]
        XN = X[goodT, :]

        bnXT = apc(XT, kmax)
        bnXN = apc(XN, kmax)

        HH = lstsq(bnXN['Lamhat'][:N1, :kmax], bnXT['Lamhat'][:N1, :kmax], rcond=None)[0]
        Lamhat = np.dot(bnXN['Lamhat'], HH)
        Fhat = bnXT['Fhat']
        Dhat = np.diag(bnXT['d'])
        Chat = np.dot(bnXT['Fhat'], Lamhat.T)
        Xhat = X.copy()  # estimated data
        Xhat[missing] = Chat[missing]

        if re_estimate:
            reest = apc(Xhat, kmax)
            out['Fhat'] = reest['Fhat']
            out['Lamhat'] = reest['Lamhat']
            out['Dhat'] = reest['Dhat']
            out['Chat'] = reest['Chat']
            data = X.copy()
            data[missing] = out['Chat'][missing]
            out['data'] = data
        else:
            out['data'] = Xhat
            out['Fhat'] = Fhat
            out['Lamhat'] = Lamhat
            out['Dhat'] = Dhat
            out['Chat'] = Chat


    out['ehat'] = out['data'] - out['Chat']

    return out