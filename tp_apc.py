# Code is coverted by Sabuhi Khalili from R to Python using https://rdrr.io/github/cykbennie/fbi/src/R/tp_apc.R
#' @author Yankang (Bennie) Chen <yankang.chen@@yale.edu>
#' @author Serena Ng <serena.ng@@columbia.edu>
#' @author Jushan Bai <jushan.bai@@columbia.edu>
#'
#' @references
#' Ercument Cahan, Jushan Bai, and Serena Ng (2021),
#' \emph{Factor-Based Imputation of Missing Values and Covariances in Panel Data of Large Dimensions}.
#' \url{https://arxiv.org/abs/2103.03045}

import numpy as np
from numpy.linalg import pinv, solve
from pca import apc


def tp_apc(X, kmax, center=False, standardize=False, re_estimate=True):
  
    if not isinstance(center, bool):
        raise ValueError("'center' must be boolean.")
    if not isinstance(standardize, bool):
        raise ValueError("'standardize' must be boolean.")
    if not isinstance(re_estimate, bool):
        raise ValueError("'re_estimate' must be boolean.")
    if (not center) and standardize:
        raise ValueError("The option 'center = False, standardize = True' is not available.")
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if not isinstance(kmax, int):
        raise ValueError("'kmax' must be an integer.")
    if kmax > min(X.shape):
        raise ValueError("'kmax' must be smaller than the size of X.")
        
    out = {}
    out['X'] = X
    out['kmax'] = kmax
    out['center'] = center
    out['standardize'] = standardize
    out['re_estimate'] = re_estimate

    T, N = X.shape
    rownames = np.arange(1, T+1)
    colnames = np.arange(1, N+1)

    missing = np.isnan(X)
    goodT = np.sum(np.isnan(X), axis=1) == 0
    goodN = np.sum(np.isnan(X), axis=0) == 0
    T1 = np.sum(goodT)
    N1 = np.sum(goodN)
    mu1 = np.tile(np.nanmean(X, axis=0), (T, 1))
    sd1 = np.tile(np.nanstd(X, axis=0, ddof=1), (T, 1))

    if center and standardize:
        # demean and standardize
        XT = (X[:, goodN] - mu1[:, goodN]) / sd1[:, goodN]
        XN = (X - mu1) / sd1

        bnXT = apc(XT, kmax)
        Fhat = bnXT['Fhat']
        Dhat = np.diag(bnXT['d'])
        Lamhat = np.zeros((N, kmax))

        for i in range(N):
            goodTi = missing[:, i] == False
            Fn1 = Fhat[goodTi, :]
            Reg = Fn1
            P = pinv(Reg.T.dot(Reg)).dot(Reg.T)
            Lamhat[i, :] = P.dot(XN[goodTi, i])

        Chat = Fhat.dot(Lamhat.T)
        Xhat = X.copy()  # estimated data
        Xhat[missing] = Chat[missing] * sd1[missing] + mu1[missing]

        if re_estimate:
            mu_hat = np.tile(np.mean(Xhat, axis=0), (T, 1))
            sd_hat = np.tile(np.std(Xhat, axis=0), (T, 1))
            Xhat_scaled = (Xhat - mu_hat) / sd_hat

            reest = apc(Xhat_scaled, kmax)
            out['Fhat'] = reest['Fhat']
            out['Lamhat'] = reest['Lamhat']
            out['Dhat'] = np.diag(reest['d'])
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

    elif center and (not standardize):
        # only demean, do not standardize

        XT = X[:, goodN] - mu1[:, goodN]
        XN = X - mu1

        bnXT = apc(XT, kmax)
        Fhat = bnXT['Fhat']
        Dhat = np.diag(bnXT['d'])
        Lamhat = np.zeros((N, kmax))

        for i in range(N):
            goodTi = missing[:, i] == False
            Fn1 = Fhat[goodTi, :]
            Reg = Fn1
            P = pinv(Reg.T.dot(Reg)).dot(Reg.T)
            Lamhat[i, :] = P.dot(XN[goodTi, i])

        Chat = Fhat.dot(Lamhat.T)
        Xhat = X.copy()  # estimated data
        Xhat[missing] = Chat[missing] + mu1[missing]

        if re_estimate:
            mu_hat = np.tile(np.mean(Xhat, axis=0), (T, 1))
            Xhat_scaled = Xhat - mu_hat

            reest = apc(Xhat_scaled, kmax)
            out['Fhat'] = reest['Fhat']
            out['Lamhat'] = reest['Lamhat']
            out['Dhat'] = np.diag(reest['d'])
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
        # no demeaning or standardizing

        XT = X[:, goodN]
        XN = X

        bnXT = apc(XT, kmax)
        Fhat = bnXT['Fhat']
        Dhat = np.diag(bnXT['d'])
        Lamhat = np.zeros((N, kmax))

        for i in range(N):
            goodTi = missing[:, i] == False
            Fn1 = Fhat[goodTi, :]
            Reg = Fn1
            P = pinv(Reg.T.dot(Reg)).dot(Reg.T)
            Lamhat[i, :] = P.dot(XN[goodTi, i])

        Chat = Fhat.dot(Lamhat.T)
        Xhat = X.copy()  # estimated data
        Xhat[missing] = Chat[missing]

        if re_estimate:
            Xhat_scaled = Xhat
            reest = apc(Xhat_scaled, kmax)
            out['Fhat'] = reest['Fhat']
            out['Lamhat'] = reest['Lamhat']
            out['Dhat'] = np.diag(reest['d'])
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

    out = {'class':  'tp', **out}
    return out


def se_tp(object, npoints, tpoints, qq, re_estimate=True):
    """
    Compute standard errors for TP factor model.

    Args:
    object: A list object containing the following components:
            Fhat: A T-by-r matrix of estimated common factors.
            Lhat: A N-by-r matrix of estimated factor loadings.
            Chat: A T-by-T matrix of estimated idiosyncratic variances.
            ehat: A N-by-T matrix of residuals.
            Dhat: A r-by-r diagonal matrix of eigenvalues.
            X: A N-by-T matrix of data used to fit the model.
    npoints: A vector of integers indicating the number of common factors to extract.
    tpoints: A vector of integers indicating the time periods to compute standard errors for.
    qq: The number of lags to include in the covariance matrix of idiosyncratic errors.
    re_estimate: Logical. If TRUE, re-estimate the model using X-tilde.

    Returns:
    A list object containing the following components:
    tpoints: The input tpoints argument.
    npoints: The input npoints argument.
    Fhat: A vector of estimated common factors.
    Lhat: A vector of estimated factor loadings.
    Chat: A vector of estimated idiosyncratic variances.
    SigmaC: A vector of estimated standard errors for Chat.
    SigmaF: A vector of estimated standard errors for the first common factor.
    SigmaL: A vector of estimated standard errors for the first factor loading.
    """
    

    if 'tp' not in object:
        raise ValueError("Object must be of class 'tp', i.e. the output of tp_apc.")

    Fhat = object["Fhat"]
    Lhat = object["Lamhat"]
    Chat = object["Chat"]
    ehat = object["ehat"]
    Dhat = object["Dhat"]
    T = Fhat.shape[0]
    r = Fhat.shape[1]
    N = Lhat.shape[0]
    missingX = np.isnan(object["X"])   # 1 not observed; 0 observed
    goodN = np.where(np.sum(missingX, axis=0) == 0)[0]   # goodN: no missing for any period
    No = len(goodN)

    out = {
        "tpoints": tpoints,
        "npoints": npoints,
        "Fhat": [],
        "Lhat": [],
        "Chat": [],
        "SigmaC": [],
        "SigmaF": [],
        "SigmaL": []
    }


    if re_estimate:
        for ipoints, tt in enumerate(tpoints):
            ii = npoints[ipoints]
            N_tt = N - np.sum(missingX[tt,])
            T_ii = T - np.sum(missingX[:,ii])
            obsi_tt = np.arange(N)[~missingX[tt,:]]
            obst_ii = np.arange(T)[~missingX[:,ii]]
            notobsi_tt = np.argwhere(missingX[tt,:]).ravel()
            notobst_ii = np.argwhere(missingX[:,ii]).ravel()

            var_Lhato = Lhat[obsi_tt,:].T @ Lhat[obsi_tt,:] / No
            var_Lhatm = Lhat[notobsi_tt,:].T @ Lhat[notobsi_tt,:] / No
            A_Lam = var_Lhatm @ solve(var_Lhato)

            if ii<No:
                B_Lam = (np.eye(r) + A_Lam) * (N_tt/N)
            else:
                B_Lam = np.eye(r) * (N_tt/N)

            Lhat_ii = Lhat[ii,:].reshape((1,r))
            LEhat_t = np.tile(ehat[tt,obsi_tt].reshape((-1,1)), r) * Lhat[obsi_tt,:]
            var_Lehat_t = B_Lam @ LEhat_t.T @ LEhat_t @ B_Lam.T / N_tt
            var_Lhat = Lhat.T @ Lhat / N

            Fhat_tt = Fhat[tt,:].reshape((1,r))
            FEhat_i = np.tile(ehat[obst_ii,ii].reshape((-1,1)), r) * Fhat[obst_ii,:]
            var_Fehat_i = (FEhat_i.T @ FEhat_i) / T_ii

            for k in range(1, qq + 1):
                var_FEhat_i += FEhat_i[k:,:].T @ FEhat_i[:-(k),:]

            V0 = (N_tt/N)**2 * Lhat_ii @ solve(var_Lhat) @ var_Lehat_t @ solve(var_Lhat) @ Lhat_ii.T
            W0 = Fhat_tt @ var_FEhat_i @ Fhat_tt.T

            out['Fhat'].append(Fhat_tt[0, 0])   # only save the first factor
            out['Lhat'].append(Lhat_ii[0, 0])   # only save the first loading
            out['Chat'].append(Chat[tt, ii])
            out['SigmaC'].append(V0/N_tt + W0/T_ii)
            temp_F = np.diag(solve(Dhat) @ var_Lehat_t @ solve(Dhat))
            temp_L = np.diag(var_FEhat_i)
            out['SigmaF'].append(temp_F[0])
            out['SigmaL'].append(temp_L[0])
    else:
         # first pass estimation
        for ipoints in range(len(tpoints)):
            ii = npoints[ipoints]
            tt = tpoints[ipoints]
            N_tt = N - np.sum(missingX[tt, ])
            T_ii = T - np.sum(missingX[:, ii])
            obsi_tt = np.where(missingX[tt, ] == 0)[0]
            obst_ii = np.where(missingX[:, ii] == 0)[0]

            Lhat_ii = Lhat[ii, ].reshape(1, -1)
            LEhat_t = np.tile(ehat[tt, obsi_tt].reshape(-1, 1), (1, r)) * Lhat[obsi_tt, ]
            var_Lehat_t = LEhat_t.T @ LEhat_t / No
            var_Lhat = Lhat.T @ Lhat / N

            Fhat_tt = Fhat[tt, ].reshape(1, -1)
            FEhat_i = np.tile(ehat[obst_ii, ii].reshape(-1, 1), (1, r)) * Fhat[obst_ii, ]
            var_Fehat_i = FEhat_i.T @ FEhat_i / T_ii

            for k in range(qq):
                var_Fehat_i += FEhat_i[(k + 1):, ].T @ FEhat_i[:(FEhat_i.shape[0] - k), ]

            # variance of common component
            V0 = Lhat_ii @ np.linalg.inv(var_Lhat) @ var_Lehat_t @ np.linalg.inv(var_Lhat) @ Lhat_ii.T
            W0 = Fhat_tt @ var_Fehat_i @ Fhat_tt.T

            # Store output
            out["Fhat"].append(Fhat_tt[0, 0])  # only save the first factor
            out["Lhat"].append(Lhat_ii[0, 0])  # only save the first loading
            out["Chat"].append(Chat[tt, ii])
            out["SigmaC"].append(V0 / No + W0 / T_ii)
            temp_F = np.diag(np.linalg.inv(Dhat) @ var_Lehat_t @ np.linalg.inv(Dhat))
            temp_L = np.diag(var_Fehat_i)
            out["SigmaF"].append(temp_F[0])
            out["SigmaL"].append(temp_L[0])

    return out