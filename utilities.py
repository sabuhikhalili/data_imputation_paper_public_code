import pandas as pd
import numpy as np
from numpy.random import choice, randn

import tensorflow as tf
from typing import Union
import time

from joblib import Parallel, delayed
from numpy import ndarray


# tf.data.experimental.enable_debug_mode()

def extract_moving_months(df: pd.DataFrame, month_length=24) -> np.array:
    feature_length = len(df.columns)
    sample_size = len(df.index)-month_length
    df.sort_index(ascending=False, inplace=True)
    dataset = np.empty((sample_size,month_length,feature_length))
    for idx in range(len(df.index)-month_length):
        monthly_batch = [df.iloc[idx+i] for i in range(month_length)]
        dataset[idx,:,: ] = monthly_batch
    return dataset

def generate_factor_structure(matrix_size, T, N, transform=None):

    # Create the 8x8 diagonal matrix
    D_r = np.diag([1/i for i in range(1, matrix_size + 1)])


    k = D_r.shape[0]

    factors = np.empty(shape=[T, k], dtype=np.float32)
    factors[:] = np.random.normal(loc=0, scale=np.sqrt(np.diag(D_r)), size=[T, k])

    loadings = np.empty(shape=[N, k], dtype=np.float32)
    loadings[:] = np.random.normal(loc=0, scale=np.sqrt(np.diag(D_r)), size=[N, k])
    C = factors@np.transpose(loadings)
    _C = C.copy()

    if transform is not None:
        _C = transform(C)

    error_term = np.empty(shape=[T, N],  dtype=np.float32)
    error_term[:] = np.random.normal(loc=0, scale=np.sqrt(0.1), size=[T, N])

    true_data = _C + error_term

    return true_data, C, factors, loadings, _C


def read_excel(file_path, sheet_name='Series'):
    raw_excel_file = pd.ExcelFile(file_path)
    return raw_excel_file.parse(sheet_name, index_col=0)

def sigmoid(arr:np.ndarray):
    return 1 / (1 + np.exp(-arr))

def reverse_sigmoid(arr:np.ndarray):
    return np.log(arr / (1 - arr))


def missing_random(data:np.ndarray, perc_missing=0.1, missing_indices=None) -> pd.DataFrame:
    df = data.copy()
    N_hat = df.shape[1]
    T_hat = df.shape[0]
    # Randomly choose 10% of the cells to set to missing
    num_missing = int(perc_missing * T_hat * N_hat)
    if missing_indices is None:
        missing_indices = np.random.choice(T_hat * N_hat, num_missing, replace=False)
    missing_rows, missing_cols = np.unravel_index(missing_indices, (T_hat, N_hat))
    for row, col in zip(missing_rows, missing_cols):
        df[row, col] = np.nan

    # Ensure that at least one cell in each row is non-missing
    # for row in df.index:
    #     if df.loc[row].isnull().all():
    #         # If all cells in row are missing, choose a random column to keep
    #         print('All columns are missing in the row, recovering one of them ')
    #         keep_col = np.random.randint(N)
    #         df.loc[row].iloc[keep_col] = data.loc[row].iloc[keep_col]

    # Create a new DataFrame to store the binary mask
    # mask = pd.DataFrame(index=df.index, columns=df.columns, data=(df.isna().astype(int)))
    return df


def res_overlay(imputed_data, nan_mask, Chat, method=1, S=500, parallel=None):
    """
    Residual Overlay

    res_overlay estimates the covariance and correlation matrix of the unbalanced panel data using the method of residual overlay.

    Args:
        data_imputer (tw or tp): An object of class 'tw' or 'tp', i.e., the output of tw_apc or tp_apc.
        method (int): Integer 1 to 4, indicating which residual overlay method to use. They correspond to the four methods described in the paper.
        S (int): The number of iterations.

    Returns:
        dict: A dictionary with the following elements:
            - method (int): The method of residual overlay.
            - S (int): The number of iterations.
            - cov (ndarray): Estimated covariance matrix.
            - cor (ndarray): Estimated correlation matrix.

    Author (R):
        Yankang (Bennie) Chen <yankang.chen@@yale.edu>
        Serena Ng <serena.ng@@columbia.edu>
        Jushan Bai <jushan.bai@@columbia.edu>

    Author (Python):
        Sabuhi Khalili

    References:
        Cahan, E., Bai, J. and Ng, S. 2019, Factor Based Imputation of Missing Data and Covariance Matrix Estimation. unpublished manuscript, Columbia University
    """

    if parallel is None:
        parallel = Parallel(n_jobs=-1, prefer="threads")
    # Error checking
    if method not in [1, 2, 3, 4]:
        raise ValueError("'method' must be an integer 1, 2, 3, or 4.")
    if not isinstance(S, int):
        raise ValueError("'S' must be an integer.")
  
    # Preparation
    N = imputed_data.shape[1]
    # goodT = np.where(np.sum(nan_mask, axis=1) == 0)[0]
    # goodN = np.where(np.sum(nan_mask, axis=0) == 0)[0]
    badN = np.where(np.sum(nan_mask, axis=0) != 0)[0]
    # badT = np.where(np.sum(nan_mask, axis=1) != 0)[0]
  
    ehat = imputed_data - Chat
    # cov_matrix = np.cov(imputed_data, rowvar=False)
    # cor_matrix = np.corrcoef(imputed_data, rowvar=False)
  
    dum1 = np.zeros((N, N))
    # dum2 = np.zeros((N, N))
    out = {}
  
    # Residual overlay
    
    uhat_obs = ehat[~nan_mask]
    No = len(uhat_obs)
    Nm = len(ehat[nan_mask])

    if method == 1:
        # residual overlay 1
        # iid sampling
        def process_iteration():
            uhat = ehat.copy()
            trial = choice(No, size=Nm)
            uhat[nan_mask] = uhat_obs[trial]
            data = imputed_data.copy()
            data[nan_mask] = imputed_data[nan_mask] + uhat[nan_mask]
            cov_matrix = np.cov(data, rowvar=False)
            return cov_matrix
            # dum2 += np.corrcoef(out['data'], rowvar=False)

        
        cov_matrices = parallel(delayed(process_iteration)() for _ in range(S))

        # Aggregate results (e.g., taking the mean of the covariance matrices)
        dum1 = np.sum(cov_matrices,axis=0)

        out['cov'] = dum1 / S
        # out['cor'] = dum2 / S
  
    elif method == 2:
        # residual overlay 2
        # sampling by columns

        parallel = Parallel(n_jobs=-1, prefer="processes")

        def process_iteration():
            data = imputed_data.copy()
            uhat = ehat.copy()
            for j in range(len(badN)):
                jj = badN[j]
                UU1 = ehat[:, jj]
                UUgood = np.where(~nan_mask[:, jj])[0]
                UUbad = np.where(nan_mask[:, jj])[0]
                UU2 = UU1[UUgood]
                trial = choice(len(UUgood), size=len(UUbad), replace=True)
                uhat[UUbad, jj] = UU2[trial]
                data[UUbad, jj] = data[UUbad, jj] + uhat[UUbad, jj]
            cov_matrix = np.cov(data, rowvar=False)
            return cov_matrix

        cov_matrices = parallel(delayed(process_iteration)() for _ in range(S))

        dum1 = np.sum(cov_matrices,axis=0)

        out['cov'] = dum1 / S


  
    elif method == 3:
        # residual overlay 3
        # iid sampling
        uhat_mu, uhat_sigma = np.mean(uhat_obs), np.std(uhat_obs)
        
        def process_iteration():
            uhat = ehat.copy()
            uhat[nan_mask] = randn(Nm) * uhat_sigma + uhat_mu
            data = imputed_data.copy()
            data[nan_mask] = imputed_data[nan_mask] + uhat[nan_mask]
            cov_matrix = np.cov(data, rowvar=False)
            return cov_matrix

        cov_matrices = parallel(delayed(process_iteration)() for _ in range(S))

        dum1 = np.sum(cov_matrices,axis=0)

        out['cov'] = dum1 / S

    else:
        # residual overlay 4
        # sampling by columns
        parallel = Parallel(n_jobs=-1, prefer="processes", verbose=1)
        
        def process_iteration():
            data = imputed_data.copy()
            uhat = ehat.copy()
            for j in range(len(badN)):
                jj = badN[j]
                UU1 = ehat[:, jj]
                UUgood = np.where(~nan_mask[:, jj])[0]
                UUbad = np.where(nan_mask[:, jj])[0]
                UU2 = UU1[UUgood]
                uu2_mu, uu2_sigma = np.mean(UU2), np.std(UU2)
                nbad = len(UUbad)
                uhat[UUbad, jj] = randn(nbad) * uu2_sigma + uu2_mu
                data[UUbad, jj] = data[UUbad, jj] + uhat[UUbad, jj]
            cov_matrix = np.cov(data, rowvar=False)
            return cov_matrix

        cov_matrices = parallel(delayed(process_iteration)() for _ in range(S))

        dum1 = np.sum(cov_matrices,axis=0)

        out['cov'] = (dum1 / S)
  
    out = {
        'method': method,
        'S': S,
        'cov': out['cov']
        # 'cor': out['cor']
    }
  
    return out




def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    source: https://github.com/scikit-learn/scikit-learn/blob/093e0cf14aff026cca6097e8c42f83b735d26358/sklearn/utils/extmath.py#L769

    Parameters
    ----------
    u : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.

    v : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        The input v should really be called vt to be consistent with scipy's
        output.

    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted : ndarray
        Array u with adjusted columns and the same dimensions as u.

    v_adjusted : ndarray
        Array v with adjusted rows and the same dimensions as v.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v

def train_test_split(data: np.array, train_pct: float=0.8, idx=None):
    N = data.shape[0]
    # Get permutation index
    if idx is None:
        idx = np.random.permutation(N)
    shuffled_data = data[idx]
    n_train = int(N * train_pct)
    train_data = shuffled_data[:n_train]
    test_data = shuffled_data[n_train:]
    return train_data, test_data, idx


def compute_mse(x, y=None, m_mask=None):
    assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
    count = x.shape[0]*x.shape[1]*x.shape[2]
    mse = np.square(x - y)
    if m_mask is not None:
        count = m_mask.sum()
        m_mask = m_mask.astype(bool)
        mse[~m_mask] = 0  # !!! inverse mask, set zeros for observed
    return np.sum(mse) / count

def rms_difference(matrix1, matrix2):
    diff = matrix1 - matrix2
    squared_diff = diff ** 2
    mean_squared_diff = np.mean(squared_diff)
    rms_diff = np.sqrt(mean_squared_diff)
    return rms_diff


class MinMaxScaler():
    """Min Max normalizer.
    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """
    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data


    def fit(self, data):    
        self.mini = np.nanmin(data, 0)
        self.range = np.nanmax(data, 0) - self.mini
        return self
        

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data

    
    def inverse_transform(self, data):
        inverse_data = data*self.range
        inverse_data += self.mini
        return inverse_data



class NoScaler():
    """No scaling as name suggests.
    Args:
    - data: original data

    Returns:
    - norm_data: original data
    """
    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data

    def fit(self, data):
        pass

    def transform(self, data):
        return data.copy()

    
    def inverse_transform(self, data):
        return data.copy()


class StandardScaler():
    """StandardScaler.
    Args:
    - data: original data

    Returns:
    - norm_data: original data
    """
    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data

    def fit(self, data):
        self.mu = np.nanmean(data, axis=0)
        self.sd = np.nanstd(data, axis=0)

    def transform(self, data):
        scaled_data =  (data - self.mu) / (self.sd)
        return scaled_data

    
    def inverse_transform(self, scaled_data):
        data =  (self.sd*scaled_data)+self.mu
        return data



class CenterScaler():
    """CenterScaler.
    Args:
    - data: original data

    Returns:
    - norm_data: original data
    """
    def fit_transform(self, data): 
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data

    def fit(self, data):
        self.mu = np.nanmean(data, axis=0)

    def transform(self, data):
        scaled_data =  data - self.mu
        return scaled_data

    
    def inverse_transform(self, scaled_data):
        data = scaled_data+self.mu
        return data




def compute_mse_loss(x, y, m_mask=None, reduction='MEAN'):
    mse = tf.math.squared_difference(x, y)
    if m_mask is not None:
        # m_mask = tf.cast(m_mask, tf.bool)
        mse = tf.where(~m_mask, mse, tf.zeros_like(mse))  # !!! set zeros for unobserved
        # mse = tf.reduce_sum(mse, axis=-1)
        num_observed = tf.reduce_sum(tf.cast(~m_mask, tf.float32),axis=-1)

        mse = tf.reduce_sum(mse)
        # mse = tf.reduce_sum(mse, axis=-1)/num_observed

        # will return nan for rows with no observable values, set them to zero instead of nan
        # TODO: taking simple mean after this creates a bias. it is only problem in very small datasets
        # nan_values = tf.math.is_nan(mse)
        # Replace NaN values with zeros
        # mse = tf.where(nan_values, tf.zeros_like(mse), mse)

    else:
        mse = tf.reduce_mean(mse, axis=-1)

    # reduction over batches
    if reduction.lower()=='sum':
        mse = tf.reduce_sum(mse)
    else:
        mse = tf.reduce_mean(mse)
    return mse


def write_to_excel(dfs: Union[pd.DataFrame, np.array], filepath: str, axis=0, sheetname='Series', index=None, columns=None):
    """
    Write pandas dataframe to excel file

    Parameters
    df : Pandas dataframe

    filepath : full file path including the name of file
    """
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter', engine_kwargs={"options": {"strings_to_numbers": True}})

    if type(dfs) == np.ndarray:
        dfs = np.swapaxes(dfs, axis, 0)
        for idx, arr in enumerate(dfs):
            df = pd.DataFrame(arr, index=index, columns=columns)
            df.to_excel(writer, sheet_name=sheetname[idx], index=True, header=True)
    elif type(dfs) == list:
         for idx, df in enumerate(dfs):
            df.to_excel(writer, sheet_name=sheetname[idx], index=True, header=True)
    else:
        dfs.to_excel(writer, sheet_name=sheetname, index=True, header=True)
    writer.close()


from numpy.linalg import LinAlgError

def get_cov_mat(char_matrix):
    """
    Calculate the covariance matrix of a partially observed panel using the method from Xiong & Pelger
    Parameters
    ----------
        char_matrix : the panel over which to calculate the covariance N x L
    """
    ct_int = (~np.isnan(char_matrix)).astype(float)
    assert ct_int.shape[0] != 0
    mu = np.nanmean(char_matrix, axis=0).reshape(-1, 1)
    ct = np.nan_to_num(char_matrix)
    temp = ct.T.dot(ct) 
    temp_counts = ct_int.T.dot(ct_int)
    sigma_t = temp / temp_counts - mu @ mu.T
    return sigma_t


def estimate_lambda(data_panel, k):
    """
    Fit the cross-sectional Loadings using the XP method
    Parameters
    ----------
        data_panel : the panel over which to fit the model T x N
        K : the number of cross-sectional factors
    """

    cov_mat = get_cov_mat(data_panel)
    eig_vals, eig_vects = np.linalg.eigh(cov_mat)

    idx = np.abs(eig_vals).argsort()[::-1]
    lmbda = eig_vects[:, idx[:k]] * np.sqrt(eig_vals[idx[:k]].reshape(1, -1))
    # lmbda = eig_vects[:, idx[:k]] * np.sqrt(np.abs(eig_vals)[idx[:k]].reshape(1, -1))

    return lmbda, cov_mat

