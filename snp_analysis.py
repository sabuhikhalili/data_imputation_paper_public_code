import numpy as np
import pandas as pd
import os

from sklearn.linear_model import LinearRegression
from tp_apc import tp_apc
from tw_apc import tw_apc
from pca import apc
from auto_enc import LinearAutoencoder, SimpleAutoencoder

from joblib import Parallel, delayed
import argparse 
import warnings

from missforest import MissForest
import miceforest as mf

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore') 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.callbacks import EarlyStopping


from utilities import rms_difference, NoScaler, StandardScaler, CenterScaler


np.random.seed(12345)


cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'data')
file_path = os.path.join(data_dir, 'snp_balanced_monthly.xlsx')

raw_data = pd.ExcelFile(file_path).parse(0, index_col=0)
raw_data = raw_data.sort_index(ascending=False)

raw_data.iloc[:,:] = np.log(raw_data)
raw_data = raw_data.diff(-1)*100



raw_data = raw_data.dropna() # this drops only one row of missing values due to diff

# this is used to set different random seeds in parallel processing, fully technical stuff, nothing to do with the model
# you can ignore this when you use single application without parallel processing
m= 2147483648 #sys.maxsize for 32 bit
max_iter = 1000

parser = argparse.ArgumentParser(description='Command-line interface for setting the model variable.')

# Add an argument for the model
parser.add_argument('--model', type=str, default='tp_apc', help='Specify the model type')
parser.add_argument('--k', type=int, default='2', help='Specify the dimension of bottleneck/pca factors')
parser.add_argument('--center', action='store_true')
parser.add_argument('--standardize', action='store_true')

# Parse the command-line arguments
args = parser.parse_args()

args.T = raw_data.shape[0]
args.N = raw_data.shape[1]

observed_data = raw_data

pca_results = apc(observed_data, kmax=20)

C =  apc(observed_data, kmax=args.k)['Chat']

n_max = args.k+10
explained_variance = pca_results['explained_variance_ratio'][0:n_max]
# comment this out to see explained variance plot
# plt.bar(range(1, len(explained_variance) + 1), explained_variance*100)
# plt.xlabel('Rank of Principal Components')
# plt.ylabel('Explained Variance (%)')
# plt.title('Explained Variance by Principal Components')
#
# plt.show()

rms_bal = np.zeros((max_iter, 1))
rms_tall = np.zeros((max_iter, 1))
rms_wide = np.zeros((max_iter, 1))
rms_miss = np.zeros((max_iter, 1))
rms_chat = np.zeros((max_iter, 1))
rms_data = np.zeros((max_iter, 1))

def process_iteration(iter, rn, args):
    tf.random.set_seed(rn)
    np.random.seed(rn)
    model = args.model
    k = args.k
    T = args.T
    N = args.N
    missing_perc = 0.20
    T_o = T - int(T*missing_perc)
    N_o =  N - int(N*missing_perc)
    scaler =NoScaler()
    if args.standardize:
        scaler = StandardScaler()
    elif args.center:
        scaler = CenterScaler()


    missing_df = observed_data.copy()

    # shuffle columns
    column_indices = np.arange(N)
    np.random.shuffle(column_indices)
    row_indices = np.arange(T)
    np.random.shuffle(row_indices)

    missing_cols = column_indices[N_o:]
    missing_rows = row_indices[T_o:]

    # set miss block to NaN
    for j in missing_cols:
        for i in missing_rows:
            missing_df.iloc[i, j] = None

    missing_data = missing_df.to_numpy()


    nan_mask = np.isnan(missing_data)

    scaled_train_data = scaler.fit_transform(missing_data)

    C_bal = C[0:T_o, 0:N_o]
    C_tall = C[0:T, 0:N_o]
    C_wide = C[0:T_o, 0:N]
    C_miss = C[T_o:T, N_o:N]


    imputed_data = missing_data.copy()
    if model == 'tp_apc':
        res = tp_apc(X=missing_data, kmax=k, center=args.center, standardize=args.standardize, re_estimate=True)
        Chat = res['Chat']
        imputed_data = res['data']
    elif model == 'tw_apc':
        res = tw_apc(X=missing_data, kmax=k, center=args.center, standardize=args.standardize, re_estimate=True)
        Chat = res['Chat']
        imputed_data = res['data']
    elif model == 'auto_enc':
        # linear autoencoder without masked loss function
        autoencoder = LinearAutoencoder(input_dim=N, hidden_layers=[], hidden_dim=k)
        autoencoder.compile(optimizer="adam")
        callback = EarlyStopping(monitor='mse_loss',
                                 patience=1000, min_delta=0.00001, mode='min')
        autoencoder.fit(scaled_train_data, scaled_train_data,
                        epochs=10000,
                        batch_size=T,
                        shuffle=False, verbose=False, callbacks=[callback])
        x_pr = scaler.inverse_transform(autoencoder.predict(np.nan_to_num(scaled_train_data)))
        imputed_data[nan_mask] = x_pr[nan_mask]
        scaled_imputed_data = scaler.fit_transform(imputed_data)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])   

    elif model == 'auto_enc_masked':
        # linear autoencoder with masked loss function
        autoencoder = LinearAutoencoder(input_dim=N, hidden_layers=[], hidden_dim=k)
        autoencoder.enable_masked_loss_function()
        autoencoder.compile(optimizer="adam")
        callback = EarlyStopping(monitor='mse_loss',
                                 patience=1000, min_delta=0.00001, mode='min')
        autoencoder.fit(scaled_train_data, scaled_train_data,
                        epochs=10000,
                        batch_size=T,
                        shuffle=False, verbose=False, callbacks=[callback])
        # res = tp_apc(X=scaled_train_data, kmax=k, center=False, standardize=False, re_estimate=True)['data']
        x_pr = scaler.inverse_transform(autoencoder.predict(np.nan_to_num(scaled_train_data)))
        imputed_data[nan_mask] = x_pr[nan_mask]
        scaled_imputed_data = scaler.fit_transform(imputed_data)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])

    elif model == 'auto_enc_non_linear_masked':
        # linear autoencoder with masked loss function
        autoencoder = SimpleAutoencoder(input_dim=N, hidden_layers=[int(N/2), int(N/4)], hidden_dim=k, use_bias=True)
        autoencoder.enable_masked_loss_function()
        autoencoder.compile(optimizer="adam")
        callback = EarlyStopping(monitor='mse_loss',
                                 patience=1000, min_delta=0.00001, mode='min')
        autoencoder.fit(scaled_train_data, scaled_train_data,
                        epochs=10000,
                        batch_size=T,
                        shuffle=False, verbose=False, callbacks=[callback])
        # res = tp_apc(X=scaled_train_data, kmax=k, center=False, standardize=False, re_estimate=True)['data']
        x_pr = scaler.inverse_transform(autoencoder.predict(np.nan_to_num(scaled_train_data)))
        imputed_data[nan_mask] = x_pr[nan_mask]
        scaled_imputed_data = scaler.fit_transform(imputed_data)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])

    elif model == 'auto_enc_masked_reg':
        # 2SLinear autoencoder with masked loss function and regression
        autoencoder = LinearAutoencoder(input_dim=N, hidden_layers=[], hidden_dim=k)
        autoencoder.enable_masked_loss_function()
        autoencoder.compile(optimizer="adam")
        callback = EarlyStopping(monitor='mse_loss',
                                 patience=1000, min_delta=0.00001, mode='min')
        autoencoder.fit(scaled_train_data, scaled_train_data,
                        epochs=10000,
                        batch_size=T,
                        shuffle=False, verbose=False, callbacks=[callback])
        # predict fhat using regression
        w_hat = autoencoder.get_weights()[1]
        # Initialize an array to store the predicted values of fhat_autoenc
        fhat_autoenc_reest = np.zeros((T, k))
        # Iterate over each row in scaled_train_data
        for jj in range(scaled_train_data.shape[0]):
            # Extract the current row from scaled_train_data
            current_row = scaled_train_data[jj, :].reshape(1, -1)
            current_row = current_row[~np.isnan(current_row)]
            w_hat_observed = w_hat[:, ~np.isnan(scaled_train_data[jj, :])]
            # Perform linear regression to estimate fhat_autoenc without intercept
            model = LinearRegression(fit_intercept=False)
            model.fit(w_hat_observed.T, current_row)
            fhat_autoenc_reest[jj, :] = model.coef_

        x_pr = fhat_autoenc_reest@w_hat
        # x_pr = scaler.inverse_transform(autoencoder.predict(np.nan_to_num(scaled_train_data)))
        imputed_data[nan_mask] = scaler.inverse_transform(x_pr)[nan_mask]
        scaled_imputed_data = scaler.fit_transform(imputed_data)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])   

    elif model == 'auto_enc_masked_reg_pl':
        # 2SLinear autoencoder with masked loss function, regression and reestimation
        autoencoder = LinearAutoencoder(input_dim=N, hidden_layers=[], hidden_dim=k)
        autoencoder.enable_masked_loss_function()
        autoencoder.compile(optimizer="adam")
        callback = EarlyStopping(monitor='mse_loss',
                                 patience=1000, min_delta=0.00001, mode='min')
        autoencoder.fit(scaled_train_data, scaled_train_data,
                        epochs=10000,
                        batch_size=T,
                        shuffle=False, verbose=False, callbacks=[callback])
        # predict fhat using regression
        w_hat = autoencoder.get_weights()[1]
        # Initialize an array to store the predicted values of fhat_autoenc
        fhat_autoenc_reest = np.zeros((T, k))
        # Iterate over each row in scaled_train_data
        for jj in range(scaled_train_data.shape[0]):
            # Extract the current row from scaled_train_data
            current_row = scaled_train_data[jj, :].reshape(1, -1)
            current_row = current_row[~np.isnan(current_row)]
            w_hat_observed = w_hat[:, ~np.isnan(scaled_train_data[jj, :])]
            # Perform linear regression to estimate fhat_autoenc
            # Perform linear regression to estimate fhat_autoenc without intercept
            model = LinearRegression(fit_intercept=False)
            model.fit(w_hat_observed.T, current_row)
            fhat_autoenc_reest[jj, :] = model.coef_

        x_pr = scaler.inverse_transform(fhat_autoenc_reest@w_hat)
        # x_pr = scaler.inverse_transform(autoencoder.predict(np.nan_to_num(scaled_train_data)))
        imputed_data[nan_mask] = x_pr[nan_mask]
        scaled_imputed_data = scaler.fit_transform(imputed_data)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])


    elif model == "missforest":
        imputer = MissForest()
        imputed_data = missing_data.copy()
        x_pr = scaler.inverse_transform(np.array(imputer.fit_transform(pd.DataFrame(scaled_train_data))))

        imputed_data[nan_mask] = x_pr[nan_mask]
        # Chat = apc(X_missing, kmax=k)['Chat']

        scaled_imputed_data = scaler.fit_transform(imputed_data)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])


    elif model == "miceforest":
        x_pr = pd.DataFrame(scaled_train_data, columns=[f'col_{i}' for i in range(scaled_train_data.shape[1])])
        kds = mf.ImputationKernel(x_pr)

        # Run the MICE algorithm for 2 iterations
        kds.mice(5)

        # Return the completed dataset.
        x_pr = np.array(kds.complete_data())

        imputed_data = missing_data.copy()
        imputed_data[nan_mask] = x_pr[nan_mask]

        scaled_imputed_data = scaler.fit_transform(imputed_data)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])


    elif model=='zero':
        # standard pca imputation with zeros for missing values
        Chat = scaler.inverse_transform(apc(np.nan_to_num(scaled_train_data), kmax=k)['Chat'])
        imputed_data[nan_mask] = Chat[nan_mask]
        scaled_imputed_data = scaler.fit_transform(imputed_data)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])


    elif model=='complete':
        scaled_train_data = scaler.fit_transform(observed_data)
        Chat = scaler.inverse_transform(apc(scaled_train_data, kmax=k)['Chat'])
        imputed_data = observed_data
        scaled_imputed_data = scaler.fit_transform(imputed_data)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])

    # save the blocks for reconstruction error calculation
    Chat_bal  = Chat[0:T_o, 0:N_o]
    Chat_tall = Chat[0:T, 0:N_o]
    Chat_wide = Chat[0:T_o, 0:N]
    Chat_miss = Chat[T_o:T, N_o:N]


    # Print at every step
    print(f"Processing iteration {iter}")
    return [rms_difference(C, Chat), rms_difference(C_bal, Chat_bal), rms_difference(C_tall, Chat_tall),
            rms_difference(C_wide, Chat_wide), rms_difference(C_miss, Chat_miss), rms_difference(observed_data, imputed_data),
            Chat]
        
# parallel processing to speed up the iterations
# max number of parallel jobs can be set according to the number of CPU cores available
with Parallel(n_jobs=20, prefer="processes", verbose=1) as parallel:
    results = parallel(delayed(process_iteration)(iter, np.random.randint(0, m), args=args) for iter in range(max_iter))


for idx, res in enumerate(results):
    rms_chat[idx] = res[0]
    rms_bal[idx] = res[1]
    rms_tall[idx] = res[2]
    rms_wide[idx] = res[3]
    rms_miss[idx] = res[4]
    rms_data[idx] = res[5]


rms_bal_mean = np.mean(rms_bal)
rms_tall_mean = np.mean(rms_tall)
rms_wide_mean = np.mean(rms_wide)
rms_miss_mean = np.mean(rms_miss)
rms_chat_mean = np.mean(rms_chat)
rms_data_mean = np.mean(rms_data)


scaler_type = int(args.standardize) + int(args.center)

with pd.ExcelWriter(f'output/snp/table1_snp_blocks_{args.model}_modul_{scaler_type}.xlsx') as writer:
        row_names = ['rms_chat_tall', 'rms_wide_mean', 'rms_bal_mean', 'rms_miss_mean', 'rms_data']  
        values = [rms_tall_mean, rms_wide_mean, rms_bal_mean, rms_miss_mean, rms_data_mean]
        pd.DataFrame(values, index=row_names).to_excel(writer, sheet_name=f'table_1')


