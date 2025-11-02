import numpy as np
import pandas as pd
import os
from missforest import MissForest
import miceforest as mf

from sklearn.linear_model import LinearRegression
from tp_apc import tp_apc
from tw_apc import tw_apc
from pca import apc,pca
from joblib import Parallel, delayed
import argparse 
import warnings
import time

from auto_enc import LinearAutoencoder

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# from hi_vae import VAE, DiagonalEncoder, GaussianDecoder

warnings.filterwarnings('ignore') 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from utilities import missing_random, estimate_lambda, train_test_split, rms_difference, res_overlay, MinMaxScaler, NoScaler, StandardScaler, CenterScaler

np.random.seed(12345)



parser = argparse.ArgumentParser(description='Command-line interface for setting the model variable.')

# Add an argument for the model
# tp_apc, tw_apc, auto_enc, auto_enc_masked, auto_enc_masked_reg, pca_weighted, pca, hi_vae, missforest, miceforest, zero, complete
parser.add_argument('--model', type=str, default='miceforest', help='Specify the model type (default: auto_enc)')
parser.add_argument('--center', action='store_true')
parser.add_argument('--standardize', action='store_true')

# Parse the command-line arguments
args = parser.parse_args()



model = args.model
standardize = args.standardize
center = args.center
re_estimate = True
do_sum_stats = True

scaler_type = int(standardize) + int(center)

## scaling for autoencoders
# scaler =NoScaler()
# if standardize:
#     scaler = StandardScaler()
# elif center:
#     scaler = CenterScaler()
D_r = np.array([[1,0],[0,0.5]])

m= 2147483648 #sys.maxsize for 32 bit

k = D_r.shape[0]

T = 200
N = 500
missing_percent = 0.30
max_iter = 1

factors = np.empty(shape=[T, k], dtype=float)
factors[:] = np.random.normal(loc=0, scale=np.sqrt(np.diag(D_r)), size=[T, k])

loadings = np.empty(shape=[N, k], dtype=float)
loadings[:] = np.random.normal(loc=0, scale=np.sqrt(np.diag(D_r)), size=[N, k])
C = factors@np.transpose(loadings)

## Case 1, missing 80x80
# size of bal
T_o = 120
N_o = 60



# T_hat = T-T_o
# N_hat = N-N_o


C_bal = C[0:T_o, 0:N_o]
C_tall = C[0:T, 0:N_o]
C_wide = C[0:T_o, 0:N]
C_miss = C[T_o:T, N_o:N]


# Randomly choose missing_percent of the cells to set to missing
num_missing = int(missing_percent * T * N)
missing_indices = np.random.choice(T * N, num_missing, replace=False)


# Generate non-missing indices
all_indices = np.arange(T * N)
non_missing_indices = np.setdiff1d(all_indices, missing_indices)

# Randomly choose two points from each set of indices
missing_points = list(np.random.choice(missing_indices, 2, replace=False))
non_missing_points = list(np.random.choice(non_missing_indices, 2, replace=False))
if do_sum_stats:
    selected_points = missing_points + non_missing_points
    C_locs = []
    selected_rows, selected_cols = np.unravel_index(selected_points, (T, N))

    for i in range(len(selected_points)):
        C_locs.append(C[selected_rows[i], selected_cols[i]])



rms_bal = np.zeros((max_iter, 1))
rms_tall = np.zeros((max_iter, 1))
rms_wide = np.zeros((max_iter, 1))
rms_miss = np.zeros((max_iter, 1))
rms_chat = np.zeros((max_iter, 1))
rms_data = np.zeros((max_iter, 1))
rms_cov = np.zeros((max_iter, 1))
rms_cov_1 = np.zeros((max_iter, 1))

if do_sum_stats:
    Chats_loc_1 = np.zeros((max_iter, 1))
    Chats_loc_2 = np.zeros((max_iter, 1))
    Chats_loc_3 = np.zeros((max_iter, 1))
    Chats_loc_4 = np.zeros((max_iter, 1))

generated_data_points = np.empty(0)
total_line_time = 0


print(f"Starting the job with following parameters: max_iter:{max_iter}, T:{T}, N:{N}, miss perc: {missing_percent}, args:{args}, modul:{scaler_type}")
start = time.time()
def process_iteration(i, rn,args):
    tf.random.set_seed(rn)
    np.random.seed(rn)
    scaler =NoScaler()
    if standardize:
        scaler = StandardScaler()
    elif center:
        scaler = CenterScaler()

    model = args.model
    error_term_train = np.empty(shape=[T, N],  dtype=float)
    error_term_train[:] = np.random.normal(loc=0, scale=np.sqrt(2.5), size=[T, N])
    # print(error_term_train[0:1,0:2])

    X_synth = C+error_term_train

    X_missing =X_synth.copy()

    # centered_data = CenterScaler().fit_transform(X_synth) # probably unnecessary
    # Calculate the covariance matrix
    # true_covariance_matrix = np.cov(X_synth, rowvar=False)

    # error_term_test = np.random.normal(loc=0, scale=np.sqrt(2.5), size=[T, N])

    # X_synth_test = C+error_term_test

    # X_missing_test =X_synth_test.copy()

    # miss_block = missing_random(data=X_synth[-T_hat:, -N_hat:], perc_missing=missing_percent, missing_indices=missing_indices)
    # X_missing[-(T-T_o):, -(N-N_o):] = None #miss_block  

    X_missing = missing_random(data=X_missing, perc_missing=missing_percent, missing_indices=missing_indices)

    nan_mask = np.isnan(X_missing)

    # miss_block_test = missing_random(data=X_synth_test[-T_hat:, -N_hat:], perc_missing=missing_percent, missing_indices=missing_indices)
    # X_missing_test[-(T-T_o):, -(N-N_o):] = miss_block_test

    if model == 'tp_apc':
        res = tp_apc(X=X_missing, kmax=k, center=center, standardize=standardize, re_estimate=re_estimate)
        Chat = res['Chat']
        X_missing = res['data']
    elif model == 'tw_apc':
        res = tw_apc(X=X_missing, kmax=k, center=center, standardize=standardize, re_estimate=re_estimate)
        Chat = res['Chat']
        X_missing = res['data']
    elif model == 'auto_enc':
        scaled_train_data = scaler.fit_transform(X_missing)
        autoencoder = LinearAutoencoder(input_dim=N, hidden_layers=[], hidden_dim=k)
        autoencoder.compile(optimizer="adam")
        callback = EarlyStopping(monitor='mse_loss',
                                 patience=1000, min_delta=0.00001, mode='min')
        autoencoder.fit(scaled_train_data, scaled_train_data,
                        epochs=10000,
                        batch_size=T,
                        shuffle=False, verbose=False, callbacks=[callback])
        # res = tp_apc(X=scaled_train_data, kmax=k, center=False, standardize=False, re_estimate=True)['data']
        x_pr = scaler.inverse_transform(autoencoder.predict(np.nan_to_num(scaled_train_data)))
        X_missing[nan_mask] = x_pr[nan_mask]
        # Chat = apc(X_missing, kmax=k)['Chat']   
        scaled_imputed_data = scaler.fit_transform(X_missing)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])   


    elif model == 'auto_enc_masked':
        scaled_train_data = scaler.fit_transform(X_missing)
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
        X_missing[nan_mask] = x_pr[nan_mask]
        # Chat = apc(X_missing, kmax=k)['Chat']
        
        scaled_imputed_data = scaler.fit_transform(X_missing)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])   


    elif model == 'auto_enc_masked_reg':
        scaled_train_data = scaler.fit_transform(X_missing)
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

        x_pr = fhat_autoenc_reest@w_hat
        # x_pr = scaler.inverse_transform(autoencoder.predict(np.nan_to_num(scaled_train_data)))
        X_missing[nan_mask] = scaler.inverse_transform(x_pr)[nan_mask]
        # Chat = apc(X_missing, kmax=k)['Chat']  
        
        scaled_imputed_data = scaler.fit_transform(X_missing)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])   
  
 
    elif model == 'pca_weighted':
        # Xiong & Pelger (2021) weighted PCA
        scaled_train_data = scaler.fit_transform(X_missing)
        w_hat, cov_mat_hat = estimate_lambda(scaled_train_data, k)
        w_hat = w_hat.T
        # Initialize an array to store the predicted values of fhat_autoenc
        fhat_pca = np.zeros((T, k))
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
            fhat_pca[jj, :] = model.coef_

        Chat = scaler.inverse_transform(fhat_pca@w_hat)
        # x_pr = scaler.inverse_transform(autoencoder.predict(np.nan_to_num(scaled_train_data)))
        X_missing[nan_mask] = Chat[nan_mask]
        
        scaled_imputed_data = scaler.fit_transform(X_missing)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])
        X_missing[nan_mask] = Chat[nan_mask]


    elif model == "missforest":
        scaled_train_data = scaler.fit_transform(X_missing)
        imputer = MissForest()

        x_pr = scaler.inverse_transform(np.array(imputer.fit_transform(pd.DataFrame(scaled_train_data))))

        X_missing[nan_mask] = x_pr[nan_mask]
        # Chat = apc(X_missing, kmax=k)['Chat']

        scaled_imputed_data = scaler.fit_transform(X_missing)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])

    elif model == "miceforest":
        scaled_train_data = scaler.fit_transform(X_missing)
        x_pr = pd.DataFrame(scaled_train_data, columns=[f'col_{i}' for i in range(scaled_train_data.shape[1])])
        kds = mf.ImputationKernel(x_pr)

        # Run the MICE algorithm for 2 iterations
        kds.mice(5)

        # Return the completed dataset.
        x_pr = np.array(kds.complete_data())
        X_missing[nan_mask] = x_pr[nan_mask]
        scaled_imputed_data = scaler.fit_transform(X_missing)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])

    elif model=='zero':
        scaled_train_data = scaler.fit_transform(X_missing)
        Chat = scaler.inverse_transform(apc(np.nan_to_num(scaled_train_data), kmax=k)['Chat'])
        X_missing[nan_mask] = Chat[nan_mask]
        scaled_imputed_data = scaler.fit_transform(X_missing)
        Chat = scaler.inverse_transform(apc(scaled_imputed_data, kmax=k)['Chat'])   
    elif model == 'pca':
        res = pca(np.nan_to_num(X_missing), kmax=k)
        Chat = res['Chat']
        X_missing[nan_mask] = Chat[nan_mask]
    elif model=='complete':
        scaled_train_data = scaler.fit_transform(X_synth)
        Chat = scaler.inverse_transform(apc(scaled_train_data, kmax=k)['Chat'])
        X_missing = X_synth


    Chat_bal  = Chat[0:T_o, 0:N_o]
    Chat_tall = Chat[0:T, 0:N_o]
    Chat_wide = Chat[0:T_o, 0:N]
    Chat_miss = Chat[T_o:T, N_o:N]

    # Print at every step
    print(f"Processing iteration {i}")
    if not do_sum_stats:
        return [rms_difference(C, Chat), rms_difference(C_bal, Chat_bal), rms_difference(C_tall, Chat_tall), 
                rms_difference(C_wide, Chat_wide), rms_difference(C_miss, Chat_miss), rms_difference(X_synth, X_missing)]
    else:
        return [rms_difference(C, Chat), rms_difference(C_bal, Chat_bal), rms_difference(C_tall, Chat_tall), 
                rms_difference(C_wide, Chat_wide), rms_difference(C_miss, Chat_miss), rms_difference(X_synth, X_missing), 
                Chat]
        

with Parallel(n_jobs=1, prefer="processes", verbose=1) as parallel:
    results = parallel(delayed(process_iteration)(i, np.random.randint(0, m),  args=args) for i in range(max_iter))

# results = []
# for i in range(max_iter):
#     results.append(process_iteration(i))

for idx, res in enumerate(results):
    rms_chat[idx] = res[0]
    rms_bal[idx] = res[1]
    rms_tall[idx] = res[2]
    rms_wide[idx] = res[3]
    rms_miss[idx] = res[4]
    rms_data[idx] = res[5]

    if do_sum_stats:
        Chats_loc_1[idx] = res[6][selected_rows[0], selected_cols[0]]
        Chats_loc_2[idx] = res[6][selected_rows[1], selected_cols[1]]
        Chats_loc_3[idx] = res[6][selected_rows[2], selected_cols[2]]
        Chats_loc_4[idx] = res[6][selected_rows[3], selected_cols[3]]




rms_bal_mean = np.mean(rms_bal)
rms_tall_mean = np.mean(rms_tall)
rms_wide_mean = np.mean(rms_wide)
rms_miss_mean = np.mean(rms_miss)
rms_chat_mean = np.mean(rms_chat)
rms_data_mean = np.mean(rms_data)



# generated_data_points = np.append(generated_data_points, np.reshape(X_missing_test[-(T-T_o):, -(N-N_o):], -1))

# plt.hist(generated_data_points, bins='auto', density=True)
# # plt.show()
# print(rms_miss)
print(rms_chat_mean)
print(rms_tall_mean)
print(rms_wide_mean)
print(rms_bal_mean)
print(rms_miss_mean)
print(rms_data_mean)


with pd.ExcelWriter(f'output/mcar/table1_mcar_{model}_{missing_percent}_modul_{scaler_type}.xlsx') as writer:
        row_names = [f'rms_chat', 'rms_data']  
        values = [rms_chat_mean, rms_data_mean]
        pd.DataFrame(values, index=row_names).to_excel(writer, sheet_name=f'table_1')


# save table 2 finite sample properties
def save_fin_smpl_table(C_at_locs:list,Chat_at_locs:list, filename:str):
    with pd.ExcelWriter(f'output/mcar/table2_{filename}.xlsx') as writer:
            for i in range(len(C_at_locs)):
                C_at_loc = C_at_locs[i]
                Chat_at_loc = Chat_at_locs[i]
                row_names = [f'C_loc_{i+1}', 'mean', 'sd', 'ase', 'q05', 'q95', 'coverage']  
                ase = np.mean(np.sqrt(np.square(C_at_loc - np.array(Chat_at_loc))))
                standardized_Chat = StandardScaler().fit_transform(Chat_at_loc)
                # Estimate the coverage  -  the number of values within 95% of standard normal distribution
                count_within_range = np.sum((standardized_Chat >= -1.65) & (standardized_Chat <= 1.65))
                total_elements = standardized_Chat.size
                percentage_within_range = (count_within_range / total_elements)
                values = [C_at_loc, np.mean(Chat_at_loc), np.std(Chat_at_loc), ase, 
                np.percentile(standardized_Chat, 5), np.percentile(standardized_Chat, 95), percentage_within_range]
                pd.DataFrame(values, index=row_names).to_excel(writer, sheet_name=f'c_{i+1}')
   
if do_sum_stats:
    save_fin_smpl_table(C_at_locs=C_locs, Chat_at_locs=[Chats_loc_1, Chats_loc_2, Chats_loc_3, Chats_loc_4], 
                        filename=f'mcar_{model}_{missing_percent}_modul_{scaler_type}')

print(f"Spent time: {time.time()-start:.2f} seconds")

