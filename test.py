import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.linear_model import LinearRegression

from auto_enc import LinearAutoencoder
import matplotlib.pyplot as plt

from utilities import svd_flip, missing_random, NoScaler, rms_difference

from pca import pca
from tp_apc import tp_apc

import os 
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

mse = MeanSquaredError()
# k = 2


# Set random seed for reproducibility
np.random.seed(12345)
tf.random.set_seed(12345)
# Step 1: Generate random data with a linear factor structure
T = 200
N = 200
T_o = 120
N_o = 60

# Define a linear factor structure
matrix_size = 2

# Create the 8x8 diagonal matrix
D_r = np.diag([1/i for i in range(1, matrix_size + 1)])
k = D_r.shape[0]

factors = np.empty(shape=[T, k], dtype=np.float32)
factors[:] = np.random.normal(loc=0, scale=np.sqrt(np.diag(D_r)), size=[T, k])

loadings = np.empty(shape=[N, k], dtype=np.float32)
loadings[:] = np.random.normal(loc=0, scale=np.sqrt(np.diag(D_r)), size=[N, k])
C = factors@np.transpose(loadings)

error_term = np.empty(shape=[T, N],  dtype=np.float32)
error_term[:] = np.random.normal(loc=0, scale=np.sqrt(2.5), size=[T, N])

true_data = C+error_term

# true_loadings = pd.DataFrame(loadings.T, index=['PC1'])


# df = pd.read_csv("C:/Users/sebuh/OneDrive - ADA University/Desktop/PHD WORK/paper_2nd/Datos_acciones_completos.csv", encoding="UTF-8",header=0, index_col=0)

# diagonal_matrix = np.diag(np.arange(0, T))
# true_data = np.dot(diagonal_matrix, C)
# true_data = df.to_numpy()

# C = pca(true_data, k)['Chat']

C_bal  = C[0:T_o, 0:N_o]
C_tall = C[0:T, 0:N_o]
C_wide = C[0:T_o, 0:N]
C_miss = C[T_o:T, N_o:N]


missing_percent = 0.30
# Calculate the number of missing points based on the predefined percentage
num_missing = int((T) * (N) * missing_percent)

# Randomly choose missing_percent of the cells to set to missing
num_missing = int(missing_percent * T * N)
missing_indices = np.random.choice(T * N, num_missing, replace=False)

# missing_data = missing_random(data=true_data, perc_missing=missing_percent, missing_indices=missing_indices)


missing_data = true_data.copy() #missing_random(data=true_data, perc_missing=0.35)
missing_data[-(T-T_o):, -(N-N_o):] = None
nan_mask = np.isnan(missing_data)

# Step 2: Perform PCA

scaler = NoScaler()
scaled_data = scaler.fit_transform(np.nan_to_num(missing_data))
pca_results = tp_apc(scaled_data, kmax=k)
# explained_variance = pca_results['explained_variance_ratio']

# Principal component loadings
# pca_loadings =pca_results['Lamhat']

# pca_loadings = pd.DataFrame(pca_loadings.T, index=['PC'+str(i+1) for i in range(k)])

pca_reconstructed = missing_data.copy()
pca_reconstructed[nan_mask] = scaler.inverse_transform(pca_results['Chat'])[nan_mask]
mse_pca = mse(pca_reconstructed, true_data)
mse_chat_pca = mse(scaler.inverse_transform(pca_results['Chat']), C)
mse_pca_full_recon = mse(true_data, scaler.inverse_transform(pca_results['Chat']))

Chat_bal  = pca_results['Chat'][0:T_o, 0:N_o]
Chat_tall = pca_results['Chat'][0:T, 0:N_o]
Chat_wide = pca_results['Chat'][0:T_o, 0:N]
Chat_miss = pca_results['Chat'][T_o:T, N_o:N]

print("PCA Chat RMS diff by blocks")
print(rms_difference(C_bal, Chat_bal))
print(rms_difference(C_tall, Chat_tall))
print(rms_difference(C_wide, Chat_wide))
print(rms_difference(C_miss, Chat_miss))


# Step 3: Create a linear autoencoder
scaler = NoScaler()
scaled_train_data = scaler.fit_transform(missing_data)
T,N = scaled_train_data.shape

# input_layer = Input(shape=(N,))
# encoded = Dense(k, activation=None, use_bias=False)(input_layer)
# decoded = Dense(N, activation=None, use_bias=False)(encoded)
# autoencoder = Model(input_layer, decoded)
# autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder = LinearAutoencoder(input_dim=N, hidden_layers=[], hidden_dim=k)
autoencoder.enable_masked_loss_function()
autoencoder.compile(optimizer="adam")
model_history = autoencoder.fit(scaled_train_data, scaled_train_data,
                epochs=10000,
                batch_size=T,
                shuffle=False, verbose=False,validation_data=(missing_data, true_data))
model_history.history['mse_loss']

# Plot 2: MSE Autoencoder
plt.figure(figsize=(12, 5))


if autoencoder.use_mask:
    
    # Plot 1: Difference vs PCA loss
    plt.subplot(1, 2, 1)
    plt.plot(model_history.history['val_mse_loss'] - mse_pca, label='Diff vs pca loss')
    plt.axhline(0, color='red', linestyle='--', label='Zero')
    plt.title('Difference vs PCA Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.legend()
    plt.subplot(1, 2, 2)
    # plt.plot(model_history.history['mse_loss'], label='MSE Autoenc')
    plt.plot(model_history.history['val_mse_loss'], label='MSE Autoenc')
    plt.axhline(mse_pca, color='red', linestyle='--', label='MSE PCA')
    plt.title('MSE Autoencoder')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
else:
    plt.subplot(1, 2, 1)
    plt.plot(model_history.history['mse_loss'] - mse(np.nan_to_num(missing_data), pca_results['Chat']), label='Diff vs pca loss')
    plt.axhline(0, color='red', linestyle='--', label='Zero')
    plt.title('Difference vs PCA Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.legend()
    # Plot 2: MSE Autoencoder
    plt.subplot(1, 2, 2)
    plt.plot(model_history.history['mse_loss'], label='MSE Autoenc')
    plt.axhline(mse(np.nan_to_num(missing_data), pca_results['Chat']), color='red', linestyle='--', label='MSE PCA')
    plt.title('MSE Autoencoder')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()

# # Show the plots
plt.tight_layout()
plt.show()

# res = tp_apc(X=missing_data, kmax=k, center=False, standardize=False, re_estimate=False)['data']

# auto_pred = autoencoder.predict(res)
auto_pred = autoencoder.predict(np.nan_to_num(scaled_train_data))

# predict fhat using regression
fhat_autoenc = autoencoder.encoder(np.nan_to_num(scaled_train_data)).numpy()

w_hat = autoencoder.get_weights()[1]

# Initialize an array to store the predicted values of fhat_autoenc
fhat_autoenc_reest = np.zeros((T, k))
# Iterate over each row in scaled_train_data
for i in range(scaled_train_data.shape[0]):
    # Extract the current row from scaled_train_data
    current_row = scaled_train_data[i, :].reshape(1, -1)
    current_row = current_row[~np.isnan(current_row)]
    w_hat_observed = w_hat[:, ~np.isnan(scaled_train_data[i, :])]
    # Perform linear regression to estimate fhat_autoenc
    # Perform linear regression to estimate fhat_autoenc without intercept
    model = LinearRegression(fit_intercept=False)
    model.fit(w_hat_observed.T, current_row)
    fhat_autoenc_reest[i, :] = model.coef_

auto_pred = fhat_autoenc_reest@w_hat


Chat = pca(auto_pred, kmax=k)['Chat']    
Chat_bal  = Chat[0:T_o, 0:N_o]
Chat_tall = Chat[0:T, 0:N_o]
Chat_wide = Chat[0:T_o, 0:N]
Chat_miss = Chat[T_o:T, N_o:N]

print("AutoEnc RMS diff by blocks")
print(rms_difference(C_bal, Chat_bal))
print(rms_difference(C_tall, Chat_tall))
print(rms_difference(C_wide, Chat_wide))
print(rms_difference(C_miss, Chat_miss))

auto_reconst = missing_data.copy()
auto_reconst[nan_mask] = scaler.inverse_transform(auto_pred)[nan_mask]

# Get the autoencoder weights for the encoding layer
# autoencoder_weights = autoencoder.get_layer(index=6).get_weights()[0]
# auto_reconst = autoencoder.predict(scaled_data,verbose=1, batch_size=T)
mse_auto = mse(true_data, auto_reconst)

mse_chat_auto_full_recon = mse(Chat, C)

mse_auto_full_recon = mse(true_data, scaler.inverse_transform(auto_pred))

Chat = pca(scaler.inverse_transform(auto_reconst), kmax=k)['Chat'] 
mse_chat_auto = mse(Chat, C)

# Step 4: Compare PCA loadings with Autoencoder weights
print(f"PCA results MSE_unscaled:{mse_pca}, MSE_full_recon:{mse_pca_full_recon},  MSE_C: {mse_chat_pca}")
# print(f"PCA results MSE_scaled: {mse_pca}, MSE_unscaled:{mse_pca_original}")

print(f"Autoencoder results MSE_unscaled:{mse_auto},  MSE_full_recon:{mse_auto_full_recon},MSE_C: {mse_chat_auto}, MSE_C_full_recons: {mse_chat_auto_full_recon}")
# print(f"Autoencoder results MSE_scaled: {mse_auto}, MSE_unscaled:{mse_auto_original}")

print("Autoencoder Weights for Encoding Layer:")
# print(autoencoder_weights)

# Visualize PCA explained variance
plt.plot(np.cumsum(explained_variance))
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.show()

fhat_pca = pca_results['Fhat']



# Plot 1: Scatter plot for fhat_pca
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(fhat_pca[:, 0], fhat_pca[:, 1], alpha=0.5)
plt.title('Scatter plot for fhat_pca')
plt.xlabel('Fhat_pca Column 1')
plt.ylabel('Fhat_pca Column 2')

# Plot 2: Scatter plot for fhat_autoenc
plt.subplot(1, 2, 2)
plt.scatter(fhat_autoenc[:, 0], fhat_autoenc[:, 1], alpha=0.5)
plt.title('Scatter plot for fhat_autoenc')
plt.xlabel('Fhat_autoenc Column 1')
plt.ylabel('Fhat_autoenc Column 2')

plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(auto_reconst[:,0], auto_reconst[:,1], color='blue', label='Generated Points (autoenc)')
plt.scatter(auto_reconst[nan_mask[:, 0], 0], auto_reconst[nan_mask[:, 0], 1], color='red', label='Points with NaN')
plt.scatter(true_data[nan_mask[:, 0], 0], true_data[nan_mask[:, 0], 1], color='green', label='Points with NaN - actual values')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Points and Functions')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(pca_reconstructed[:,0], pca_reconstructed[:,1], color='blue', label='Generated Points (pca')
plt.scatter(pca_reconstructed[nan_mask[:, 0], 0], pca_reconstructed[nan_mask[:, 0], 1], color='red', label='Points with NaN')
plt.scatter(true_data[nan_mask[:, 0], 0], true_data[nan_mask[:, 0], 1], color='green', label='Points with NaN - actual values')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Points and Functions')
plt.legend()

plt.tight_layout()
plt.show()
