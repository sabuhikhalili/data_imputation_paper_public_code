import numpy as np
import pandas as pd
import os

from tp_apc import tp_apc
from tw_apc import tw_apc
from pca import apc
from auto_enc import LinearAutoencoder

from joblib import Parallel, delayed
import argparse 
import warnings

from missforest import MissForest
import miceforest as mf

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore') 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.callbacks import EarlyStopping


from utilities import rms_difference, NoScaler, StandardScaler, CenterScaler, missing_random, extract_moving_months

from matplotlib import pyplot as plt

np.random.seed(12345)


cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, 'data')
file_path = os.path.join(data_dir, 'credit_default_taiwan.xlsx')


df=pd.read_excel(file_path)

data=df.copy()
data = data.drop(labels=['ID', 'Default'], axis=1)

Y = df['Default'].to_numpy()

labels = {"Male":1, "Female":2, "Graduate":1, "University":2, "High school":3, "Others":4, "Married":1, "Single":2}



# Subset for dropping NaN (all columns except those in exclude_cols)

cat_columns = ['SEX', 'EDUCATION', 'MARRIAGE']

numeric_columns =  [col for col in data.columns if col not in cat_columns]

observed_data = data.loc[:,numeric_columns+cat_columns]


m= 2147483648 #sys.maxsize for 32 bit
max_iter = 1000

parser = argparse.ArgumentParser(description='Command-line interface for setting the model variable.')

# Add an argument for the model
parser.add_argument('--model', type=str, default='auto_enc_masked', help='Specify the model type (default: auto_enc)')
parser.add_argument('--k', type=int, default='4', help='Specify the dimension of bottleneck/pca factors')
parser.add_argument('--center', action='store_true')
parser.add_argument('--standardize', action='store_true')

# Parse the command-line arguments
args = parser.parse_args()

args.T = observed_data.shape[0]
args.N = observed_data.shape[1]
scaler = StandardScaler()
pca_results = apc(scaler.fit_transform(observed_data.loc[:, numeric_columns]), kmax=20)

# these lines are used to decide on the number of components to be used in the imputation
explained_variance = pca_results['explained_variance_ratio'][0:20]
plt.bar(range(1, len(explained_variance) + 1), explained_variance*100)
plt.xlabel('Rank of Principal Components')
plt.ylabel('Explained Variance (%)')
plt.title('Explained Variance by Principal Components')

# plt.show()

rms_pred_train = np.zeros((max_iter, 1))
rms_pred_test = np.zeros((max_iter, 1))
rms_auc_test = np.zeros((max_iter, 1))
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

    missing_df = observed_data.copy()

    # shuffle columns
    column_indices = np.arange(N-len(cat_columns))
    np.random.shuffle(column_indices)
    row_indices = np.arange(T)
    np.random.shuffle(row_indices)

    # block missingness - non-random is applied
    missing_cols = column_indices[N_o:]
    missing_rows = row_indices[T_o:]

    # set artificial points to NaN
    for j in missing_cols:
        for i in missing_rows:
            missing_df.iloc[i, j] = None


    imputed_data = missing_df.copy()
    if model == 'tp_apc':
        scaler = StandardScaler()
        data_dummies = pd.get_dummies(imputed_data.loc[:, cat_columns], drop_first=True).astype(int)
        data_numeric = imputed_data.loc[:, numeric_columns]
        data_numeric = scaler.fit_transform(data_numeric)
        imputed_data = pd.concat([data_numeric, data_dummies], axis=1)
        data_numeric_observed = scaler.fit_transform(observed_data[numeric_columns])
        res = tp_apc(X=imputed_data, kmax=k, center=False, standardize=False, re_estimate=True)
        imputed_data.iloc[:] = res['data']
        rms_dif = rms_difference(data_numeric_observed, imputed_data[numeric_columns])

    elif model == 'tw_apc':
        scaler = StandardScaler()
        data_dummies = pd.get_dummies(imputed_data.loc[:, cat_columns], drop_first=True).astype(int)
        data_numeric = imputed_data.loc[:, numeric_columns]
        data_numeric = scaler.fit_transform(data_numeric)
        imputed_data = pd.concat([data_numeric, data_dummies], axis=1)

        res = tw_apc(X=imputed_data, kmax=k, center=False, standardize=False, re_estimate=True)
        imputed_data.iloc[:] = res['data']
        rms_dif = rms_difference(observed_data[numeric_columns], imputed_data[numeric_columns])


    elif model == 'auto_enc_masked':


        scaler = StandardScaler()
        data_dummies = pd.get_dummies(imputed_data.loc[:, cat_columns], drop_first=True).astype(int)
        data_numeric = imputed_data.loc[:, numeric_columns]
        data_numeric = scaler.fit_transform(data_numeric)
        nan_mask = np.isnan(data_numeric)
        imputed_data = pd.concat([data_numeric, data_dummies], axis=1)
        data_numeric_observed = scaler.fit_transform(observed_data[numeric_columns])
        N = imputed_data.shape[1]

        autoencoder = LinearAutoencoder(input_dim=N, hidden_layers=[], hidden_dim=k, use_bias=True)
        autoencoder.enable_masked_loss_function()

        callback = EarlyStopping(monitor='mse_loss',
                                              patience=1000, min_delta=0.00001, mode='min')
        autoencoder.compile(optimizer="adam")
        autoencoder.fit(imputed_data, imputed_data,
                        epochs=5000,
                        batch_size=T,
                        shuffle=False, verbose=False, callbacks=[callback])
        x_pr = pd.DataFrame(autoencoder.predict(np.nan_to_num(imputed_data)), columns=imputed_data.columns).loc[:, numeric_columns]
        data_num = np.array(data_numeric)
        data_num[nan_mask]= x_pr.to_numpy()[nan_mask]
        data_numeric = pd.DataFrame(data_num, columns=data_numeric.columns)

        imputed_data = pd.concat([data_numeric, data_dummies], axis=1)
        rms_dif = rms_difference(data_numeric_observed, data_numeric)


    elif model == "missforest":
        scaler  = StandardScaler()
        imputer = MissForest(categorical=cat_columns)
        for col in cat_columns:
            missing_df[col] = missing_df[col].replace(labels).astype(int)
        imputed_data = imputer.fit_transform(missing_df)
        imputed_data = imputed_data.loc[:, missing_df.columns]

        data_dummies = pd.get_dummies(imputed_data.loc[:, cat_columns], drop_first=True).astype(int)
        data_numeric = imputed_data.loc[:, numeric_columns]
        data_numeric = scaler.fit_transform(data_numeric)
        data_numeric_observed = scaler.fit_transform(observed_data[numeric_columns])
        imputed_data = pd.concat([data_numeric, data_dummies], axis=1)
        rms_dif = rms_difference(data_numeric_observed, data_numeric)


    elif model == "miceforest":

        scaler  = StandardScaler()
        for col in cat_columns:
            missing_df[col] = missing_df[col].replace(labels).astype(int)
        kds = mf.ImputationKernel(missing_df)
        kds.mice(5)
        imputed_data = kds.complete_data()
        imputed_data = imputed_data.loc[:, missing_df.columns]

        data_dummies = pd.get_dummies(imputed_data.loc[:, cat_columns], drop_first=True).astype(int)
        data_numeric = imputed_data.loc[:, numeric_columns]
        data_numeric = scaler.fit_transform(data_numeric)
        data_numeric_observed = scaler.fit_transform(observed_data[numeric_columns])
        imputed_data = pd.concat([data_numeric, data_dummies], axis=1)
        rms_dif = rms_difference(data_numeric_observed, data_numeric)

    elif model=='mean':
        scaler = StandardScaler()
        data_dummies = pd.get_dummies(imputed_data.loc[:, cat_columns], drop_first=True).astype(int)
        data_numeric = imputed_data.loc[:, numeric_columns]
        data_numeric = scaler.fit_transform(data_numeric)
        data_numeric_observed = scaler.fit_transform(observed_data[numeric_columns])


        data_numeric.fillna(data_numeric.mean(), inplace=True)

        imputed_data = pd.concat([data_numeric, data_dummies], axis=1)
        rms_dif = rms_difference(data_numeric_observed, data_numeric)
    elif model=='complete':
        imputed_data = observed_data
        rms_dif = rms_difference(observed_data, imputed_data)
    

    X = imputed_data.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=rn)


    classifier=LogisticRegression()

    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)

    # Print at every step
    print(f"Processing iteration {iter}")
    return [classifier.score(X_train,y_train), classifier.score(X_test,y_test), roc_auc_score(y_pred, y_test),  rms_dif]
        

with Parallel(n_jobs=20, prefer="processes", verbose=1) as parallel:
    results = parallel(delayed(process_iteration)(iter, np.random.randint(0, m), args=args) for iter in range(max_iter))


for idx, res in enumerate(results):
    rms_pred_train[idx] = res[0]
    rms_pred_test[idx] = res[1]
    rms_auc_test[idx] = res[2]
    rms_data[idx] = res[3]


rms_pred_train_mean = np.mean(rms_pred_train)
rms_pred_test_mean = np.mean(rms_pred_test)
rms_auc_test_mean = np.mean(rms_auc_test)
rms_data_mean = np.mean(rms_data)

std_pred_train_mean = np.std(rms_pred_train)
std_pred_test_mean = np.std(rms_pred_test)
std_auc_test_mean = np.std(rms_auc_test)
std_data_mean = np.std(rms_data)



scaler_type = int(args.standardize) + int(args.center)

with pd.ExcelWriter(f'output/table1_bank_blocks_{args.model}_modul_{scaler_type}.xlsx') as writer:
        row_names = ['rms_pred_train', 'rms_pred_test', 'rms_auc_test', 'rms_data', 'std_pred_train',
                     'std_pred_test', "std_auc_test", "std_data_mean"]
        values = [rms_pred_train_mean, rms_pred_test_mean, rms_auc_test_mean, rms_data_mean,
                  std_pred_train_mean, std_pred_test_mean, std_auc_test_mean, std_data_mean]
        pd.DataFrame(values, index=row_names).to_excel(writer, sheet_name=f'table_1')


