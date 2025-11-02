import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from missforest import MissForest
import miceforest as mf

from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.linear_model import LogisticRegression

from auto_enc import LinearAutoencoder

from utilities import  StandardScaler
from pca import pca, apc
from tp_apc import tp_apc

from sklearn.model_selection import train_test_split

df=pd.read_csv("data/credit_train.csv")
cols_to_impute = ['Credit Score', 'Annual Income', 'Bankruptcies', 'Years in current job']

Y = np.ravel(pd.get_dummies(df.loc[:, 'Credit Default'], drop_first=True).astype(int))
df = df.drop(labels=['Credit Default'], axis=1)  # drop target from features


def clean_data(data:pd.DataFrame, cols_to_impute):
    data.drop_duplicates(inplace=True)
    # save ids and drop
    _ids = data['Id']
    data = data.drop(labels=['Id'], axis=1)

    data.loc[:, "Years in current job"] = data.loc[:, "Years in current job"].str.replace("< 1 year","0")

    data["Years in current job"] = data.loc[:, "Years in current job"].str.extract(r'(\d+)').astype(float)
    data = data.drop(labels=['Months since last delinquent'], axis=1) # drop this column due to too many NaN, redundant column
    # data.loc[:,'Months since last delinquent'] = data.loc[:,'Months since last delinquent'].fillna(0)
    numeric_columns = data.select_dtypes(exclude=['object']).columns.tolist()


    cat_columns = data.select_dtypes(include=['object']).columns.tolist()

    data_dummies = pd.get_dummies(data.loc[:, cat_columns], drop_first=True).astype(int)

    data_numeric = data.loc[:, numeric_columns]

    scaler = StandardScaler()

    data_numeric = scaler.fit_transform(data_numeric)

    data = pd.concat([data_numeric, data_dummies], axis=1)
    return data, data_numeric.shape[1], data_dummies.shape[1], _ids

df, n_numeric, n_dummies, ids = clean_data(df, cols_to_impute)

X = df.copy()

mse = mean_squared_error


pca_res = apc(np.nan_to_num(df), kmax=12)
C = pca_res['Chat']
exp_var = pca_res['explained_variance_ratio']

pca_res = apc(np.nan_to_num(df.iloc[:, 0:n_numeric]), kmax=12)
# check the explained variance ratio to see how many components need to be used in imputation
exp_var = pca_res['explained_variance_ratio']


nan_mask = np.isnan(np.array(df))

k = 3  # number of components to use for imputation based on explained variance ratio
pca_imputed = np.array(df.copy())
auto_imputed = np.array(df.copy())
missfor_imputed = np.array(df.copy())
mice_imputed = np.array(df.copy())

pca_results = tp_apc(np.array(df), kmax=k, center=False, standardize=False, re_estimate=True)
# pca_imputed[nan_mask] = pca_results['Chat'][nan_mask]
for ix, col in enumerate(df):
    dep_var = df.loc[:, col]
    ind_vars =pca_results['Fhat']
    _nan_mask = np.isnan(dep_var)
    not_nan_mask = ~_nan_mask

    # Fit linear regression only on non-missing values
    model = LinearRegression(fit_intercept=True)
    model.fit(ind_vars[not_nan_mask], dep_var[not_nan_mask])

    # Predict missing values
    pred_dep_var = model.predict(ind_vars)
    pca_imputed[:, ix][nan_mask[:, ix]] = pred_dep_var[nan_mask[:, ix]]

#
# nan_mask_pca = np.isnan(np.array(df)[:, 0:n_numeric])
# pca_results = tp_apc(np.array(df)[:, 0:n_numeric], kmax=k, center=False, standardize=False, re_estimate=True)
# pca_imputed[:, 0:n_numeric][nan_mask_pca] = pca_results['Chat'][nan_mask_pca]

# cat_columns = list(df.columns[n_numeric:])
# for col in cat_columns:
#     df.loc[:, col] = df[col].astype('category')

imputer = MissForest()
missfor_imp = np.array(imputer.fit_transform(df.copy()))
missfor_imputed[nan_mask] = missfor_imp[nan_mask]

# Create kernel.
kds = mf.ImputationKernel(df.copy())

# Run the MICE algorithm for 2 iterations
kds.mice(2)

# Return the completed dataset.
mice_imp = np.array(kds.complete_data())
mice_imputed[nan_mask] = mice_imp[nan_mask]




T, N = df.shape

autoencoder = LinearAutoencoder(input_dim=N, hidden_layers=[], hidden_dim=k, use_bias=True)
autoencoder.enable_masked_loss_function()

autoencoder.compile(optimizer="adam")

model_history = autoencoder.fit(df, df,
                                epochs=5000,
                                batch_size=T,
                                shuffle=False, verbose=True)
model_history.history['mse_loss']
auto_pred = autoencoder.predict(np.nan_to_num(df))

auto_imputed[nan_mask] = auto_pred[nan_mask]

for col in cols_to_impute:
    X[col] = X[col].fillna(X[col].mean())

X_imputed_pca = np.array(pca_imputed)

X_imputed_auto = np.array(auto_imputed)
X_imputed_missfor = np.array(missfor_imputed)
X_imputed_mice = np.array(mice_imputed)

test_size = 0.35
np.random.seed(12345)

base_res_train = []
base_res_test = []

pca_res_train = []
pca_res_test = []

auto_enc_res_train = []
auto_enc_res_test = []

missfor_res_train = []
missfor_res_test = []

mice_res_train = []
mice_res_test = []

base_res_auc = []
pca_res_auc = []
auto_enc_res_auc = []
missfor_res_auc = []
mice_res_auc = []

# X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_imputed_pca, Y, test_size=test_size,
#                                                                     random_state=np.random.randint(0, 2147483648))

classifier = LogisticRegression()

classifier.fit(X_imputed_auto, Y)



for i in range(1000):
    # randomstate
    r_state = np.random.randint(0, 2147483648)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
                                                        random_state=r_state)
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_imputed_pca, Y, test_size=test_size,
                                                                        random_state=r_state)
    X_train_auto, X_test_auto, y_train_auto, y_test_auto = train_test_split(X_imputed_auto, Y, test_size=test_size,
                                                                            random_state=r_state)

    X_train_missfor, X_test_missfor, y_train_missfor, y_test_missfor = train_test_split(X_imputed_missfor, Y, test_size=test_size,
                                                                            random_state=r_state)

    X_train_mice, X_test_mice, y_train_mice, y_test_mice = train_test_split(X_imputed_mice, Y, test_size=test_size,
                                                                            random_state=r_state)

    classifier=LogisticRegression()

    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)

    base_res_train.append(classifier.score(X_train,y_train))
    base_res_test.append(classifier.score(X_test,y_test))
    base_res_auc.append(roc_auc_score(y_test, y_pred))
    # base_res_auc = roc_auc_score(y_train, classifier.predict(X_train))



    classifier=LogisticRegression()

    classifier.fit(X_train_pca,y_train_pca)
    y_pred_pca=classifier.predict(X_test_pca)

    pca_res_train.append(classifier.score(X_train_pca,y_train_pca))
    pca_res_test.append(classifier.score(X_test_pca,y_test_pca))
    pca_res_auc.append(roc_auc_score(y_test_pca, y_pred_pca))
    # pca_res_auc = roc_auc_score(y_train_pca, classifier.predict(X_train_pca))


    classifier=LogisticRegression()
    classifier.fit(X_train_auto,y_train_auto)
    y_pred_auto=classifier.predict(X_test_auto)
    auto_enc_res_auc.append(roc_auc_score(y_test_auto, y_pred_auto))
    # auto_res_auc = roc_auc_score(y_train_auto, classifier.predict(X_train_auto))

    auto_enc_res_train.append(classifier.score(X_train_auto,y_train_auto))
    auto_enc_res_test.append(classifier.score(X_test_auto,y_test_auto))

    classifier=LogisticRegression()
    classifier.fit(X_train_missfor,y_train_missfor)
    y_pred_missfor=classifier.predict(X_test_missfor)
    missfor_res_auc.append(roc_auc_score(y_test_missfor, y_pred_missfor))
    # auto_res_auc = roc_auc_score(y_train_auto, classifier.predict(X_train_auto))

    missfor_res_train.append(classifier.score(X_train_missfor,y_train_missfor))
    missfor_res_test.append(classifier.score(X_test_missfor,y_test_missfor))

    classifier=LogisticRegression()
    classifier.fit(X_train_mice,y_train_mice)
    y_pred_mice=classifier.predict(X_test_mice)
    mice_res_auc.append(roc_auc_score(y_test_mice, y_pred_mice))
    # auto_res_auc = roc_auc_score(y_train_auto, classifier.predict(X_train_auto))

    mice_res_train.append(classifier.score(X_train_mice,y_train_mice))
    mice_res_test.append(classifier.score(X_test_mice,y_test_mice))



# print results
# print the name of each method followed by mean and std of train and test accuracies
print("Method Train_Mean Test_Mean Train_Std Test_Std")
print("Baseline", np.mean(base_res_train), np.mean(base_res_test), np.std(base_res_train), np.std(base_res_test))
print("PCA", np.mean(pca_res_train), np.mean(pca_res_test), np.std(pca_res_train), np.std(pca_res_test))
print("AutoEnc", np.mean(auto_enc_res_train), np.mean(auto_enc_res_test), np.std(auto_enc_res_train), np.std(auto_enc_res_test))
print("Missforest", np.mean(missfor_res_train), np.mean(missfor_res_test), np.std(missfor_res_train), np.std(missfor_res_test))
print("Miceforest", np.mean(mice_res_train), np.mean(mice_res_test), np.std(mice_res_train), np.std(mice_res_test))
print("Method Test_Auc Test_Std")
print("Baseline", np.mean(base_res_auc), np.std(base_res_auc))
print("PCA", np.mean(pca_res_auc), np.std(pca_res_auc))
print("AutoEnc", np.mean(auto_enc_res_auc), np.std(auto_enc_res_auc))
print("Missforest", np.mean(missfor_res_auc), np.std(missfor_res_auc))
print("Miceforest", np.mean(mice_res_auc), np.std(mice_res_auc))