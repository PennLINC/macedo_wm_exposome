import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold

ALPHAS = (1, 10, 100, 500, 1000, 5000, 10000)
DEFAULT_N_COMPONENTS = 17

def haufe_transform(X, y, weights):
    X = np.asarray(X)
    cov_x = np.cov(X.T)
    return cov_x @ weights

# regresses nuisance variables from X (train and test) using a linear model fit on the training set only.
def regress_nuisance(X_train, X_test, nuisance_train, nuisance_test):
    model = LinearRegression()
    model.fit(nuisance_train, X_train)
    X_train_resid = X_train - model.predict(nuisance_train)
    X_test_resid  = X_test  - model.predict(nuisance_test)
    return X_train_resid, X_test_resid

def fit_model(X_train, y_train, X_test, model_type='ridge', alphas=ALPHAS, n_components=DEFAULT_N_COMPONENTS):
    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)
    y_train = np.asarray(y_train).reshape(-1)

    if model_type == 'ridge': 
        model = RidgeCV(alphas=alphas)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_type == 'pls': 
        model = PLSRegression(n_components=n_components)
        model.fit(X_train, y_train.reshape(-1, 1))
        y_pred = model.predict(X_test).reshape(-1)        

    return model, y_pred

def evaluate_model(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    r, _ = pearsonr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    y_median_pred = np.full_like(y_true, np.median(y_true))
    mae_baseline = median_absolute_error(y_true, y_median_pred)
    return r, mse, r2, medae

def permutation_test(X_train, y_train, X_test, y_test, n_permutations=1, seed=42,  model_type='ridge', alphas=ALPHAS,
                     n_components=DEFAULT_N_COMPONENTS):

    np.random.seed(seed)
    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)
    y_train = np.asarray(y_train).reshape(-1)
    y_test  = np.asarray(y_test).reshape(-1)

    model, y_pred_true = fit_model(X_train, y_train, X_test, model_type=model_type, alphas=alphas, n_components=n_components)
    r_true, _, _, _ = evaluate_model(y_test, y_pred_true)

    null_r_vals = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y_train)
        _, y_pred_perm = fit_model(X_train, y_perm, X_test, model_type=model_type, alphas=alphas, n_components=n_components)
        r_perm, _, _, _ = evaluate_model(y_test, y_pred_perm)
        null_r_vals.append(r_perm)

    null_r_vals = np.array(null_r_vals)
    p_val = (np.sum(null_r_vals >= r_true) + 1) / (n_permutations + 1)
    return r_true, null_r_vals, p_val

def get_scaled_data(df, features_subset, confounds, target):
    features = pd.DataFrame(StandardScaler().fit_transform(df[features_subset]), columns=features_subset, index=df.index)
    nuisance = pd.DataFrame(StandardScaler().fit_transform(df[confounds]), columns=confounds, index=df.index)
    target_vals = df[target]
    return features, nuisance, target_vals

def regress_nuisance_train_only(X_train, nuisance_train, X_test=None, nuisance_test=None):
    # fit nuisance model on training only.
    model = LinearRegression()
    model.fit(nuisance_train, X_train)
    X_train_resid = X_train - model.predict(nuisance_train)

    if X_test is None:
        return X_train_resid, None

    X_test_resid = X_test - model.predict(nuisance_test)
    return X_train_resid, X_test_resid

def prepare_direction_data(df_train, df_test, features_subset, confounds, target, test_features_already_residualized=False):
    X_train = df_train[features_subset].values
    X_test  = df_test[features_subset].values

    y_train = df_train[[target]].values
    y_test  = df_test[[target]].values

    # nuisance regression
    nuisance_train = df_train[confounds].values

    if test_features_already_residualized:
        # Residualize train only; leave test as-is 
        X_train_resid, _ = regress_nuisance_train_only(X_train, nuisance_train, X_test=None, nuisance_test=None)
        X_test_resid = X_test
    else:
        nuisance_test = df_test[confounds].values
        X_train_resid, X_test_resid = regress_nuisance_train_only(X_train, nuisance_train, X_test=X_test, nuisance_test=nuisance_test)

    # scale using train only (still correct)
    X_scaler = StandardScaler().fit(X_train_resid)
    y_scaler = StandardScaler().fit(y_train)

    X_train_std = X_scaler.transform(X_train_resid)
    X_test_std  = X_scaler.transform(X_test_resid)
    y_train_std = y_scaler.transform(y_train).reshape(-1)
    y_test_std  = y_scaler.transform(y_test).reshape(-1)

    return X_train_std, y_train_std, X_test_std, y_test_std

def evaluate_direction(X_train, y_train, X_test, y_test, seed, label, output_dir, target, model_type='ridge', permutations=1,
                       alphas=ALPHAS, n_components=DEFAULT_N_COMPONENTS):
    
    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)
    y_train = np.asarray(y_train).reshape(-1)
    y_test  = np.asarray(y_test).reshape(-1)

    # Permutation test
    r_true, null_r_vals, p_val = permutation_test(X_train, y_train, X_test, y_test, n_permutations=permutations,
                                                  seed=seed, model_type=model_type, alphas=alphas, n_components=n_components)
    # Final fit
    model, y_pred = fit_model(X_train, y_train, X_test, model_type=model_type, alphas=alphas, n_components=n_components)

    # Haufe transform 
    haufe = None
    weights = np.asarray(model.coef_).reshape(-1)
    haufe = haufe_transform(X_train, y_train, weights)

    if model_type == 'ridge':
        # save ridge coefs and alpha for reproducibility
        np.save(f"{output_dir}/coefs_{label}_{target}.npy", weights.astype(np.float32))
        np.save(f"{output_dir}/alphas_{label}_{target}.npy",np.array(model.alpha_, ndmin=1))

    # Save prediction
    np.save(f"{output_dir}/prediction_{label}_{target}.npy", y_pred)
    r, mse, r2, medae = evaluate_model(y_test, y_pred)

    return {'r': r, 'mse': mse, 'r2': r2, 'medae': medae, 'p_val': p_val, 'haufe': haufe, 'preds': y_pred, 'y_true': y_test}

def run_model_comparison(features_subset, df_A, df_B, target, output_dir, confounds, model_type='ridge', permutations=1, n_components=None):
    if model_type == 'pls' and n_components is None: n_components = DEFAULT_N_COMPONENTS

    os.makedirs(output_dir, exist_ok=True)

    # Split group info within each harmonized set
    A_train = df_A[df_A['matched_group'] == 1]  # group 1/A is training
    A_test  = df_A[df_A['matched_group'] == 2]  # group 2/B is testing

    B_train = df_B[df_B['matched_group'] == 2]  # group 2/B is training
    B_test  = df_B[df_B['matched_group'] == 1]  # group 1/A is testing

    # ----------------- A → B direction -----------------
    X_A_std, y_A_std, X_B_std, y_B_std = prepare_direction_data(A_train, A_test, features_subset, confounds, target)
    results_AtoB = evaluate_direction(X_A_std, y_A_std, X_B_std, y_B_std, seed=42, label='AtoB', output_dir=output_dir, target=target,
                                      model_type=model_type, permutations=permutations, n_components=n_components)

    # ----------------- B → A direction -----------------
    X_B2_std,y_B2_std,X_A2_std,y_A2_std = prepare_direction_data(B_train, B_test, features_subset, confounds, target)
    results_BtoA = evaluate_direction(X_B2_std, y_B2_std, X_A2_std, y_A2_std, seed=43, label='BtoA',
        output_dir=output_dir, target=target, model_type=model_type, permutations=permutations, n_components=n_components)

    performance = {'r_A':   results_BtoA['r'], 'mse_A': results_BtoA['mse'], 'r2_A':  results_BtoA['r2'], 'medae_A': results_BtoA['medae'],
                   'r_B':   results_AtoB['r'], 'mse_B': results_AtoB['mse'], 'r2_B':  results_AtoB['r2'], 'medae_B': results_AtoB['medae'],
                   'p_val_A': results_AtoB['p_val'], 'p_val_B': results_BtoA['p_val'],
                   'y_true_A': results_BtoA['y_true'], 'y_pred_A': results_BtoA['preds'], 
                   'y_true_B': results_AtoB['y_true'], 'y_pred_B': results_AtoB['preds'],
                   'n_components': n_components if model_type == 'pls' else None}

    haufe_A = results_AtoB['haufe']
    haufe_B = results_BtoA['haufe']

    haufe_df = pd.DataFrame({'feature': features_subset, 'haufe_weight_AtoB': haufe_A, 'haufe_weight_BtoA': haufe_B})

    return performance, haufe_df

def fit_pls_direction(df_train, df_test, features_subset, target, confounds, max_components=20, cv_tuning=True, fixed_n=None):
    X_train_std, y_train_std, X_test_std, y_test_std = prepare_direction_data(df_train, df_test, features_subset, confounds, target)
    Y_train_std = y_train_std.reshape(-1, 1)
    Y_test_std  = y_test_std.reshape(-1, 1)

    if cv_tuning:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        max_n_allowed = min(max_components, X_train_std.shape[1], X_train_std.shape[0]-1)
        max_n_allowed = max(max_n_allowed, 1)

        best_n, best_score = 1, -np.inf
        for n in range(1, max_n_allowed+1):
            pls = PLSRegression(n_components=n)
            scores = []
            for tr_idx, val_idx in kf.split(X_train_std):
                pls.fit(X_train_std[tr_idx], Y_train_std[tr_idx])
                preds = pls.predict(X_train_std[val_idx])[:, 0]
                yt = Y_train_std[val_idx, 0]
                r_val = np.nan if (np.std(yt)==0 or np.std(preds)==0) else pearsonr(yt, preds)[0]
                scores.append(r_val)
            mean_r = np.nanmean(scores)
            if mean_r > best_score:
                best_score, best_n = mean_r, n

        n_components = best_n
    else:
        n_components = max_components if fixed_n is None else fixed_n

    pls_final = PLSRegression(n_components=n_components).fit(X_train_std, Y_train_std)

    y_train_true = Y_train_std[:, 0]
    y_test_true  = Y_test_std[:, 0]
    y_train_pred = pls_final.predict(X_train_std)[:, 0]
    y_test_pred  = pls_final.predict(X_test_std)[:, 0]

    r_tr, mse_tr, r2_tr, medae_tr = evaluate_model(y_train_true, y_train_pred)
    r_te, mse_te, r2_te, medae_te = evaluate_model(y_test_true, y_test_pred)

    return {"r_train": r_tr, "r_test": r_te, "mse_train": mse_tr, "mse_test": mse_te, "r2_train": r2_tr, "r2_test": r_te,
            "medae_train": medae_tr, "medae_test": medae_te, "y_true_train": y_train_true, "y_pred_train": y_train_pred,
            "y_true_test": y_test_true, "y_pred_test": y_test_pred, "n_components": n_components, "model": pls_final}

def train_and_test_external(df_train, df_ext, features_subset, confounds, target,
                            label, output_dir, model_type='ridge', alphas=ALPHAS, n_components=DEFAULT_N_COMPONENTS,
                           test_features_already_residualized=False):
    # Train on df_train (subsetted beforehand, e.g. matched_group==1), test on df_ext (HBN).

    # align features
    shared_features = [f for f in features_subset if f in df_train.columns and f in df_ext.columns]

    X_train_std, y_train_std, X_ext_std, y_ext_std = prepare_direction_data(df_train, df_ext, shared_features, confounds, target,
                                                                            test_features_already_residualized=test_features_already_residualized)

    model, y_pred = fit_model(X_train_std, y_train_std, X_ext_std, model_type=model_type, alphas=alphas, n_components=n_components)

    # evaluate on external (HBN)
    r, mse, r2, medae = evaluate_model(y_ext_std, y_pred)

    # save predictions / coefs
    os.makedirs(output_dir, exist_ok=True)
    np.save(f"{output_dir}/prediction_{label}_{target}.npy", y_pred.astype(np.float32))
    np.save(f"{output_dir}/ytrue_{label}_{target}.npy", y_ext_std.astype(np.float32))
    if model_type == 'ridge':
        np.save(f"{output_dir}/coefs_{label}_{target}.npy", np.asarray(model.coef_, dtype=np.float32))
        np.save(f"{output_dir}/alphas_{label}_{target}.npy", np.array(model.alpha_, ndmin=1))

    return {"r": r, "mse": mse, "r2": r2, "medae": medae, "y_true": y_ext_std, "y_pred": y_pred, "features": shared_features,}

def run_pls_model_exact(df_A, df_B, features_subset, target, confounds, **pls_kwargs):
    # split halves inside each harmonization
    A_train = df_A[df_A["matched_group"] == 1].copy()
    A_test  = df_A[df_A["matched_group"] == 2].copy()

    B_train = df_B[df_B["matched_group"] == 2].copy()
    B_test  = df_B[df_B["matched_group"] == 1].copy()

    res_AtoB = fit_pls_direction(A_train, A_test, features_subset, target, confounds, **pls_kwargs)
    res_BtoA = fit_pls_direction(B_train, B_test, features_subset, target, confounds, **pls_kwargs)

    performance = {"r_A": res_BtoA["r_test"], "r_B": res_AtoB["r_test"], "mse_A": res_BtoA["mse_test"], "mse_B": res_AtoB["mse_test"],
                   "r2_A": res_BtoA["r2_test"], "r2_B": res_AtoB["r2_test"], "medae_A": res_BtoA["medae_test"], "medae_B": res_AtoB["medae_test"],
                   "y_true_A": res_BtoA["y_true_test"], "y_pred_A": res_BtoA["y_pred_test"], "y_true_B": res_AtoB["y_true_test"],
                   "y_pred_B": res_AtoB["y_pred_test"], "n_components_AtoB": res_AtoB["n_components"], 
                   "n_components_BtoA": res_BtoA["n_components"]}

    models = {"AtoB": res_AtoB["model"], "BtoA": res_BtoA["model"]}
    return performance, models


def permutation_test_external(df_train, df_ext, features_subset, target, model_type="ridge", n_permutations=1000,
                              seed=42, alphas=(1, 10, 100, 500, 1000, 5000, 10000)):
    """
    Permutation test for external generalization (e.g., ABCD → HBN).

    Permutes training labels only, refits the model, and evaluates on
    the fixed external test set.
    """

    rng = np.random.default_rng(seed)

    # ------------------
    # Align shared features
    # ------------------
    shared_features = [f for f in features_subset
                       if f in df_train.columns and f in df_ext.columns]

    X_train = df_train[shared_features].values
    X_ext   = df_ext[shared_features].values

    y_train = df_train[[target]].values
    y_ext   = df_ext[[target]].values

    # ------------------
    # Standardize (train only)
    # ------------------
    X_scaler = StandardScaler().fit(X_train)
    X_train_std = X_scaler.transform(X_train)
    X_ext_std   = X_scaler.transform(X_ext)

    y_scaler = StandardScaler().fit(y_train)
    y_train_std = y_scaler.transform(y_train).reshape(-1)
    y_ext_std   = y_scaler.transform(y_ext).reshape(-1)

    # ------------------
    # Fit true model
    # ------------------
    model = RidgeCV(alphas=alphas)
    model.fit(X_train_std, y_train_std)
    y_pred_true = model.predict(X_ext_std)

    r_true = np.corrcoef(y_ext_std, y_pred_true)[0, 1]

    # ------------------
    # Null distribution
    # ------------------
    null_r = np.empty(n_permutations)

    for i in range(n_permutations):
        y_perm = rng.permutation(y_train_std)

        model_perm = RidgeCV(alphas=alphas)
        model_perm.fit(X_train_std, y_perm)
        y_pred_perm = model_perm.predict(X_ext_std)

        null_r[i] = np.corrcoef(y_ext_std, y_pred_perm)[0, 1]

    # ------------------
    # One-sided p-value
    # ------------------
    p_val = (np.sum(null_r >= r_true) + 1) / (n_permutations + 1)

    return r_true, null_r, p_val, shared_features


