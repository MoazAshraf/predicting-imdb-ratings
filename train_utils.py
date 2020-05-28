import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib


MODELS_DIR = 'models'
TRAIN_LOGS_DIR = 'train_logs'

def cross_val_rmse(model, X, y, cv=5, random_state=None, model_name=None, verbose=False):
    """
    Using K-fold cross validation, this function evaluates root mean squared error on training folds and validation folds
    """

    # make sure X and y are numpy arrays for slicing later
    X = np.array(X)
    y = np.array(y)

    # split data into folds
    kf = KFold(n_splits=cv, shuffle=False, random_state=random_state)
    fold_indices = kf.split(X)

    if verbose:
        print(f"Starting {cv}-fold cross validation")

    rmse_list = []
    for i, indices in enumerate(fold_indices):
        train_indices = indices[0]
        val_indices = indices[1]
        
        # train the model on the training folds
        timer_start = time.perf_counter_ns()
        model.fit(X[train_indices], y[train_indices])
        train_time = time.perf_counter_ns() - timer_start

        # evaluate the model on all folds
        y_pred = model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y[train_indices], y_pred[train_indices]))
        val_rmse = np.sqrt(mean_squared_error(y[val_indices], y_pred[val_indices]))

        if verbose:
            print(f"fit {i}\ttrain RMSE: {train_rmse:.3f}\t val RMSE: {val_rmse:.3f}\t train time: {train_time * 1e-9:.2f} s")

        rmse_list.append([train_rmse, val_rmse])
    
    # create a data frame
    index = ['train', 'val']
    if model_name is not None:
        for i in range(len(index)):
            index[i] = f"{model_name} {index[i]}"
    df = pd.DataFrame(np.array(rmse_list).T, index=index, columns=["fold " + str(i) for i in range(cv)])
    
    # compute mean and standard deviation
    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)
    df['mean'] = df_mean
    df['std'] = df_std

    return df

def save_model(model, filename):
    """
    Saves a scikit-learn model
    """

    if not os.path.isdir(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    joblib.dump(model, os.path.join(MODELS_DIR, filename))


def load_model(filename):
    """
    Loads a scikit-learn model
    """

    return joblib.load(os.path.join(MODELS_DIR, filename))

def save_cv_results(cv_results, filename):
    df = pd.DataFrame(cv_results)

    if not os.path.isdir(TRAIN_LOGS_DIR):
        os.makedirs(TRAIN_LOGS_DIR)
    
    df.to_csv(os.path.join(TRAIN_LOGS_DIR, filename))

def summarize_cv_results(cv_results):

    # count number of splits
    n_splits = 0
    while f"split{n_splits}_train_score" in cv_results.keys():
        n_splits += 1

    # get parameters
    params = pd.DataFrame(cv_results['params'])

    # calculate errors rather than scores
    train_errors = np.array([np.sqrt(-cv_results[f"split{i}_train_score"]) for i in range(n_splits)]).T
    val_errors = np.array([np.sqrt(-cv_results[f"split{i}_test_score"]) for i in range(n_splits)]).T

    # mean and standard deviations of errors
    train_error_mean = train_errors.mean(axis=1)
    val_error_mean = val_errors.mean(axis=1)
    train_error_std = train_errors.std(axis=1)
    val_error_std = val_errors.std(axis=1)

    # difference between validation errors and training errors
    error_diff_mean = val_error_mean - train_error_mean

    errors_df = pd.DataFrame({
        "train_error_mean": train_error_mean,
        "val_error_mean": val_error_mean,
        "error_diff_mean": error_diff_mean,
        "train_error_std": train_error_std,
        "val_error_std": val_error_std,
        })
    
    return pd.concat([params, errors_df], axis=1)