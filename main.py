import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError
from scipy.sparse.linalg import svds


# Function to normalize the matrix
def normalize_matrix(M):
    """Normalize matrix values between 0 and 1."""
    min_val = np.nanmin(M)
    max_val = np.nanmax(M)
    return (M - min_val) / (max_val - min_val + 1e-9)


# Function to load and preprocess the dataset
def dataloader(dataset='matrix'):
    path_prefix = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(path_prefix, dataset + '.csv')
    df = pd.read_csv(fname)
    feature_columns = df.columns.difference(['userName Labels', 'admit'])
    feature_matrix = df[feature_columns].values
    label_matrix = df['admit'].values
    return feature_matrix, label_matrix


# Singular Value Thresholding
def SVT(M1, iter_num, tau, tol):
    n1, n2 = M1.shape
    Omega = ~np.isnan(M1)
    P_Omega_M = np.nan_to_num(M1)
    observed_fraction = np.sum(Omega) / (n1 * n2)
    if observed_fraction == 0:
        print("No observed entries in the matrix. Exiting.")
        return None, []
    delta = 1.0 * observed_fraction
    Y = np.random.normal(0, 1e-4, size=(n1, n2))
    rmse = []
    for k in range(iter_num):
        rank = min(10, n1, n2)
        U, s, Vt = svds(Y, k=rank)
        s = s[::-1]
        U = U[:, ::-1]
        Vt = Vt[::-1, :]
        s_threshold = np.maximum(s - tau, 0)
        X = U @ np.diag(s_threshold) @ Vt
        X_Omega = X * Omega
        residual = P_Omega_M - X_Omega
        norm_residual = np.linalg.norm(residual)
        rmse_current = norm_residual / np.sqrt(np.sum(Omega))
        rmse.append(rmse_current)
        if rmse_current < tol:
            print(f"Convergence achieved at iteration {k + 1}.")
            break
        Y += delta * residual
    return X, rmse


def k_fold_cross_validation(X, y, K):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_sizes = np.full(K, n_samples // K, dtype=int)
    fold_sizes[:n_samples % K] += 1
    current = 0
    rmses = []
    for fold_num, fold_size in enumerate(fold_sizes):
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate((indices[:start], indices[stop:]))
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        try:
            w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
        except LinAlgError:
            print("Singular matrix encountered. Skipping this fold.")
            continue
        y_pred = X_test @ w
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        rmses.append(rmse)
        current = stop
    return np.mean(rmses) if rmses else np.nan


# Main Function
if __name__ == '__main__':
    start = time.time()
    feature_matrix, label_matrix = dataloader("matrix_sparse")
    # feature_matrix = normalize_matrix(feature_matrix)
    if np.isnan(feature_matrix).any():
        print("Feature matrix contains missing values.")
    tau_list = [30, 40, 50]
    tol_list = [0.001, 0.005, 0.01]
    iter_list = [10, 20, 30, 40]
    K = 5
    results = {}
    for tau in tau_list:
        results[tau] = {}
        for tol in tol_list:
            rmses = []
            for iter_num in iter_list:
                X_completed, rmse = SVT(feature_matrix, iter_num, tau, tol)
                if X_completed is None:
                    continue
                avg_rmse = k_fold_cross_validation(X_completed, label_matrix, K)
                rmses.append(avg_rmse)
                print(f"RMSE for tau={tau}, tol={tol}, iterations={iter_num}: {avg_rmse}")
            results[tau][tol] = rmses

    # Plot RMSE Results
    plt.figure(figsize=(12, 6))
    for tau in tau_list:
        for tol in tol_list:
            rmse_values = results[tau][tol]
            plt.plot(iter_list, rmse_values, label=f"tau={tau}, tol={tol}")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Average RMSE")
    plt.title("RMSE vs Iterations for Different Parameters")
    plt.legend()
    plt.grid()
    plt.savefig('RMSE_vs_Iterations.png')
    plt.show()

    print(f"Execution Time: {time.time() - start:.2f} seconds")
