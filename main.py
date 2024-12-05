import time
import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from scipy.sparse.linalg import svds

# Function to load the dataset
def dataloader(dataset='matrix'):
    """
    Loads the dataset from a CSV file, shuffles the rows, and splits into features and labels.

    Parameters:
    - dataset (str): Name of the dataset file (without extension).

    Returns:
    - feature_matrix (numpy.ndarray): The feature matrix.
    - label_matrix (numpy.ndarray): The labels associated with the rows.
    """
    path_prefix = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(path_prefix, dataset + '.csv')
    df = pd.read_csv(fname)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    feature_columns = df.columns.difference(['userName Labels', 'admit'])
    feature_matrix = df[feature_columns].values
    label_matrix = df['admit'].values
    return feature_matrix, label_matrix
def SVT(M1, iter_num, tau, tol):
    """
    Singular Value Thresholding (SVT) algorithm for matrix completion.

    Parameters:
    - M1 (numpy.ndarray): The input matrix with missing values (NaNs).
    - iter_num (int): Maximum number of iterations.
    - tau (float): Threshold for singular values.
    - tol (float): Convergence tolerance.

    Returns:
    - rmse (list): Root Mean Square Error (RMSE) at each iteration.
    """
    n1, n2 = M1.shape
    # Create a mask for observed entries (True where not NaN)
    Omega = ~np.isnan(M1)
    # Replace NaNs with zeros to create the observed matrix
    P_Omega_M = np.nan_to_num(M1)
    # Fraction of observed entries
    observed_fraction = np.sum(Omega) / (n1 * n2)
    # Step size for the dual variable update
    delta = 1.2 * observed_fraction
    Y = np.zeros((n1, n2))  # Initialize the dual variable
    rmse = []
    for k in range(iter_num):
        # Set the rank for truncated SVD; ensure it's valid
        rank = min(50, n1, n2) - 1  # Leave room for valid `k`
        if rank < 1:  # Ensure rank is valid for svds
            rank = 1
        if np.all(Y == 0):
            # Special case for zero matrix Y
            X = np.zeros_like(Y)
        else:
            try:
                # Compute truncated SVD of Y
                U, s, Vt = svds(Y, k=rank)
                s = s[::-1]
                U = U[:, ::-1]
                Vt = Vt[::-1, :]
                # Apply the soft-thresholding operator
                s_threshold = np.maximum(s - tau, 0)
                X = U @ np.diag(s_threshold) @ Vt
            except ValueError as e:
                # Handle case where rank is invalid
                print(f"Error in SVD computation: {e}")
                break
        # Project X onto observed entries
        X_Omega = X * Omega
        # Compute residual
        residual = P_Omega_M - X_Omega
        # Calculate RMSE
        norm_residual = la.norm(residual)
        rmse_current = norm_residual / np.sqrt(np.sum(Omega))
        rmse.append(rmse_current)
        # Check for convergence
        if norm_residual / la.norm(P_Omega_M) < tol:
            print(f"Convergence achieved at iteration {k + 1}.")
            break
        # Update the dual variable
        Y += delta * residual
    return rmse
# Updated Parameters for SVT
tau_list = [30, 40, 50]  # Increased tau values for faster shrinkage
iter_list = [10, 20, 30, 40]  # Increased iteration count
tol_list = [0.001, 0.005, 0.01]  # Looser tolerances

if __name__ == '__main__':
    start = time.time()
    # Load the feature matrix from sparse_matrix_75.csv
    feature_matrix, label_matrix = dataloader("sparse_matrix_modified")
    M1 = feature_matrix
    print(f"Feature matrix shape: {M1.shape}")
    if np.isnan(M1).any():
        print("Feature matrix contains missing values.")
    else:
        print("Feature matrix does not contain missing values. SVT is unnecessary.")
    
    results = {}
    # Iterate over different tau and tol values
    for tau in tau_list:
        results[tau] = {}
        for tol in tol_list:
            rmse_list = []
            print(f"\nProcessing tau = {tau}, tol = {tol}")
            for iter_num in iter_list:
                print(f"\nRunning SVT: tau = {tau}, tol = {tol}, max iterations = {iter_num}")
                rmse = SVT(M1, iter_num, tau, tol)
                if len(rmse) == 0:
                    print(f"No valid RMSE computed for tau={tau}, tol={tol}, iterations={iter_num}.")
                    rmse_list.append(None)
                else:
                    rmse_list.append(rmse[-1])
            results[tau][tol] = rmse_list

    # Plot results
    if plt:
        plt.figure(figsize=(12, 8))
        markers = ['o', 's', '^', 'D', 'v']
        for idx, tau in enumerate(tau_list):
            for tol in tol_list:
                rmse_values = results[tau][tol]
                plt.plot(iter_list, rmse_values, marker=markers[idx % len(markers)],
                         label=f'tau={tau}, tol={tol}')
        plt.xlabel('Number of Iterations')
        plt.ylabel('RMSE')
        plt.title('RMSE vs. Number of Iterations for Different Tau and Tol Values')
        plt.grid(True)
        plt.legend()
        now = datetime.datetime.now()
        formatted_time = now.strftime("%H_%M_%S")
        plt.savefig(f'rmse_plot_tau_tol_{formatted_time}.png')
        print(f"Plot saved as 'rmse_plot_tau_tol_{formatted_time}.png'.")
    else:
        print("No valid iterations to plot.")

    print(f"Total execution time: {time.time() - start:.2f} seconds")