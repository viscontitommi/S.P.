import numpy as np
import pandas as pd
import math as m
import scipy.stats as stats
import Functions_GBM as GBM
import Functions_Variance as f_var
import matplotlib.pyplot as plt
from IPython.display import display

def add_micro_noise(price_series, n_paths):
    noise = pd.DataFrame(np.random.normal(0, 0.005, (len(price_series), n_paths)))
    df_obs_price = price_series + noise
    return df_obs_price

def tsrv_estimator(prices, K): #K=300 is 5 minutes
    n = len(prices)
    log_prices = np.log(prices)
    
    # Questa stima contiene sia la Volatilità vera che 2*n*E[noise^2]
    rv_all = f_var.calculate_mean_rv2(f_var.calculate_rv_per_path2(prices)) #Fast Scale 
    
    rv_subgrids = [] # K Subgrids for Slow Scale
    for k in range(K):
        subgrid_prices = prices[k::K] 
        rv_sub = f_var.calculate_mean_rv2(f_var.calculate_rv_per_path2(subgrid_prices))
        rv_subgrids.append(rv_sub)

    rv_avg = np.mean(rv_subgrids) # Subgrids Mean

    n_bar = (n - K + 1) / K # constant for correction
    constraint_factor = 1 / (1 - (n_bar / n)) # Correction for small samples
    
    tsrv = constraint_factor * (rv_avg - (n_bar / n) * rv_all)
    
    return np.array(tsrv)

def calculate_error(est1, benchmark):
    err = abs(est1 - benchmark) / benchmark
    return np.array(err)

def generate_tsrv_table(prices, true_iv, k_range, k_chosen):
    tsrv_chosen = tsrv_estimator(prices, k_chosen) # K chosen from user
    
    k_values = list(k_range)
    estimates_list = []
    errors_list = []
    
    for k in k_values:
        est = tsrv_estimator(prices, k)
        estimates_list.append(est)
    
        err = (est - true_iv)**2 #MSE
        errors_list.append(err)
    tsrv_grid = np.vstack(estimates_list) # Matrice (n_k_values x n_paths)
    errors_grid = np.vstack(errors_list)
    
    best_indices = np.argmin(errors_grid, axis=0) #best K through the columns, K value for every path
    n_paths = tsrv_grid.shape[1] # extract optimal values
    
    optimal_k = []
    tsrv_optimal_values = []
    for i in range(n_paths):
        idx_best = best_indices[i] # Indice del K migliore per il path 'i'
        optimal_k.append(k_values[idx_best]) # Valore di K corrispondente
        tsrv_optimal_values.append(tsrv_grid[idx_best, i]) # Valore di TSRV corrispondente (riga=idx_best, colonna=i)
        
    if isinstance(prices, pd.DataFrame): # 5. Costruisci il DataFrame finale
        path_names = prices.columns
    else:
        path_names = [f"Path_{i}" for i in range(n_paths)]
        
    results_df = pd.DataFrame({
        'Optimal K': optimal_k,
        'TSRV (Optimal K)': tsrv_optimal_values,
        f'TSRV (K={k_chosen})': tsrv_chosen
    }, index=path_names)
    
    return results_df

def plot_consistency_check(mu, sigma, S0, noise_std, n_paths=1000):
    n_steps_list = [25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 3500, 5000] # testing frequencies
    target_iv = sigma**2
    
    rv_means = []
    tsrv_means = []
    for n in n_steps_list:
        dt = 1 / (252 * n)
        
        prices = GBM.sim_gbm_paths(n_paths, n, mu, sigma, S0, dt, seed=42)
        noise = np.random.normal(0, noise_std, prices.shape)
        noisy_prices = prices + noise
        
        rv_vec = f_var.calculate_rv_per_path2(noisy_prices)
        rv_means.append(np.mean(rv_vec))
        
        # 4. Calcola TSRV
        K = int(0.5 * (n**(2/3))) # Regola euristica per K
        if K < 1: K = 1
        
        # Chiamata alla funzione TSRV che hai già in questo file
        tsrv_vec = tsrv_estimator(noisy_prices, K)
        tsrv_means.append(np.mean(tsrv_vec))

    plt.figure(figsize=(10, 6))
    plt.plot(n_steps_list, rv_means, 'r-o', label='Realized Volatility (RV)', linewidth=1)
    plt.plot(n_steps_list, tsrv_means, 'g-s', label='Two-Scales RV (TSRV)', linewidth=1)
    plt.axhline(y=target_iv, color='blue', linestyle='--', label='True IV')
    plt.title(f"Volatility Signature Plot\n(Noise={noise_std})")
    plt.xlabel("Sampling Frequency (N steps/day)")
    plt.ylabel("Estimated Volatility")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()

def run_signature_plot(mu, sigma, S0, noise_std, n_paths):
    freq_list = [500, 1000, 5000, 10000, 23400] # Da bassa ad alta frequenza
    target_iv = (sigma**2) * (1/252) # Target giornaliero (approx)
    
    rv_means = []
    tsrv_means = []
    
    for n in freq_list:
        dt = 1 / (252 * n)
    
        prices = GBM.sim_gbm_paths(n_paths, n, mu, sigma, S0, dt)
        
        noise = pd.DataFrame(np.random.normal(0, noise_std, prices.shape))
        noisy = prices + noise
        
        rv_vec = f_var.calculate_rv_per_path2(noisy)
        rv_means.append(np.mean(rv_vec))
        
        K = int(0.5 * (n**(2/3)))
        if K < 1: K = 1
        tsrv_vec = tsrv_estimator(noisy, K) # Usa la tua funzione tsrv_estimator
        tsrv_means.append(np.mean(tsrv_vec))
        
    plt.figure(figsize=(10, 6))
    plt.plot(freq_list, rv_means, 'r-o', label='Realized Volatility (RV)')
    plt.plot(freq_list, tsrv_means, 'g-s', label='TSRV')
    plt.axhline(y=target_iv, color='blue', linestyle='--', label='True IV')
    plt.title("Volatility Signature Plot")
    plt.xlabel("Frequency (N steps)")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
