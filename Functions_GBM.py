import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#Simulation of multiple GBM paths
def compute_gbm_log_ret(T, n_paths, mu, sigma, dt):
    err = np.random.normal(0, 1, (T, n_paths))
    drift_term = (mu - 0.5 * sigma**2) * dt
    diff_term = sigma * np.sqrt(dt) * err
    log_ret = drift_term + diff_term
    return pd.DataFrame(log_ret)

def compute_prices(log_ret, S0, n_paths):
    ini_log_pr = np.log(S0)
    cumulative_log_returns = np.cumsum(log_ret, axis=0) # Cumulative sum along each path
    start_row = np.full((1, n_paths), ini_log_pr) # Initial row with S0 log-price for each path
    log_prices = np.vstack([start_row, ini_log_pr + cumulative_log_returns]) # Add first row to cumulative sums
    prices = np.exp(log_prices)
    return pd.DataFrame(prices)

def sim_gbm_paths(n_paths, T, mu, sigma, S0, dt, seed=42):
    np.random.seed(seed)
    log_ret = compute_gbm_log_ret(T, n_paths, mu, sigma, dt)
    prices = compute_prices(log_ret, S0, n_paths)
    return prices

def plot_gbm_paths(ts_prices):
    plt.figure(figsize=(12, 6))
    plt.plot(ts_prices, alpha=0.6, linewidth=1)
    n_paths = ts_prices.shape[1]
    plt.title(f"Stock Price Simulation (GBM) - {n_paths} Scenarios")
    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    plt.show()

def martingale(prices, dt, mu):
    """
    Prepara i dati per il test di martingala scontando i prezzi.
    
    Parametri:
    prices (pd.DataFrame): DataFrame dei prezzi simulati (Tempo x Path).
    dt (float): Passo temporale (es. 1/252).
    mu (float): Drift utilizzato nella simulazione (tasso di sconto).
    
    Return:
    martingale_paths (pd.DataFrame): Tutti i percorsi scontati.
    mean_path (pd.Series): La media empirica dei percorsi scontati.
    """
    n_steps = len(prices)
    time_grid = np.arange(n_steps) * dt
    discount_factors = np.exp(-mu * time_grid)
    martingale_paths = prices.multiply(discount_factors, axis=0)
    mean_path = martingale_paths.mean(axis=1)

    return martingale_paths, mean_path

# Visual Test of Martingality
def plot_martingale_test(martingale_paths, mean_path, S0):
    """
    Genera il grafico per il test visivo di martingala.
    
    Parametri:
    martingale_paths (pd.DataFrame): I percorsi scontati.
    mean_path (pd.Series): La media dei percorsi.
    S0 (float): Il prezzo iniziale (valore teorico atteso).
    """
    plt.figure(figsize=(12, 6))
    n_plot = min(100, martingale_paths.shape[1])
    plt.plot(martingale_paths.iloc[:, :n_plot], color='grey', alpha=0.1, linewidth=0.5)
    
    plt.plot(mean_path, color='red', linewidth=2.5, label='Empirical Mean (Monte Carlo)')
    
    plt.axhline(y=S0, color='blue', linestyle='--', linewidth=2, label=f'Theoretical Value ({S0})')
    
    plt.title("Visual Test of Martingality (Discounted Prices)")
    plt.xlabel("Steps")
    plt.ylabel("Discounted Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

# Statistical Tests of Martingality
def test_martingale_statistics(martingale_paths, S0, alpha=0.05):
    """
    Esegue test statistici rigorosi per verificare la proprietà di martingala.
    
    Input:
    martingale_paths (pd.DataFrame): I percorsi dei prezzi scontati.
    S0 (float): Il prezzo iniziale teorico.
    alpha (float): Livello di significatività (default 5%).
    """
    print("=" * 60)
    print("       MARTINGALITY T-TESTS ON DISCOUNTED PRICES       ")
    print("=" * 60)
    
    final_values = martingale_paths.iloc[-1, :] #discounted price at the end of the simulation, T
    
    t_stat_final, p_val_final = stats.ttest_1samp(final_values, popmean=S0) # H0: Final Value Mean = S0
    
    print(f"Final Empirical Mean: {final_values.mean():.4f} (Target: {S0})")
    print(f"T-Statistic: {t_stat_final:.4f}")
    print(f"P-Value:     {p_val_final:.4f}")
    
    if p_val_final > alpha:
        print(">> Do not Reject H0. IT IS A MARTINGALE.")
    else:
        print(">> Reject H0. NOT A MARTINGALE.")

