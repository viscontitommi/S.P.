import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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




# def compute_log_ret(prices):   
    # log_prices = np.log(prices)
    # log_returns = log_prices.diff()
    # return log_returns.dropna()



