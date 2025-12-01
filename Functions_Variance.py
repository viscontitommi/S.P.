import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


def calculate_rv_per_path(prices):
    log_prices = np.log(prices)
    returns = np.diff(log_prices, axis=0)
    rv_vector = np.sum(returns**2, axis=0)
    print(f"The Realized Volatility for each simulated path is:")
    return rv_vector

def calculate_rv_per_path2(prices):
    log_prices = np.log(prices)
    returns = np.diff(log_prices, axis=0)
    rv_vector = np.sum(returns**2, axis=0)
    return rv_vector

def calculate_mean_rv(rv_vector):
    print(f"The Mean Realized Volatility between all simulated paths is: {np.mean(rv_vector)}")
    return np.mean(rv_vector)

def calculate_mean_rv2(rv_vector):
    return np.mean(rv_vector)
