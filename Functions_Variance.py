import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_rv_per_path(prices):
    log_prices = np.log(prices)
    returns = np.diff(log_prices, axis=0)
    rv_vector = np.sum(returns**2, axis=0)
    return rv_vector

def calculate_mean_rv(rv_vector):
    return np.mean(rv_vector)

