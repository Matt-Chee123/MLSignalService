import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def rank_ic(signal, future_returns):
    return spearmanr(signal, future_returns)

def hit_rate(y_true, y_preds):
    hits = y_true == y_preds
    return np.mean(hits)

def cumulative_return(returns):
    return np.prod(1 + returns) - 1

def sharpe(returns):
    vol = volatility(returns)
    return ((returns.mean()*252) - 0.02) / vol

def max_drawdown(returns):
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max

    return np.max(drawdowns)

def volatility(returns):
    return returns.std * np.sqrt(252)

def prediction_autocorr(signals, lag=1):
    return signals.autocorr(lag=lag)
