import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr

def mse(y_true, y_pred, extra=None):
    return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred, extra=None):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2(y_true, y_pred, extra=None):
    return r2_score(y_true, y_pred)

def rank_ic(signal, future_returns, extra=None):
    return spearmanr(signal, future_returns).correlation

def ic(signal, future_returns, extra=None):
    return pearsonr(signal, future_returns)[0]

def hit_rate(y_true, y_pred, extra=None):
    hits = np.sign(y_true) == np.sign(y_pred)
    return np.mean(hits)

def cumulative_return(returns, extra=None):
    return np.prod(1 + returns) - 1

def volatility(returns, extra=None):
    return returns.std() * np.sqrt(252)

def sharpe(returns, extra=None, risk_free_rate=0.02):
    vol = volatility(returns)
    mean_return = returns.mean() * 252
    return (mean_return - risk_free_rate) / vol if vol != 0 else 0.0

def max_drawdown(returns, extra=None):
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    return np.max(drawdowns)

def prediction_autocorr(signals, extra=None, lag=1):
    return signals.autocorr(lag=lag)
