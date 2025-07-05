import numpy as np

def calculate_volatility(prices):
    """
    Calculate the volatility (standard deviation) of a list of prices.
    """
    return float(np.std(prices))


def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    """
    Calculate the Sharpe ratio for a list of returns.
    """
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    if std_return == 0:
        return 0.0
    return float((mean_return - risk_free_rate) / std_return)
