import numpy as np
from scipy.stats import norm


def calculate_d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> tuple[float, float]:
    if T <= 0:
        d1_val = np.inf if S >= K else -np.inf
        d2_val = np.inf if S >= K else -np.inf
        return d1_val, d2_val

    if sigma <= 0:
        pv_k = K * np.exp(-r * T)
        d1_val = np.inf if S >= pv_k else -np.inf
        d2_val = np.inf if S >= pv_k else -np.inf
        return d1_val, d2_val

    numerator_d1 = np.log(S / K) + (r + 0.5 * sigma**2) * T
    denominator_d1 = sigma * np.sqrt(T)
    d1_val = numerator_d1 / denominator_d1
    d2_val = d1_val - sigma * np.sqrt(T)
    return d1_val, d2_val


def european_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(0.0, S - K)

    if sigma <= 0:
        return max(0.0, S - K * np.exp(-r * T))

    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price


def european_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(0.0, K - S)

    if sigma <= 0:
        return max(0.0, K * np.exp(-r * T) - S)

    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price
