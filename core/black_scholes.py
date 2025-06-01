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


def normal_pdf(x: float) -> float:
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)


def call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 1.0 if S > K else (0.5 if S == K else 0.0)
    if sigma <= 0:
        return 1.0 if S >= K * np.exp(-r * T) else 0.0
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)


def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return -1.0 if S < K else (-0.5 if S == K else 0.0)
    if sigma <= 0:
        return -1.0 if S < K * np.exp(-r * T) else 0.0
    return call_delta(S, K, T, r, sigma) - 1.0


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)

    return normal_pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    # Vega được báo cáo là thay đổi giá cho mỗi 1% thay đổi trong volatility
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return S * normal_pdf(d1) * np.sqrt(T) * 0.01


def theta_call_annual(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    if sigma <= 0:
        return 0.0

    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    term1 = -(S * normal_pdf(d1) * sigma) / (2.0 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    return term1 - term2


def theta_put_annual(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    if sigma <= 0:
        return 0.0

    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    term1 = -(S * normal_pdf(d1) * sigma) / (2.0 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return term1 + term2


def rho_call_annual(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    if sigma <= 0 and abs(S - K * np.exp(-r * T)) < 1e-6:
        return 0.0

    _, d2 = calculate_d1_d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01


def rho_put_annual(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    if sigma <= 0 and abs(K * np.exp(-r * T) - S) < 1e-6:
        return 0.0

    _, d2 = calculate_d1_d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01


def get_all_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    days_in_year: int = 365,
) -> dict:
    if T <= 0:  # Đã đáo hạn
        delta_val = 0.0
        if option_type.lower() == "call":
            delta_val = 1.0 if S > K else (0.5 if S == K else 0.0)
        else:
            delta_val = -1.0 if S < K else (-0.5 if S == K else 0.0)
        return {"delta": delta_val, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    if sigma <= 0:
        delta_val = 0.0
        rho_val = 0.0
        pv_k = K * np.exp(-r * T)
        if option_type.lower() == "call":
            delta_val = 1.0 if S >= pv_k else 0.0
        else:  # put
            delta_val = -1.0 if S < pv_k else 0.0
        return {
            "delta": delta_val,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": rho_val,
        }

    # Tính toán bình thường
    calculated_delta = 0.0
    calculated_theta_annual = 0.0
    calculated_rho_annual = 0.0

    if option_type.lower() == "call":
        calculated_delta = call_delta(S, K, T, r, sigma)
        calculated_theta_annual = theta_call_annual(S, K, T, r, sigma)
        calculated_rho_annual = rho_call_annual(S, K, T, r, sigma)
    elif option_type.lower() == "put":
        calculated_delta = put_delta(S, K, T, r, sigma)
        calculated_theta_annual = theta_put_annual(S, K, T, r, sigma)
        calculated_rho_annual = rho_put_annual(S, K, T, r, sigma)
    else:
        nan_greeks = {
            "delta": np.nan,
            "gamma": np.nan,
            "vega": np.nan,
            "theta": np.nan,
            "rho": np.nan,
        }
        return nan_greeks

    calculated_gamma = gamma(S, K, T, r, sigma)
    calculated_vega = vega(S, K, T, r, sigma)  # Đã nhân 0.01 bên trong hàm vega

    return {
        "delta": calculated_delta,
        "gamma": calculated_gamma,
        "vega": calculated_vega,
        "theta": calculated_theta_annual / days_in_year,  # Theta hàng ngày
        "rho": calculated_rho_annual,  # Đã nhân 0.01 bên trong hàm rho
    }
