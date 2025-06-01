# generate_test_benchmarks.py
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


# ĐẢM BẢO TÍNH NHẤT QUÁN.


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
    if denominator_d1 == 0:
        d1_val = np.inf if numerator_d1 > 0 else (-np.inf if numerator_d1 < 0 else 0)
        d2_val = d1_val
    else:
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
    pdf_d1 = normal_pdf(d1)
    denominator = S * sigma * np.sqrt(T)
    if denominator == 0:
        return np.inf if pdf_d1 > 0 else 0.0  # Tránh chia cho 0, gamma có thể rất lớn
    return pdf_d1 / denominator


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
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
    sqrt_T = np.sqrt(T)
    if sqrt_T == 0:
        return (
            -r * K * np.exp(-r * T) * norm.cdf(d2) if d2 != -np.inf else 0.0
        )  # Đơn giản hóa

    term1 = -(S * normal_pdf(d1) * sigma) / (2.0 * sqrt_T)
    term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    return term1 - term2


def theta_put_annual(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    if sigma <= 0:
        return 0.0
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    if sqrt_T == 0:
        return (
            r * K * np.exp(-r * T) * norm.cdf(-d2) if d2 != np.inf else 0.0
        )  # Đơn giản hóa

    term1 = -(S * normal_pdf(d1) * sigma) / (2.0 * sqrt_T)
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return term1 + term2


def rho_call_annual(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    if sigma <= 0:
        return T * K * np.exp(-r * T) * 0.01 if S >= K * np.exp(-r * T) else 0.0
    _, d2 = calculate_d1_d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01


def rho_put_annual(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    if sigma <= 0:
        return -T * K * np.exp(-r * T) * 0.01 if S < K * np.exp(-r * T) else 0.0
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
    if T <= 0:
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
            rho_val = T * K * np.exp(-r * T) * 0.01 if S >= pv_k else 0.0
        else:
            delta_val = -1.0 if S < pv_k else 0.0
            rho_val = -T * K * np.exp(-r * T) * 0.01 if S < pv_k else 0.0
        return {
            "delta": delta_val,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": rho_val,
        }

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
        return {k: np.nan for k in ["delta", "gamma", "vega", "theta", "rho"]}

    calculated_gamma = gamma(S, K, T, r, sigma)
    calculated_vega = vega(S, K, T, r, sigma)
    return {
        "delta": calculated_delta,
        "gamma": calculated_gamma,
        "vega": calculated_vega,
        "theta": calculated_theta_annual / days_in_year,
        "rho": calculated_rho_annual,
    }


# --- Tham số đầu vào cho các kịch bản giá Call/Put ---
option_price_test_case_params = [
    (100.0, 100.0, 1.00, 0.05, 0.20000),
    (100.0, 110.0, 1.00, 0.05, 0.20000),
    (100.0, 90.0, 1.00, 0.05, 0.20000),
    (100.0, 100.0, 0.00, 0.05, 0.20000),  # T = 0
    (105.0, 100.0, 0.00, 0.05, 0.20000),  # T = 0
    (95.0, 100.0, 0.00, 0.05, 0.20000),  # T = 0
    (100.0, 100.0, 1.00, 0.05, 0.00001),  # Sigma rất nhỏ
    (100.0, 90.0, 1.00, 0.05, 0.00001),  # Sigma rất nhỏ
    (100.0, 100.0, 0.10, 0.03, 0.20000),  # T ngắn
    (100.0, 100.0, 0.50, 0.10, 0.30000),  # r cao
    (100.0, 95.0, 0.75, 0.02, 0.10000),  # sigma thấp (khác 0)
]

# --- Tham số benchmark chuẩn cho Greeks ---
S_g_bench = 100.0
K_g_bench = 100.0
T_g_bench = 1.0
r_g_bench = 0.05
sigma_g_bench = 0.20
days_in_year_g_bench = 365

if __name__ == "__main__":
    print("# --- Expected values for test_european_option_prices_various_scenarios ---")
    print("# Paste these tuples into your @pytest.mark.parametrize list")
    print("# Format: (S, K, T, r, sigma, expected_call, expected_put)")
    print("    [")
    for i, params_tuple in enumerate(option_price_test_case_params):
        S, K, T, r, sigma_val = params_tuple
        expected_call = european_call_price(S, K, T, r, sigma_val)
        expected_put = european_put_price(S, K, T, r, sigma_val)
        comma = (
            "," if i < len(option_price_test_case_params) - 1 else ""
        )  # Không có dấu phẩy ở dòng cuối
        print(
            f"        ({S:.1f}, {K:.1f}, {T:.2f}, {r:.2f}, {sigma_val:.5f}, {expected_call:.8f}, {expected_put:.8f}){comma} # Case {i + 1}"
        )
    print("    ],")
    print("\n")

    print("# --- Expected values for individual Greek function tests ---")
    print(
        "# Params: S={}, K={}, T={}, r={}, sigma={}".format(
            S_g_bench, K_g_bench, T_g_bench, r_g_bench, sigma_g_bench
        )
    )

    expected_val = call_delta(S_g_bench, K_g_bench, T_g_bench, r_g_bench, sigma_g_bench)
    print(f"    # For test_greek_call_delta():\n    expected = {expected_val:.8f}")

    expected_val = put_delta(S_g_bench, K_g_bench, T_g_bench, r_g_bench, sigma_g_bench)
    print(f"    # For test_greek_put_delta():\n    expected = {expected_val:.8f}")

    expected_val = gamma(S_g_bench, K_g_bench, T_g_bench, r_g_bench, sigma_g_bench)
    print(f"    # For test_greek_gamma():\n    expected = {expected_val:.8f}")

    expected_val = vega(S_g_bench, K_g_bench, T_g_bench, r_g_bench, sigma_g_bench)
    print(f"    # For test_greek_vega():\n    expected = {expected_val:.8f}")

    expected_val = theta_call_annual(
        S_g_bench, K_g_bench, T_g_bench, r_g_bench, sigma_g_bench
    )
    print(
        f"    # For test_greek_theta_call_daily() (expected_annual value):\n    expected_annual = {expected_val:.8f}"
    )

    expected_val = theta_put_annual(
        S_g_bench, K_g_bench, T_g_bench, r_g_bench, sigma_g_bench
    )
    print(
        f"    # For test_greek_theta_put_daily() (expected_annual value):\n    expected_annual = {expected_val:.8f}"
    )

    expected_val = rho_call_annual(
        S_g_bench, K_g_bench, T_g_bench, r_g_bench, sigma_g_bench
    )
    print(
        f"    # For test_greek_rho_call_percentage():\n    expected = {expected_val:.8f}"
    )

    expected_val = rho_put_annual(
        S_g_bench, K_g_bench, T_g_bench, r_g_bench, sigma_g_bench
    )
    print(
        f"    # For test_greek_rho_put_percentage():\n    expected = {expected_val:.8f}"
    )
    print("\n")

    print("# --- Expected values for test_get_all_greeks_call() ---")
    greeks_c_all = get_all_greeks(
        S_g_bench,
        K_g_bench,
        T_g_bench,
        r_g_bench,
        sigma_g_bench,
        "call",
        days_in_year_g_bench,
    )
    print(
        '    # assert greeks["delta"] == pytest.approx({:.8f}, abs=1e-7)'.format(
            greeks_c_all["delta"]
        )
    )
    print(
        '    # assert greeks["gamma"] == pytest.approx({:.8f}, abs=1e-7)'.format(
            greeks_c_all["gamma"]
        )
    )
    print(
        '    # assert greeks["vega"] == pytest.approx({:.8f}, abs=1e-7)'.format(
            greeks_c_all["vega"]
        )
    )
    print(
        '    # assert greeks["theta"] == pytest.approx({:.8f}, abs=1e-7)'.format(
            greeks_c_all["theta"]
        )
    )  # Daily
    print(
        '    # assert greeks["rho"] == pytest.approx({:.8f}, abs=1e-7)'.format(
            greeks_c_all["rho"]
        )
    )
    print("\n")

    print("# --- Expected values for test_get_all_greeks_put() ---")
    greeks_p_all = get_all_greeks(
        S_g_bench,
        K_g_bench,
        T_g_bench,
        r_g_bench,
        sigma_g_bench,
        "put",
        days_in_year_g_bench,
    )
    print(
        '    # assert greeks["delta"] == pytest.approx({:.8f}, abs=1e-7)'.format(
            greeks_p_all["delta"]
        )
    )
    print(
        '    # assert greeks["gamma"] == pytest.approx({:.8f}, abs=1e-7)'.format(
            greeks_p_all["gamma"]
        )
    )
    print(
        '    # assert greeks["vega"] == pytest.approx({:.8f}, abs=1e-7)'.format(
            greeks_p_all["vega"]
        )
    )
    print(
        '    # assert greeks["theta"] == pytest.approx({:.8f}, abs=1e-7)'.format(
            greeks_p_all["theta"]
        )
    )  # Daily
    print(
        '    # assert greeks["rho"] == pytest.approx({:.8f}, abs=1e-7)'.format(
            greeks_p_all["rho"]
        )
    )
