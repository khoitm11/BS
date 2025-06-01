# core/option_analyzer.py
import numpy as np

from .black_scholes import european_call_price, european_put_price
from .gbm_simulator import get_terminal_prices, simulate_gbm_paths


def calculate_option_pnl_at_expiry(
    S_T: float,
    K: float,
    premium_transacted: float,
    option_type: str,
    position_type: str,
) -> float:
    payoff = 0.0
    if option_type.lower() == "call":
        payoff = max(0.0, S_T - K)
    elif option_type.lower() == "put":
        payoff = max(0.0, K - S_T)
    else:
        return np.nan  # Lỗi: loại quyền chọn không hợp lệ

    pnl = 0.0
    if position_type.lower() == "long":
        pnl = payoff - premium_transacted
    elif position_type.lower() == "short":
        pnl = premium_transacted - payoff
    else:
        return np.nan  # Lỗi: loại vị thế không hợp lệ
    return pnl


def simulate_option_pnl_distribution(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_for_premium_calc: float,
    sigma_for_gbm_sim: float,
    mu_for_gbm_sim: float,
    option_type: str,
    position_type: str,
    num_sim_paths: int,
    dt_gbm: float = 1.0 / 252.0,
    random_seed: int = None,
) -> tuple[np.ndarray, float]:
    assumed_premium = 0.0
    if option_type.lower() == "call":
        assumed_premium = european_call_price(S0, K, T, r, sigma_for_premium_calc)
    elif option_type.lower() == "put":
        assumed_premium = european_put_price(S0, K, T, r, sigma_for_premium_calc)
    else:
        return np.array([]), 0.0

    sim_price_paths = simulate_gbm_paths(
        S0, mu_for_gbm_sim, sigma_for_gbm_sim, T, dt_gbm, num_sim_paths, random_seed
    )
    terminal_prices = get_terminal_prices(sim_price_paths)

    pnl_values = np.zeros(num_sim_paths)
    for i in range(num_sim_paths):
        pnl_values[i] = calculate_option_pnl_at_expiry(
            terminal_prices[i], K, assumed_premium, option_type, position_type
        )
    return pnl_values, assumed_premium


def analyze_trading_edge(
    S_initial: float,
    K: float,
    T: float,
    r: float,
    sigma_model_theoretical: float,
    market_price_transacted: float,
    option_type: str,
    trade_action: str,
    mu_gbm_simulation: float,
    sigma_gbm_simulation: float,
    num_sim_paths: int,
    dt_gbm: float = 1.0 / 252.0,
    random_seed: int = None,
) -> dict:
    theoretical_price = 0.0
    if option_type.lower() == "call":
        theoretical_price = european_call_price(
            S_initial, K, T, r, sigma_model_theoretical
        )
    elif option_type.lower() == "put":
        theoretical_price = european_put_price(
            S_initial, K, T, r, sigma_model_theoretical
        )
    else:
        return {}

    edge = 0.0
    pnl_pos_type = ""
    if trade_action.lower() == "buy":
        edge = theoretical_price - market_price_transacted
        pnl_pos_type = "long"
    elif trade_action.lower() == "sell":
        edge = market_price_transacted - theoretical_price
        pnl_pos_type = "short"
    else:
        return {}

    sim_paths_edge = simulate_gbm_paths(
        S_initial,
        mu_gbm_simulation,
        sigma_gbm_simulation,
        T,
        dt_gbm,
        num_sim_paths,
        random_seed,
    )
    terminal_prices_edge = get_terminal_prices(sim_paths_edge)

    pnl_dist = np.zeros(num_sim_paths)
    for i in range(num_sim_paths):
        pnl_dist[i] = calculate_option_pnl_at_expiry(
            terminal_prices_edge[i],
            K,
            market_price_transacted,
            option_type,
            pnl_pos_type,
        )

    avg_pnl = np.mean(pnl_dist)
    std_pnl = np.std(pnl_dist)
    prob_profit = np.mean(pnl_dist > 0)

    results = {
        "theoretical_bs_price": theoretical_price,
        "market_transaction_price": market_price_transacted,
        "trade_action": trade_action,
        "identified_edge_per_share": edge,
        "simulated_average_pnl": avg_pnl,
        "simulated_std_dev_pnl": std_pnl,
        "simulated_prob_of_profit": prob_profit,
        "pnl_distribution_simulated_array": pnl_dist,
        # Thêm input params để dễ theo dõi nếu cần log
        "inputs": {
            "S0": S_initial,
            "K": K,
            "T": T,
            "r": r,
            "sigma_model": sigma_model_theoretical,
            "mu_gbm": mu_gbm_simulation,
            "sigma_gbm": sigma_gbm_simulation,
            "num_paths": num_sim_paths,
        },
    }
    return results
