# tests/test_option_analyzer.py
import pytest
import numpy as np
from core.option_analyzer import (
    calculate_option_pnl_at_expiry,
    simulate_option_pnl_distribution,
    analyze_trading_edge,
)

# Import thêm hàm BS để tự tính expected nếu cần đối chiếu chính xác.
from core.black_scholes import european_call_price, european_put_price


@pytest.mark.parametrize(
    "S_T, K, premium, option_type, position_type, expected_pnl",
    [
        (110, 100, 2.0, "call", "long", 8.0),  # Call ITM, P&L = (110-100) - 2.0
        (90, 100, 2.0, "call", "long", -2.0),  # Call OTM
        (110, 100, 2.0, "call", "short", -8.0),  # Short Call ITM
        (90, 100, 2.0, "call", "short", 2.0),  # Short Call OTM
        (90, 100, 1.5, "put", "long", 8.5),  # Put ITM, P&L = (100-90) - 1.5
        (110, 100, 1.5, "put", "long", -1.5),  # Put OTM
        (90, 100, 1.5, "put", "short", -8.5),  # Short Put ITM
        (110, 100, 1.5, "put", "short", 1.5),  # Short Put OTM
    ],
)
def test_calculate_option_pnl_at_expiry_valid_cases(
    S_T, K, premium, option_type, position_type, expected_pnl
):
    pnl = calculate_option_pnl_at_expiry(S_T, K, premium, option_type, position_type)
    assert pnl == pytest.approx(expected_pnl)


def test_calculate_option_pnl_at_expiry_invalid_inputs():
    assert np.isnan(calculate_option_pnl_at_expiry(100, 100, 1, "bad_type", "long"))
    assert np.isnan(calculate_option_pnl_at_expiry(100, 100, 1, "call", "bad_pos"))


def test_simulate_option_pnl_distribution_basic_run():
    S0, K, T, r = 100.0, 100.0, 0.1, 0.05
    sigma_prem, sigma_gbm, mu_gbm = 0.20, 0.25, 0.07
    num_paths = 50

    pnl_array_call, premium_call = simulate_option_pnl_distribution(
        S0,
        K,
        T,
        r,
        sigma_prem,
        sigma_gbm,
        mu_gbm,
        "call",
        "long",
        num_paths,
        random_seed=42,
    )
    assert isinstance(pnl_array_call, np.ndarray)
    assert len(pnl_array_call) == num_paths
    assert isinstance(premium_call, float)
    assert premium_call > 0

    pnl_array_put, premium_put = simulate_option_pnl_distribution(
        S0,
        K,
        T,
        r,
        sigma_prem,
        sigma_gbm,
        mu_gbm,
        "put",
        "short",
        num_paths,
        random_seed=43,
    )
    assert len(pnl_array_put) == num_paths
    assert premium_put > 0


def test_analyze_trading_edge_basic_run_and_results_structure():
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    sigma_model_val = 0.20
    market_price_call_transacted = 10.0
    mu_sim_val, sigma_sim_val = 0.05, 0.20
    num_paths_val = 50

    results = analyze_trading_edge(
        S_initial=S0,
        K=K,
        T=T,
        r=r,
        sigma_model_theoretical=sigma_model_val,
        market_price_transacted=market_price_call_transacted,
        option_type="call",
        trade_action="buy",
        mu_gbm_simulation=mu_sim_val,
        sigma_gbm_simulation=sigma_sim_val,
        num_sim_paths=num_paths_val,
        random_seed=101,
    )

    assert isinstance(results, dict)
    expected_keys = [
        "theoretical_bs_price",
        "market_transaction_price",
        "trade_action",
        "identified_edge_per_share",
        "simulated_average_pnl",
        "simulated_std_dev_pnl",
        "simulated_prob_of_profit",
        "pnl_distribution_simulated_array",
        "inputs",
    ]
    for key in expected_keys:
        assert key in results
    assert len(results["pnl_distribution_simulated_array"]) == num_paths_val

    # Call(100,100,1,0.05,0.2) = 10.45058357
    calculated_theoretical_call = european_call_price(S0, K, T, r, sigma_model_val)
    assert results["theoretical_bs_price"] == pytest.approx(
        calculated_theoretical_call, abs=1e-7
    )
    assert results["identified_edge_per_share"] == pytest.approx(
        calculated_theoretical_call - market_price_call_transacted, abs=1e-7
    )


def test_analyze_trading_edge_sell_put_negative_edge():
    S0, K, T, r = 100.0, 105.0, 0.5, 0.02
    sigma_model_val = 0.25
    market_price_at_which_sold_put = 3.50

    results = analyze_trading_edge(
        S_initial=S0,
        K=K,
        T=T,
        r=r,
        sigma_model_theoretical=sigma_model_val,
        market_price_transacted=market_price_at_which_sold_put,
        option_type="put",
        trade_action="sell",
        mu_gbm_simulation=r,
        sigma_gbm_simulation=sigma_model_val,
        num_sim_paths=200,
        random_seed=102,
    )

    assert results["trade_action"] == "sell"

    # Put(100,105,0.5,0.02,0.25) sẽ được tính bên trong analyze_trading_edge
    # và trả về trong results["theoretical_bs_price"]
    # Chúng ta sử dụng giá trị này để kiểm tra các tính toán phụ thuộc.
    calculated_theoretical_put = european_put_price(S0, K, T, r, sigma_model_val)

    assert results["theoretical_bs_price"] == pytest.approx(
        calculated_theoretical_put, abs=1e-7
    )

    expected_edge = market_price_at_which_sold_put - calculated_theoretical_put
    assert results["identified_edge_per_share"] == pytest.approx(
        expected_edge, abs=1e-7
    )

    # Với edge âm, PNL trung bình mô phỏng cũng nên có xu hướng âm.
    # Độ lớn cụ thể phụ thuộc vào độ lớn của edge và các tham số mô phỏng.
    assert results["simulated_average_pnl"] < 0.0
