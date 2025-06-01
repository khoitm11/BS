import pytest
import numpy as np
from core.black_scholes import european_call_price, european_put_price, calculate_d1_d2


# --- Test cases for european_call_price and european_put_price ---
@pytest.mark.parametrize(
    "S, K, T, r, sigma, expected_call, expected_put",
    [
        (100.0, 100.0, 1.00, 0.05, 0.20000, 10.45058357, 5.57352602),  # Case 1
        (100.0, 110.0, 1.00, 0.05, 0.20000, 6.04008813, 10.67532482),  # Case 2
        (100.0, 90.0, 1.00, 0.05, 0.20000, 16.69944841, 2.31009661),  # Case 3
        (100.0, 100.0, 0.00, 0.05, 0.20000, 0.00000000, 0.00000000),  # Case 4
        (105.0, 100.0, 0.00, 0.05, 0.20000, 5.00000000, 0.00000000),  # Case 5
        (95.0, 100.0, 0.00, 0.05, 0.20000, 0.00000000, 5.00000000),  # Case 6
        (100.0, 100.0, 1.00, 0.05, 0.00001, 4.87705755, 0.00000000),  # Case 7
        (100.0, 90.0, 1.00, 0.05, 0.00001, 14.38935179, 0.00000000),  # Case 8
        (100.0, 100.0, 0.10, 0.03, 0.20000, 2.67154121, 2.37199076),  # Case 9
        (100.0, 100.0, 0.50, 0.10, 0.30000, 10.90649985, 6.02944230),  # Case 10
        (100.0, 95.0, 0.75, 0.02, 0.10000, 7.48357911, 1.06921337),  # Case 11
    ],
)
def test_european_option_prices_various_scenarios(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    expected_call: float,
    expected_put: float,
):
    calculated_call = european_call_price(S, K, T, r, sigma)
    assert calculated_call == pytest.approx(expected_call, abs=1e-5)

    calculated_put = european_put_price(S, K, T, r, sigma)
    assert calculated_put == pytest.approx(expected_put, abs=1e-5)


# --- Test for Put-Call Parity ---
def test_put_call_parity():
    S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.04, 0.25

    call_price = european_call_price(S, K, T, r, sigma)
    put_price = european_put_price(S, K, T, r, sigma)

    left_hand_side = call_price - put_price
    right_hand_side = S - K * np.exp(-r * T)

    assert left_hand_side == pytest.approx(right_hand_side, abs=1e-5)


# --- Test for d1 and d2 calculation ---
def test_d1_d2_calculation_known_values():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    expected_d1 = 0.35
    expected_d2 = 0.15

    calculated_d1, calculated_d2 = calculate_d1_d2(S, K, T, r, sigma)
    assert calculated_d1 == pytest.approx(expected_d1, abs=1e-5)
    assert calculated_d2 == pytest.approx(expected_d2, abs=1e-5)


# --- Test for edge cases in d1, d2 and option pricing ---
def test_edge_cases_for_d1_d2_and_pricing():
    S_val, K_val, r_val, sigma_val = 100.0, 100.0, 0.05, 0.2
    T_zero = 0.0
    sigma_zero = 0.0
    T_normal = 1.0

    # --- Case: T = 0 (At expiry) ---
    # S = K at expiry
    d1_T0_atm, d2_T0_atm = calculate_d1_d2(S_val, K_val, T_zero, r_val, sigma_val)
    assert d1_T0_atm == np.inf
    assert d2_T0_atm == np.inf
    assert european_call_price(S_val, K_val, T_zero, r_val, sigma_val) == 0.0
    assert european_put_price(S_val, K_val, T_zero, r_val, sigma_val) == 0.0

    # S > K (Call ITM) at expiry
    S_itm = 110.0
    d1_T0_itm, d2_T0_itm = calculate_d1_d2(S_itm, K_val, T_zero, r_val, sigma_val)
    assert d1_T0_itm == np.inf
    assert d2_T0_itm == np.inf
    assert european_call_price(S_itm, K_val, T_zero, r_val, sigma_val) == S_itm - K_val

    # S < K (Put ITM) at expiry
    S_otm_call_itm_put = 90.0
    d1_T0_otm_call, d2_T0_otm_call = calculate_d1_d2(
        S_otm_call_itm_put, K_val, T_zero, r_val, sigma_val
    )
    assert d1_T0_otm_call == -np.inf
    assert d2_T0_otm_call == -np.inf
    assert (
        european_put_price(S_otm_call_itm_put, K_val, T_zero, r_val, sigma_val)
        == K_val - S_otm_call_itm_put
    )

    # --- Case: sigma = 0 (Zero volatility) ---
    # S = K, T > 0, sigma = 0. PV_K = K_val * np.exp(-r_val * T_normal) = 100 * exp(-0.05) = 95.122942
    # Since S (100) > PV_K, for a call, it's like being ITM from a risk-neutral forward perspective.
    d1_sig0_atm, d2_sig0_atm = calculate_d1_d2(
        S_val, K_val, T_normal, r_val, sigma_zero
    )
    assert d1_sig0_atm == np.inf
    assert d2_sig0_atm == np.inf
    expected_call_sig0_atm = max(0.0, S_val - K_val * np.exp(-r_val * T_normal))
    assert european_call_price(
        S_val, K_val, T_normal, r_val, sigma_zero
    ) == pytest.approx(expected_call_sig0_atm, abs=1e-5)
    expected_put_sig0_atm = max(
        0.0, K_val * np.exp(-r_val * T_normal) - S_val
    )  # Should be 0
    assert european_put_price(
        S_val, K_val, T_normal, r_val, sigma_zero
    ) == pytest.approx(expected_put_sig0_atm, abs=1e-5)

    # S_itm_put = 90, K=100, T > 0, sigma = 0. PV_K = 100 * exp(-0.05) = 95.122942
    # S (90) < PV_K. For a put, it's like ITM.
    S_itm_put_sig0 = 90.0
    d1_sig0_itm_put, d2_sig0_itm_put = calculate_d1_d2(
        S_itm_put_sig0, K_val, T_normal, r_val, sigma_zero
    )
    assert d1_sig0_itm_put == -np.inf  # S < PV_K for d1,d2 calculation with sigma=0
    assert d2_sig0_itm_put == -np.inf
    expected_put_sig0_itm = max(
        0.0, K_val * np.exp(-r_val * T_normal) - S_itm_put_sig0
    )  # 95.122942 - 90 = 5.122942
    assert european_put_price(
        S_itm_put_sig0, K_val, T_normal, r_val, sigma_zero
    ) == pytest.approx(expected_put_sig0_itm, abs=1e-5)
    expected_call_sig0_itm_put = max(
        0.0, S_itm_put_sig0 - K_val * np.exp(-r_val * T_normal)
    )  # Should be 0
    assert european_call_price(
        S_itm_put_sig0, K_val, T_normal, r_val, sigma_zero
    ) == pytest.approx(expected_call_sig0_itm_put, abs=1e-5)
