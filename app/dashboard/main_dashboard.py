import os
import sys
import time

import pandas as pd
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.black_scholes import (
    european_call_price,
    european_put_price,
    get_all_greeks,
)  # noqa: E402
from data_fetcher.live_data import (  # noqa: E402
    DEFAULT_DAYS_WINDOW_FOR_HV,
    calculate_historical_volatility_annualized,
    get_current_price_and_change,
)

st.set_page_config(layout="wide", page_title="Black-Scholes Lab", page_icon="🔬")

SESSION_DEFAULTS = {
    "ticker_symbol": "AAPL",
    "current_S_market": 150.0,
    "price_change": None,
    "price_pct_change": None,
    "hv20_market": 0.20,
    "last_fetch_timestamp": 0,
    "S_input_val": 150.0,
    "sigma_input_val": 0.20,
    "K_input_val": 150.0,
    "T_days_input_val": 30,
    "r_percent_input_val": 5.0,
    "days_theta_val": 365,
}
for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def fetch_market_data(ticker: str):
    st.session_state.last_fetch_timestamp = time.time()
    price_data = get_current_price_and_change(ticker)
    hv20_data = calculate_historical_volatility_annualized(
        ticker, hv_window_days=DEFAULT_DAYS_WINDOW_FOR_HV
    )

    if price_data and price_data[0] is not None:
        st.session_state.current_S_market = price_data[0]
        st.session_state.price_change = price_data[1]
        st.session_state.price_pct_change = price_data[2]
        st.session_state.S_input_val = price_data[0]
    else:
        st.sidebar.warning(f"Không lấy được giá cho {ticker}.")

    if hv20_data is not None:
        st.session_state.hv20_market = hv20_data
        st.session_state.sigma_input_val = hv20_data
    else:
        st.sidebar.warning(
            f"Không tính được HV{DEFAULT_DAYS_WINDOW_FOR_HV} cho {ticker}."
        )


st.sidebar.title("Thiết lập Thông số")
st.session_state.ticker_symbol = st.sidebar.text_input(
    "Mã Cổ phiếu", value=st.session_state.ticker_symbol
).upper()

if st.sidebar.button(
    f"Tải Dữ liệu cho {st.session_state.ticker_symbol}", key="fetch_button"
):
    with st.spinner(f"Đang tải dữ liệu cho {st.session_state.ticker_symbol}..."):
        fetch_market_data(st.session_state.ticker_symbol)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Giá trị Thị trường (Gợi ý)")
price_display_str = f"{st.session_state.current_S_market:.2f}"
if (
    st.session_state.price_change is not None
    and st.session_state.price_pct_change is not None
):
    change_sign_str = "+" if st.session_state.price_change >= 0 else ""
    price_display_str += (
        f" ({change_sign_str}{st.session_state.price_change:.2f}, "
        f"{change_sign_str}{st.session_state.price_pct_change:.2f}%)"
    )
st.sidebar.markdown(f"**Giá hiện tại (S):** {price_display_str}")
st.sidebar.markdown(
    f"**HV{DEFAULT_DAYS_WINDOW_FOR_HV} (σ gợi ý):** {st.session_state.hv20_market * 100:.2f}%"
)
st.sidebar.markdown("---")

st.sidebar.subheader("Tham số Tính toán Black-Scholes")
s_ui_val = st.sidebar.number_input(
    "Giá Tài sản (S)",
    value=float(st.session_state.S_input_val),
    step=0.01,
    format="%.2f",
)
k_ui_val = st.sidebar.number_input(
    "Giá Thực hiện (K)",
    value=float(st.session_state.get("K_input_val", s_ui_val)),
    step=0.50,
    format="%.2f",
)
t_days_ui_val = st.sidebar.number_input(
    "Thời gian Đáo hạn (ngày)",
    min_value=1,
    value=int(st.session_state.T_days_input_val),
    step=1,
)
r_percent_ui_val = st.sidebar.slider(
    "Lãi suất Phi Rủi ro (r %)",
    0.0,
    15.0,
    float(st.session_state.r_percent_input_val),
    0.1,
)
sigma_ui_val = st.sidebar.slider(
    "Độ Biến động Hàng năm (σ)",
    0.0001,
    2.0,
    float(st.session_state.sigma_input_val),
    0.001,
    format="%.4f",
    help=f"Giá trị HV{DEFAULT_DAYS_WINDOW_FOR_HV} ngày: {st.session_state.hv20_market * 100:.2f}%",
)
days_for_theta_ui_val = st.sidebar.selectbox(
    "Số ngày trong năm cho Theta",
    [365, 252],
    index=0 if st.session_state.days_theta_val == 365 else 1,
)

st.session_state.S_input_val = s_ui_val
st.session_state.K_input_val = k_ui_val
st.session_state.T_days_input_val = t_days_ui_val
st.session_state.r_percent_input_val = r_percent_ui_val
st.session_state.sigma_input_val = sigma_ui_val
st.session_state.days_theta_val = days_for_theta_ui_val

st.title("Ứng dụng Tính toán Black-Scholes & Greeks")

t_years_calc = t_days_ui_val / float(days_for_theta_ui_val)
r_annual_calc = r_percent_ui_val / 100.0

st.header("Giá Quyền chọn")
call_p_calc = european_call_price(
    s_ui_val, k_ui_val, t_years_calc, r_annual_calc, sigma_ui_val
)
put_p_calc = european_put_price(
    s_ui_val, k_ui_val, t_years_calc, r_annual_calc, sigma_ui_val
)
col_call_price, col_put_price = st.columns(2)
col_call_price.metric("Giá Call", f"{call_p_calc:.4f}")
col_put_price.metric("Giá Put", f"{put_p_calc:.4f}")

st.header("Phân tích Độ nhạy (Greeks)")
greeks_c_calc = get_all_greeks(
    s_ui_val,
    k_ui_val,
    t_years_calc,
    r_annual_calc,
    sigma_ui_val,
    "call",
    days_for_theta_ui_val,
)
greeks_p_calc = get_all_greeks(
    s_ui_val,
    k_ui_val,
    t_years_calc,
    r_annual_calc,
    sigma_ui_val,
    "put",
    days_for_theta_ui_val,
)

greeks_df_data_display = {
    "Call Option": [f"{v:.4f}" for v in greeks_c_calc.values()],
    "Put Option": [f"{v:.4f}" for v in greeks_p_calc.values()],
}
greeks_df_index_display = [name.capitalize() for name in greeks_c_calc.keys()]
greeks_df_display = pd.DataFrame(greeks_df_data_display, index=greeks_df_index_display)
st.table(greeks_df_display)

st.caption(
    """
    - Vega: Thay đổi giá / 1% thay đổi σ.
    - Theta: Thay đổi giá / 1 ngày.
    - Rho: Thay đổi giá / 1% thay đổi r.
"""
)
