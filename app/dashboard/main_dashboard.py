# app/dashboard/main_dashboard.py
import streamlit as st
import pandas as pd
import sys
import os
import time  # Để xử lý logic làm mới

# Thêm thư mục gốc vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.black_scholes import get_all_greeks, european_call_price, european_put_price
from data_fetcher.live_data import (
    get_current_price_and_change,
    calculate_historical_volatility_annualized,
)

st.set_page_config(layout="wide", page_title="Black-Scholes Lab", page_icon="🔬")

# Khởi tạo session state nếu chưa có
SESSION_DEFAULTS = {
    "ticker_symbol": "AAPL",
    "current_S_market": 150.0,  # Giá trị mặc định ban đầu
    "price_change": None,
    "price_pct_change": None,
    "hv20_market": 0.20,  # HV 20 ngày
    "hv60_market": 0.22,  # HV 60 ngày
    "last_fetch_timestamp": 0,
    "S_input_val": 150.0,
    "sigma_input_val": 0.20,
    "K_input_val": 150.0,
    "T_days_input_val": 30,
    "r_percent_input_val": 5.0,
}
for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- LOGIC LẤY DỮ LIỆU THỊ TRƯỜNG ---
def fetch_market_data(ticker: str):
    st.session_state.last_fetch_timestamp = time.time()
    price_data = get_current_price_and_change(ticker)
    hv20_data = calculate_historical_volatility_annualized(ticker, hv_window_days=20)
    # hv60_data = calculate_historical_volatility_annualized(ticker, hv_window_days=60) # Có thể thêm nếu muốn

    if price_data and price_data[0] is not None:
        st.session_state.current_S_market = price_data[0]
        st.session_state.price_change = price_data[1]
        st.session_state.price_pct_change = price_data[2]
        st.session_state.S_input_val = price_data[0]  # Tự động cập nhật ô input S
    else:
        st.sidebar.warning(f"Không lấy được giá cho {ticker}.")

    if hv20_data is not None:
        st.session_state.hv20_market = hv20_data
        st.session_state.sigma_input_val = hv20_data  # Tự động cập nhật ô input sigma
    else:
        st.sidebar.warning(f"Không tính được HV20 cho {ticker}.")


# --- SIDEBAR ---
st.sidebar.title("Thiết lập Thông số")
st.session_state.ticker_symbol = st.sidebar.text_input(
    "Mã Cổ phiếu", value=st.session_state.ticker_symbol
).upper()

if st.sidebar.button(
    "Tải Dữ liệu cho " + st.session_state.ticker_symbol, key="fetch_button"
):
    with st.spinner(f"Đang tải dữ liệu cho {st.session_state.ticker_symbol}..."):
        fetch_market_data(st.session_state.ticker_symbol)
    st.rerun()  # Chạy lại toàn bộ script để cập nhật UI

st.sidebar.markdown("---")
st.sidebar.subheader("Giá trị Thị trường (Gợi ý)")
price_display = "{:.2f}".format(st.session_state.current_S_market)
if (
    st.session_state.price_change is not None
    and st.session_state.price_pct_change is not None
):
    change_sign = "+" if st.session_state.price_change >= 0 else ""
    price_display += (
        " ("
        + change_sign
        + "{:.2f}".format(st.session_state.price_change)
        + ", "
        + change_sign
        + "{:.2f}".format(st.session_state.price_pct_change)
        + "%)"
    )
st.sidebar.markdown(f"**Giá hiện tại (S):** {price_display}")
st.sidebar.markdown(f"**HV20 (σ gợi ý):** {st.session_state.hv20_market * 100:.2f}%")
st.sidebar.markdown("---")

st.sidebar.subheader("Tham số Tính toán Black-Scholes")
s_ui = st.sidebar.number_input(
    "Giá Tài sản (S)",
    value=float(st.session_state.S_input_val),
    step=0.01,
    format="%.2f",
)
k_ui = st.sidebar.number_input(
    "Giá Thực hiện (K)",
    value=float(st.session_state.get("K_input_val", s_ui)),
    step=0.50,
    format="%.2f",
)
t_days_ui = st.sidebar.number_input(
    "Thời gian Đáo hạn (ngày)",
    min_value=1,
    value=int(st.session_state.T_days_input_val),
    step=1,
)
r_percent_ui = st.sidebar.slider(
    "Lãi suất Phi Rủi ro (r %)",
    0.0,
    15.0,
    float(st.session_state.r_percent_input_val),
    0.1,
)
sigma_ui = st.sidebar.slider(
    "Độ Biến động Hàng năm (σ)",
    0.0001,
    2.0,
    float(st.session_state.sigma_input_val),
    0.001,
    format="%.4f",
)
days_for_theta_ui = st.sidebar.selectbox(
    "Số ngày trong năm cho Theta",
    [365, 252],
    index=0 if st.session_state.get("days_theta_val", 365) == 365 else 1,
)

# Lưu lại giá trị input cuối cùng vào session state để nhớ giữa các lần rerun do widget tương tác
st.session_state.S_input_val = s_ui
st.session_state.K_input_val = k_ui
st.session_state.T_days_input_val = t_days_ui
st.session_state.r_percent_input_val = r_percent_ui
st.session_state.sigma_input_val = sigma_ui
st.session_state.days_theta_val = days_for_theta_ui

# --- MAIN PANEL ---
st.title("Ứng dụng Tính toán Black-Scholes & Greeks")

t_years_ui = t_days_ui / float(days_for_theta_ui)  # Sử dụng số ngày người dùng chọn
r_annual_ui = r_percent_ui / 100.0

# Tính toán và hiển thị giá
st.header("Giá Quyền chọn")
call_p = european_call_price(s_ui, k_ui, t_years_ui, r_annual_ui, sigma_ui)
put_p = european_put_price(s_ui, k_ui, t_years_ui, r_annual_ui, sigma_ui)
col_call, col_put = st.columns(2)
col_call.metric("Giá Call", "{:.4f}".format(call_p))
col_put.metric("Giá Put", "{:.4f}".format(put_p))

# Tính toán và hiển thị Greeks
st.header("Phân tích Độ nhạy (Greeks)")
greeks_c = get_all_greeks(
    s_ui, k_ui, t_years_ui, r_annual_ui, sigma_ui, "call", days_for_theta_ui
)
greeks_p = get_all_greeks(
    s_ui, k_ui, t_years_ui, r_annual_ui, sigma_ui, "put", days_for_theta_ui
)

greeks_df_data = {
    "Call Option": ["{:.4f}".format(v) for v in greeks_c.values()],
    "Put Option": ["{:.4f}".format(v) for v in greeks_p.values()],
}
greeks_df_index = [name.capitalize() for name in greeks_c.keys()]
greeks_df = pd.DataFrame(greeks_df_data, index=greeks_df_index)
st.table(greeks_df)

st.caption(
    """
    - Vega: Thay đổi giá / 1% thay đổi σ.
    - Theta: Thay đổi giá / 1 ngày.
    - Rho: Thay đổi giá / 1% thay đổi r.
"""
)
