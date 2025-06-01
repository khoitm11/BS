# app/dashboard/main_dashboard.py
import os
import sys
import time

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --- Cấu hình cơ bản ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_DAYS_WINDOW_FOR_HV = 20

# Sử dụng theme của Streamlit cho Plotly để đồng nhất
pio.templates.default = "streamlit"

st.set_page_config(
    layout="wide", page_title="Black-Scholes Pro Dashboard", page_icon="💹"
)

# --- Khởi tạo Session State ---
SESSION_DEFAULTS = {
    "ticker_symbol": "AAPL",
    "current_S_market": 150.0,
    "price_change": 0.0,
    "price_pct_change": 0.0,
    "hv_calculated_market": 0.20,
    "hv_window_used": DEFAULT_DAYS_WINDOW_FOR_HV,
    "last_successful_fetch_timestamp": 0,
    "S_input_val": 150.0,
    "sigma_input_val": 0.20,
    "K_input_val": 150.0,
    "T_days_input_val": 30,
    "r_percent_input_val": 5.0,
    "days_theta_val": 365,
    "auto_refresh_interval": 300,  # Giây
    "is_auto_refreshing": False,
    "api_error_message": None,
    "hv_window_setting": DEFAULT_DAYS_WINDOW_FOR_HV,
}
for k, v in SESSION_DEFAULTS.items():
    st.session_state.setdefault(k, v)


# --- Hàm gọi API (đã cache cho market data) ---
@st.cache_data(ttl=60, show_spinner=False)  # Cache 60s, tắt spinner mặc định của cache
def call_market_data_api_cached(ticker: str, hv_window_api: int) -> dict | None:
    url = f"{API_BASE_URL}/marketdata/{ticker}"
    params = {"hv_window": hv_window_api}
    try:
        response = httpx.get(url, params=params, timeout=10)
        response.raise_for_status()
        st.session_state.api_error_message = None
        return response.json()
    except httpx.HTTPStatusError as exc:
        st.session_state.api_error_message = f"MarketData API ({exc.response.status_code}): {exc.response.json().get('detail', exc.reason_phrase)}"
    except httpx.RequestError as exc:
        st.session_state.api_error_message = f"MarketData API Connection Error: {exc}"
    except Exception as e:
        st.session_state.api_error_message = (
            f"MarketData API Unknown Error: {type(e).__name__}"
        )
    return None


def call_bs_api(endpoint: str, payload: dict) -> dict | None:
    url = f"{API_BASE_URL}/options/{endpoint}"
    try:
        response = httpx.post(url, json=payload, timeout=10)
        response.raise_for_status()
        st.session_state.api_error_message = None
        return response.json()
    except httpx.HTTPStatusError as exc:
        st.session_state.api_error_message = f"Options/{endpoint} API ({exc.response.status_code}): {exc.response.json().get('detail', exc.reason_phrase)}"
    except httpx.RequestError as exc:
        st.session_state.api_error_message = (
            f"Options/{endpoint} API Connection Error: {exc}"
        )
    except Exception as e:
        st.session_state.api_error_message = (
            f"Options/{endpoint} API Unknown Error: {type(e).__name__}"
        )
    return None


# --- Logic cập nhật dữ liệu ---
def update_market_data_state(ticker: str):
    st.session_state.api_error_message = None
    hv_window = st.session_state.hv_window_setting

    with st.spinner(
        f"Fetching {ticker} data (HV{hv_window})..."
    ):  # Spinner cụ thể cho API call
        market_data = call_market_data_api_cached(ticker, hv_window)

    if market_data and market_data.get("current_price") is not None:
        st.session_state.current_S_market = market_data["current_price"]
        st.session_state.price_change = market_data.get("price_change", 0.0)
        st.session_state.price_pct_change = market_data.get("price_percent_change", 0.0)
        st.session_state.S_input_val = market_data["current_price"]

        hv = market_data.get("historical_volatility_calculated")
        if hv is not None:
            st.session_state.hv_calculated_market = hv
            st.session_state.sigma_input_val = hv
            st.session_state.hv_window_used = market_data.get(
                "hv_window_days_used", hv_window
            )
        else:
            st.session_state.hv_calculated_market = SESSION_DEFAULTS[
                "hv_calculated_market"
            ]
            st.session_state.sigma_input_val = SESSION_DEFAULTS["sigma_input_val"]
            st.session_state.hv_window_used = hv_window
            if (
                not st.session_state.api_error_message
                and market_data.get("current_price") is not None
            ):  # Chỉ cảnh báo nếu giá có mà HV không có
                st.toast(f"API: No HV{hv_window} for {ticker}.", icon="⚠️")
        st.session_state.last_successful_fetch_timestamp = time.time()
    else:
        if not st.session_state.api_error_message:
            st.toast(f"API: No market data for {ticker}.", icon="❌")
        # Reset về default nếu API call thất bại hoặc không có giá
        for key in [
            "current_S_market",
            "S_input_val",
            "price_change",
            "price_pct_change",
            "hv_calculated_market",
            "sigma_input_val",
        ]:
            st.session_state[key] = SESSION_DEFAULTS[key]
        st.session_state.hv_window_used = hv_window


# --- Hàm hiển thị UI ---
def display_sidebar():
    with st.sidebar:
        st.title("⚙️ Controls")

        current_ticker = st.session_state.ticker_symbol
        new_ticker = st.text_input(
            "Ticker Symbol", current_ticker, key="ticker_input_key"
        ).upper()
        if new_ticker != current_ticker and new_ticker:
            st.session_state.ticker_symbol = new_ticker
            st.session_state.last_successful_fetch_timestamp = 0
            st.rerun()

        hv_options = [20, 60, 90, 120]  # Các lựa chọn phổ biến
        st.session_state.hv_window_setting = st.selectbox(
            "Historical Volatility Window (days)",
            hv_options,
            index=hv_options.index(st.session_state.hv_window_setting),
            key="hv_window_selector_key",
        )

        if st.button(
            f"Fetch Market Data for {st.session_state.ticker_symbol}",
            use_container_width=True,
            key="fetch_data_button",
        ):
            if st.session_state.ticker_symbol:
                update_market_data_state(st.session_state.ticker_symbol)
                st.rerun()
            else:
                st.error("Please enter a ticker symbol.")

        st.markdown("---")
        st.subheader("Market Snapshot")
        s_market, p_change, pct_change = (
            st.session_state.current_S_market,
            st.session_state.price_change,
            st.session_state.price_pct_change,
        )

        price_str = f"{s_market:.2f}"
        delta_str = ""
        if p_change is not None and pct_change is not None:
            sign = "🔼" if p_change >= 0 else "🔽"
            # Sử dụng f-string thông thường, không cần phức tạp
            delta_str = f"{sign} {abs(p_change):.2f} ({abs(pct_change * 100):.2f}%)"

        st.metric(
            label="Current Price (S)",
            value=price_str,
            delta=delta_str if p_change != 0 else None,
        )

        hv_win, hv_market = (
            st.session_state.hv_window_used,
            st.session_state.hv_calculated_market,
        )
        st.metric(label=f"HV{hv_win} (Implied σ)", value=f"{hv_market * 100:.2f}%")

        last_fetch_time_str = (
            time.strftime(
                "%H:%M:%S",
                time.localtime(st.session_state.last_successful_fetch_timestamp),
            )
            if st.session_state.last_successful_fetch_timestamp > 0
            else "N/A"
        )
        st.caption(f"Last update: {last_fetch_time_str}")

        st.markdown("---")
        st.subheader("Black-Scholes Inputs")
        # Sử dụng key để Streamlit quản lý state của widget tốt hơn
        st.session_state.S_input_val = st.number_input(
            "Asset Price (S)",
            float(st.session_state.S_input_val),
            step=0.01,
            format="%.2f",
            key="s_input_sidebar",
        )
        st.session_state.K_input_val = st.number_input(
            "Strike Price (K)",
            float(st.session_state.get("K_input_val", st.session_state.S_input_val)),
            step=0.50,
            format="%.2f",
            key="k_input_sidebar",
        )
        st.session_state.T_days_input_val = st.number_input(
            "Time to Maturity (days)",
            1,
            int(st.session_state.T_days_input_val),
            step=1,
            key="t_days_sidebar",
        )
        st.session_state.r_percent_input_val = st.slider(
            "Risk-Free Rate (r %)",
            0.0,
            15.0,
            float(st.session_state.r_percent_input_val),
            0.1,
            key="r_pct_sidebar",
        )
        st.session_state.sigma_input_val = st.slider(
            "Annual Volatility (σ)",
            0.0001,
            2.0,
            float(st.session_state.sigma_input_val),
            0.001,
            format="%.4f",
            key="sigma_sidebar",
            help=f"Implied from HV{hv_win}: {hv_market * 100:.2f}%",
        )
        st.session_state.days_theta_val = st.selectbox(
            "Days/Year (for Theta calc)",
            [365, 252],
            index=[365, 252].index(st.session_state.days_theta_val),
            key="days_theta_sidebar",
        )

        st.markdown("---")
        st.subheader("Auto Refresh")
        col1_refresh, col2_interval = st.columns([1, 2])
        with col1_refresh:
            st.session_state.is_auto_refreshing = st.toggle(
                "Enable", st.session_state.is_auto_refreshing, key="auto_refresh_toggle"
            )
        with col2_interval:
            st.session_state.auto_refresh_interval = st.selectbox(
                "Interval (s)",
                [30, 60, 120, 300, 600],
                index=[30, 60, 120, 300, 600].index(
                    st.session_state.auto_refresh_interval
                ),
                disabled=not st.session_state.is_auto_refreshing,
                key="auto_refresh_interval_selector",
            )
        if st.session_state.is_auto_refreshing:
            time_left = int(
                st.session_state.auto_refresh_interval
                - (time.time() - st.session_state.last_successful_fetch_timestamp)
            )
            if time_left > 0:
                st.caption(f"Next refresh in: {time_left}s")
            else:
                st.caption("Refreshing soon...")


def display_option_pricing_and_greeks_tab():
    S, K, T_days, r_pct, sigma, days_yr = (
        st.session_state.S_input_val,
        st.session_state.K_input_val,
        st.session_state.T_days_input_val,
        st.session_state.r_percent_input_val,
        st.session_state.sigma_input_val,
        st.session_state.days_theta_val,
    )
    T_yrs, r_ann = T_days / float(days_yr), r_pct / 100.0

    if T_yrs <= 0 or sigma <= 0:
        st.warning("Time to Maturity (T) and Volatility (σ) must be greater than 0.")
        return

    price_payload = {"S": S, "K": K, "T": T_yrs, "r": r_ann, "sigma": sigma}
    greeks_payload = {**price_payload, "days_in_year_for_theta": days_yr}

    col_calc_price, col_calc_greeks = st.columns(2)

    with col_calc_price:
        st.subheader("Option Prices")
        call_p_resp = call_bs_api("price", {**price_payload, "option_type": "call"})
        put_p_resp = call_bs_api("price", {**price_payload, "option_type": "put"})
        call_p = call_p_resp.get("calculated_price") if call_p_resp else None
        put_p = put_p_resp.get("calculated_price") if put_p_resp else None

        st.metric("Call Price", f"{call_p:.4f}" if isinstance(call_p, float) else "N/A")
        st.metric("Put Price", f"{put_p:.4f}" if isinstance(put_p, float) else "N/A")

    with col_calc_greeks:
        st.subheader("Option Greeks")
        greeks_c_resp = call_bs_api("greeks", {**greeks_payload, "option_type": "call"})
        greeks_p_resp = call_bs_api("greeks", {**greeks_payload, "option_type": "put"})
        GREEKS_KEYS = ["delta", "gamma", "vega", "theta", "rho"]

        if (
            greeks_c_resp
            and greeks_p_resp
            and all(k in greeks_c_resp for k in GREEKS_KEYS)
            and all(k in greeks_p_resp for k in GREEKS_KEYS)
        ):
            data = {
                "Call": [greeks_c_resp[k] for k in GREEKS_KEYS],
                "Put": [greeks_p_resp[k] for k in GREEKS_KEYS],
            }
            df_greeks = pd.DataFrame(data, index=[k.capitalize() for k in GREEKS_KEYS])
            # Sử dụng st.dataframe để có thể tùy chỉnh style tốt hơn nếu muốn, hoặc st.table cho đơn giản
            st.dataframe(df_greeks.style.format("{:.4f}"), use_container_width=True)
        else:
            st.error("Could not load full Greeks data from API.")

    st.caption(
        """
        - Vega: Price change for a 1 percentage point change in σ.
        - Theta: Price change per day (based on selected days/year).
        - Rho: Price change for a 1 percentage point change in r.
        """
    )


def display_greeks_charts_tab():
    st.subheader("Greeks Sensitivity Charts vs. Asset Price (S)")
    S, K, T_days, r_pct, sigma, days_yr = (
        st.session_state.S_input_val,
        st.session_state.K_input_val,
        st.session_state.T_days_input_val,
        st.session_state.r_percent_input_val,
        st.session_state.sigma_input_val,
        st.session_state.days_theta_val,
    )
    T_yrs, r_ann = T_days / float(days_yr), r_pct / 100.0

    if T_yrs <= 0 or sigma <= 0:
        st.warning("Set T > 0 and σ > 0 to view charts.")
        return

    s_min, s_max = max(0.1, S * 0.7), S * 1.3  # Đảm bảo S không quá gần 0
    s_range = np.linspace(
        s_min, s_max, 40
    )  # Giảm số điểm để API call nhanh hơn cho chart

    chart_data, GREEKS_TO_CHART = [], ["delta", "gamma", "vega", "theta"]
    api_call_count = 0  # Đếm số lần gọi API

    # Hiển thị spinner trong khi fetch dữ liệu cho chart
    with st.spinner("Calculating chart data via API... This may take a moment."):
        for s_i in s_range:
            if (
                api_call_count >= 80
            ):  # Giới hạn số lần gọi API để tránh treo (40 S_points * 2 calls)
                st.warning(
                    "Chart data generation limited to avoid excessive API calls."
                )
                break
            p_load = {"S": s_i, "K": K, "T": T_yrs, "r": r_ann, "sigma": sigma}
            g_load = {**p_load, "days_in_year_for_theta": days_yr}
            gc = call_bs_api("greeks", {**g_load, "option_type": "call"})
            api_call_count += 1
            gp = call_bs_api("greeks", {**g_load, "option_type": "put"})
            api_call_count += 1

            if gc and gp and all(k in gc for k in GREEKS_TO_CHART):
                chart_data.append(
                    {
                        "S": s_i,
                        "Call Delta": gc["delta"],
                        "Put Delta": gp["delta"],
                        "Gamma": gc["gamma"],
                        "Vega": gc["vega"],
                        "Call Theta": gc["theta"],
                    }
                )

    if not chart_data:
        st.warning("Could not generate sufficient data for charts from API.")
        return

    df_chart = pd.DataFrame(chart_data)

    col1_charts, col2_charts = st.columns(2)
    hover_template_greek = (
        "<b>S = %{x:.2f}</b><br>%{fullData.name} = %{y:.4f}<extra></extra>"
    )

    with col1_charts:
        fig_dg = go.Figure()
        fig_dg.add_trace(
            go.Scatter(
                x=df_chart["S"],
                y=df_chart["Call Delta"],
                name="Call Delta",
                line=dict(width=2),
                hovertemplate=hover_template_greek,
            )
        )
        fig_dg.add_trace(
            go.Scatter(
                x=df_chart["S"],
                y=df_chart["Put Delta"],
                name="Put Delta",
                line=dict(width=2),
                hovertemplate=hover_template_greek,
            )
        )
        fig_dg.add_trace(
            go.Scatter(
                x=df_chart["S"],
                y=df_chart["Gamma"],
                name="Gamma",
                yaxis="y2",
                line=dict(dash="dash", width=2),
                hovertemplate=hover_template_greek,
            )
        )
        fig_dg.update_layout(
            title_text="Delta & Gamma vs. Asset Price",
            yaxis_title="Delta",
            yaxis2=dict(
                title="Gamma",
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False,
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig_dg, use_container_width=True)

    with col2_charts:
        fig_vt = go.Figure()
        fig_vt.add_trace(
            go.Scatter(
                x=df_chart["S"],
                y=df_chart["Vega"],
                name="Vega",
                line=dict(width=2),
                hovertemplate=hover_template_greek,
            )
        )
        fig_vt.add_trace(
            go.Scatter(
                x=df_chart["S"],
                y=df_chart["Call Theta"],
                name="Call Theta",
                yaxis="y2",
                line=dict(dash="dash", width=2),
                hovertemplate=hover_template_greek,
            )
        )
        fig_vt.update_layout(
            title_text="Vega & Call Theta vs. Asset Price",
            yaxis_title="Vega",
            yaxis2=dict(
                title="Call Theta (Daily)",
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False,
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig_vt, use_container_width=True)


# --- Main App Flow ---
st.title("📈 Black-Scholes Pro Option Dashboard")

if st.session_state.api_error_message:
    st.error(st.session_state.api_error_message)  # Hiển thị lỗi API một cách nổi bật

display_sidebar()  # Gọi để vẽ sidebar trước

# Sử dụng tabs cho nội dung chính
tab_pricing, tab_charts = st.tabs(
    ["📊 Option Pricing & Greeks Table", "📉 Greeks Sensitivity Charts"]
)

with tab_pricing:
    display_option_pricing_and_greeks_tab()

with tab_charts:
    display_greeks_charts_tab()

# --- Auto Refresh Logic ---
if st.session_state.is_auto_refreshing:
    if (
        time.time() - st.session_state.last_successful_fetch_timestamp
        > st.session_state.auto_refresh_interval
    ):
        if st.session_state.ticker_symbol:
            # Không cần spinner ở đây vì update_market_data_state đã có spinner riêng
            update_market_data_state(st.session_state.ticker_symbol)
            # Cần clear cache của market data để lần refresh tiếp theo lấy dữ liệu mới
            call_market_data_api_cached.clear()
            st.rerun()

            # Tạo một "placeholder" để chạy lại script sau một khoảng thời gian ngắn cho auto-refresh
    # Điều này giúp cập nhật bộ đếm ngược mà không block thread chính
    # Tuy nhiên, nó vẫn sẽ trigger reruns. Cân nhắc tần suất.
    if st.session_state.is_auto_refreshing:  # Chỉ làm nếu đang bật
        time.sleep(1)  # Chạy lại script mỗi giây để cập nhật caption đếm ngược
        st.rerun()
