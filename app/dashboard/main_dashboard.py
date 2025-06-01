# app/dashboard/main_dashboard.py
import os
import sys
import time

import httpx
import pandas as pd
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Các import này có thể vẫn cần noqa nếu CI của bạn báo lỗi E402

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(layout="wide", page_title="Black-Scholes Lab", page_icon="🚀")

SESSION_DEFAULTS = {
    "ticker_symbol": "AAPL",
    "current_S_market": 150.0,
    "price_change": None,
    "price_pct_change": None,
    "hv_calculated_market": 0.20,
    "hv_window_used": 20,
    "last_fetch_timestamp": 0,
    "S_input_val": 150.0,
    "sigma_input_val": 0.20,
    "K_input_val": 150.0,
    "T_days_input_val": 30,
    "r_percent_input_val": 5.0,
    "days_theta_val": 365,
    "hv_window_selected_for_api": 20,
}
for k, v in SESSION_DEFAULTS.items():
    st.session_state.setdefault(k, v)


def api_request(
    method: str, endpoint: str, params: dict = None, json_payload: dict = None
) -> dict | None:
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method.upper() == "GET":
            response = httpx.get(url, params=params, timeout=10)
        elif method.upper() == "POST":
            response = httpx.post(url, json=json_payload, timeout=10)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        error_detail = exc.response.json().get("detail", exc.reason_phrase)
        st.error(f"API Error ({exc.response.status_code}): {error_detail}")
    except httpx.RequestError as exc:
        st.error(f"Connection Error: {exc}")
    except Exception as e:
        st.error(f"Unexpected Error: {type(e).__name__} - {e}")
    return None


def fetch_market_data_via_api(ticker: str):
    st.session_state.last_fetch_timestamp = time.time()
    hv_window = st.session_state.hv_window_selected_for_api

    market_data = api_request(
        "GET", f"/marketdata/{ticker}", params={"hv_window": hv_window}
    )

    if market_data:
        current_price = market_data.get("current_price")
        st.session_state.current_S_market = (
            current_price
            if current_price is not None
            else SESSION_DEFAULTS["current_S_market"]
        )
        st.session_state.S_input_val = st.session_state.current_S_market
        st.session_state.price_change = market_data.get("price_change")
        st.session_state.price_pct_change = market_data.get("price_percent_change")

        hv_val = market_data.get("historical_volatility_calculated")
        st.session_state.hv_calculated_market = (
            hv_val if hv_val is not None else SESSION_DEFAULTS["hv_calculated_market"]
        )
        st.session_state.sigma_input_val = st.session_state.hv_calculated_market
        st.session_state.hv_window_used = market_data.get(
            "hv_window_days_used", hv_window
        )
        if hv_val is None and current_price is not None:
            st.sidebar.warning(f"API: No HV{hv_window} for {ticker}.")
    else:
        st.sidebar.warning(f"API: No market data for {ticker}.")
        for key in [
            "current_S_market",
            "S_input_val",
            "price_change",
            "price_pct_change",
            "hv_calculated_market",
            "sigma_input_val",
            "hv_window_used",
        ]:
            st.session_state[key] = SESSION_DEFAULTS[key]


st.sidebar.title("Parameters")
st.session_state.ticker_symbol = st.sidebar.text_input(
    "Ticker", st.session_state.ticker_symbol
).upper()
hv_options = [20, 60, 90, 120]
st.session_state.hv_window_selected_for_api = st.sidebar.selectbox(
    "HV Window (days)",
    hv_options,
    index=hv_options.index(st.session_state.hv_window_selected_for_api),
)

if st.sidebar.button(
    f"Fetch API Data for {st.session_state.ticker_symbol}", key="fetch_api_button"
):
    if st.session_state.ticker_symbol:
        with st.spinner("Fetching API data..."):
            fetch_market_data_via_api(st.session_state.ticker_symbol)
        st.rerun()
    else:
        st.sidebar.error("Enter a ticker.")

st.sidebar.markdown("---")
st.sidebar.subheader("Market Values (API)")
price_str = f"{st.session_state.current_S_market:.2f}"
if (
    st.session_state.price_change is not None
    and st.session_state.price_pct_change is not None
):
    sign = "+" if st.session_state.price_change >= 0 else ""
    price_str += f" ({sign}{st.session_state.price_change:.2f}, {sign}{st.session_state.price_pct_change * 100:.2f}%)"
st.sidebar.markdown(f"**Price (S):** {price_str}")
hv_win_disp = st.session_state.hv_window_used
st.sidebar.markdown(
    f"**HV{hv_win_disp} (σ):** {st.session_state.hv_calculated_market * 100:.2f}%"
)
st.sidebar.markdown("---")

st.sidebar.subheader("Black-Scholes Inputs")
s_val = st.sidebar.number_input(
    "Asset Price (S)",
    float(st.session_state.S_input_val),
    step=0.01,
    format="%.2f",
    key="s_ui",
)
k_val = st.sidebar.number_input(
    "Strike Price (K)",
    float(st.session_state.get("K_input_val", s_val)),
    step=0.50,
    format="%.2f",
    key="k_ui",
)
t_days = st.sidebar.number_input(
    "Time to Maturity (days)",
    1,
    int(st.session_state.T_days_input_val),
    step=1,
    key="t_ui",
)
r_pct = st.sidebar.slider(
    "Risk-Free Rate (r %)",
    0.0,
    15.0,
    float(st.session_state.r_percent_input_val),
    0.1,
    key="r_ui",
)
sigma_val = st.sidebar.slider(
    "Annual Volatility (σ)",
    0.0001,
    2.0000,
    float(st.session_state.sigma_input_val),
    0.0010,
    format="%.4f",
    key="sigma_ui",
    help=f"Suggest HV{hv_win_disp}: {st.session_state.hv_calculated_market * 100:.2f}%",
)
days_theta = st.sidebar.selectbox(
    "Days in Year (for Theta)",
    [365, 252],
    index=[365, 252].index(st.session_state.days_theta_val),
    key="days_theta_ui",
)

st.title("Black-Scholes & Greeks (API-Driven)")

t_yrs = t_days / float(days_theta)
r_ann = r_pct / 100.0

st.header("Option Prices (API)")
cols_price = st.columns(2)
price_payload = {"S": s_val, "K": k_val, "T": t_yrs, "r": r_ann, "sigma": sigma_val}

call_resp = api_request(
    "POST", "/options/price", json_payload={**price_payload, "option_type": "call"}
)
put_resp = api_request(
    "POST", "/options/price", json_payload={**price_payload, "option_type": "put"}
)

call_p = call_resp.get("calculated_price") if call_resp else None
put_p = put_resp.get("calculated_price") if put_resp else None

cols_price[0].metric(
    "Call Price", f"{call_p:.4f}" if isinstance(call_p, float) else "N/A"
)
cols_price[1].metric("Put Price", f"{put_p:.4f}" if isinstance(put_p, float) else "N/A")

st.header("Greeks Analysis (API)")
greeks_payload = {**price_payload, "days_in_year_for_theta": days_theta}
GREEKS_KEYS = ["delta", "gamma", "vega", "theta", "rho"]

greeks_call_resp = api_request(
    "POST", "/options/greeks", json_payload={**greeks_payload, "option_type": "call"}
)
greeks_put_resp = api_request(
    "POST", "/options/greeks", json_payload={**greeks_payload, "option_type": "put"}
)

if (
    greeks_call_resp
    and greeks_put_resp
    and all(k in greeks_call_resp for k in GREEKS_KEYS)
    and all(k in greeks_put_resp for k in GREEKS_KEYS)
):

    data = {
        "Call": [f"{greeks_call_resp[k]:.4f}" for k in GREEKS_KEYS],
        "Put": [f"{greeks_put_resp[k]:.4f}" for k in GREEKS_KEYS],
    }
    df_greeks = pd.DataFrame(data, index=[k.capitalize() for k in GREEKS_KEYS])
    st.table(df_greeks)
else:
    st.error("API: Cannot load Greeks data.")
    if greeks_call_resp:
        st.json({"Call Greeks RAW": greeks_call_resp})  # Debugging
    if greeks_put_resp:
        st.json({"Put Greeks RAW": greeks_put_resp})  # Debugging

st.caption(
    "- Vega: Price change / 1% σ change.\n- Theta: Price change / 1 day.\n- Rho: Price change / 1% r change."
)
