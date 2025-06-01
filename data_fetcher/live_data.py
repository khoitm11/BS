# data_fetcher/live_data.py
import numpy as np
import yfinance as yf

DEFAULT_DATA_PERIOD_FOR_HV = "1y"
DEFAULT_DAYS_WINDOW_FOR_HV = 20


def get_current_price_and_change(
    ticker_symbol: str,
) -> tuple[float | None, float | None, float | None]:
    price_info = None
    prev_close_info = None

    price_hist = None
    prev_close_hist = None

    change_final = None
    pct_change_final = None

    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        # Cố gắng lấy từ info trước
        cp_info_key = info.get("currentPrice")
        rmp_info_key = info.get("regularMarketPrice")
        pc_info_key = info.get("previousClose")
        rmpc_info_key = info.get("regularMarketPreviousClose")

        if cp_info_key is not None:
            price_info = float(cp_info_key)
        elif rmp_info_key is not None:
            price_info = float(rmp_info_key)

        if pc_info_key is not None:
            prev_close_info = float(pc_info_key)
        elif rmpc_info_key is not None:
            prev_close_info = float(rmpc_info_key)

        # Trường hợp lý tưởng: info có đủ
        if price_info is not None and prev_close_info is not None:
            change_final = price_info - prev_close_info
            pct_change_final = (
                (change_final / prev_close_info) * 100 if prev_close_info != 0 else 0.0
            )
            return price_info, change_final, pct_change_final

        # Nếu info không đủ, thử với history
        data_day = ticker.history(period="2d")
        if not data_day.empty and "Close" in data_day.columns and len(data_day) >= 2:
            price_hist = float(data_day["Close"].iloc[-1])
            prev_close_hist = float(data_day["Close"].iloc[-2])

            change_final = price_hist - prev_close_hist
            pct_change_final = (
                (change_final / prev_close_hist) * 100 if prev_close_hist != 0 else 0.0
            )
            return price_hist, change_final, pct_change_final
        elif not data_day.empty and "Close" in data_day.columns and len(data_day) == 1:
            # Chỉ có 1 dòng trong history, lấy giá nhưng không có change
            price_hist = float(data_day["Close"].iloc[-1])
            return price_hist, None, None

        # Nếu history cũng không đủ, nhưng info có giá (dù không có prev_close)
        if price_info is not None:
            return price_info, None, None  # Trả về giá từ info, không có change

        return None, None, None  # Không lấy được từ nguồn nào

    except Exception:
        return None, None, None


def calculate_historical_volatility_annualized(
    ticker_symbol: str,
    hv_period: str = DEFAULT_DATA_PERIOD_FOR_HV,
    hv_window_days: int = DEFAULT_DAYS_WINDOW_FOR_HV,
) -> float | None:
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist_data = ticker.history(period=hv_period)

        if hist_data.empty or len(hist_data) < hv_window_days + 2:
            return None

        log_returns = np.log(hist_data["Close"] / hist_data["Close"].shift(1))

        if log_returns.iloc[1:].empty:
            return None

        daily_std_dev_rolling = (
            log_returns.iloc[1:].rolling(window=hv_window_days).std()
        )

        if daily_std_dev_rolling.dropna().empty:
            return None

        last_valid_daily_std_dev = daily_std_dev_rolling.dropna().iloc[-1]
        annualized_hv = last_valid_daily_std_dev * np.sqrt(252)

        return float(annualized_hv) if not np.isnan(annualized_hv) else None
    except Exception:
        return None
