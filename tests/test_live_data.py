# tests/test_live_data.py
import pytest
import pandas as pd
import numpy as np
from unittest import mock

from data_fetcher.live_data import (
    get_current_price_and_change,
    calculate_historical_volatility_annualized,
)


class MockYFinanceTicker:
    def __init__(self, ticker_symbol="MOCKTICKER", info_data=None, history_data=None):
        self.ticker_symbol = ticker_symbol
        self.mock_info_data = info_data
        self.mock_history_data = history_data

    @property
    def info(self):
        if self.ticker_symbol == "RAISE_ERROR":
            raise ValueError("Simulated API error in info")
        return self.mock_info_data if self.mock_info_data is not None else {}

    def history(self, period=None, interval=None):
        if self.ticker_symbol == "RAISE_ERROR":
            raise ValueError("Simulated API error in history")
        return (
            self.mock_history_data
            if self.mock_history_data is not None
            else pd.DataFrame({"Close": []})
        )


@mock.patch("yfinance.Ticker", MockYFinanceTicker)
def test_get_current_price_from_full_info():
    info_fixture = {"currentPrice": 150.50, "previousClose": 148.00}
    with mock.patch(
        "yfinance.Ticker",
        lambda ticker: MockYFinanceTicker(ticker, info_data=info_fixture),
    ):
        price, change, pct_change = get_current_price_and_change("MOCK_FULL_INFO")
        assert price == 150.50
        assert change == pytest.approx(150.50 - 148.00)
        assert pct_change == pytest.approx(((150.50 - 148.00) / 148.00) * 100)


@mock.patch("yfinance.Ticker", MockYFinanceTicker)
def test_get_current_price_from_regular_market_info():
    info_fixture = {
        "regularMarketPrice": 150.00,
        "regularMarketPreviousClose": 147.50,
    }  # Không có currentPrice, previousClose
    with mock.patch(
        "yfinance.Ticker",
        lambda ticker: MockYFinanceTicker(ticker, info_data=info_fixture),
    ):
        price, change, pct_change = get_current_price_and_change("MOCK_REGULAR_INFO")
        assert price == 150.00
        assert change == pytest.approx(150.00 - 147.50)
        assert pct_change == pytest.approx(((150.00 - 147.50) / 147.50) * 100)


@mock.patch("yfinance.Ticker", MockYFinanceTicker)
def test_get_current_price_info_priority():  # currentPrice ưu tiên hơn regularMarketPrice
    info_fixture = {
        "currentPrice": 151.00,
        "regularMarketPrice": 150.00,
        "previousClose": 149.00,
        "regularMarketPreviousClose": 148.00,
    }
    with mock.patch(
        "yfinance.Ticker",
        lambda ticker: MockYFinanceTicker(ticker, info_data=info_fixture),
    ):
        price, change, pct_change = get_current_price_and_change("MOCK_PRIORITY")
        assert price == 151.00
        assert change == pytest.approx(151.00 - 149.00)
        assert pct_change == pytest.approx(((151.00 - 149.00) / 149.00) * 100)


@mock.patch("yfinance.Ticker", MockYFinanceTicker)
def test_get_current_price_fallback_to_history_when_info_incomplete():
    info_fixture_missing_prev_close = {"currentPrice": 150.0}  # Thiếu prev close
    history_fixture = pd.DataFrame(
        {"Close": [145.0, 146.0]},  # Ngày gần nhất là 146, trước đó 145
        index=pd.date_range(end=pd.Timestamp.today(), periods=2, freq="B"),
    )

    with mock.patch(
        "yfinance.Ticker",
        lambda ticker: MockYFinanceTicker(
            ticker,
            info_data=info_fixture_missing_prev_close,
            history_data=history_fixture,
        ),
    ):
        price, change, pct_change = get_current_price_and_change("MOCK_FALLBACK_HIST")
        assert price == 146.0  # Giá phải lấy từ history
        assert change == pytest.approx(146.0 - 145.0)
        assert pct_change == pytest.approx(((146.0 - 145.0) / 145.0) * 100)


@mock.patch("yfinance.Ticker", MockYFinanceTicker)
def test_get_current_price_fallback_to_history_when_info_no_price():
    info_fixture_no_price = {"previousClose": 140.0}  # Thiếu giá hiện tại
    history_fixture = pd.DataFrame(
        {"Close": [142.0, 143.0]},
        index=pd.date_range(end=pd.Timestamp.today(), periods=2, freq="B"),
    )
    with mock.patch(
        "yfinance.Ticker",
        lambda ticker: MockYFinanceTicker(
            ticker, info_data=info_fixture_no_price, history_data=history_fixture
        ),
    ):
        price, change, pct_change = get_current_price_and_change("MOCK_FALLBACK_HIST2")
        assert price == 143.0
        assert change == pytest.approx(143.0 - 142.0)
        assert pct_change == pytest.approx(((143.0 - 142.0) / 142.0) * 100)


@mock.patch("yfinance.Ticker", MockYFinanceTicker)
def test_get_current_price_history_only_one_day():
    history_fixture_one_day = pd.DataFrame(
        {"Close": [150.0]},  # Chỉ có 1 ngày, không tính được change
        index=pd.date_range(end=pd.Timestamp.today(), periods=1, freq="B"),
    )
    with mock.patch(
        "yfinance.Ticker",
        lambda ticker: MockYFinanceTicker(
            ticker, info_data={}, history_data=history_fixture_one_day
        ),
    ):
        price, change, pct_change = get_current_price_and_change("MOCK_HIST_ONE_DAY")
        assert price == 150.0
        assert change is None
        assert pct_change is None


@mock.patch("yfinance.Ticker", MockYFinanceTicker)
def test_get_current_price_info_has_price_history_empty():
    # Info có giá, nhưng không có PCoS. History lại rỗng.
    info_fixture = {"currentPrice": 160.0}
    history_fixture_empty = pd.DataFrame({"Close": []})
    with mock.patch(
        "yfinance.Ticker",
        lambda ticker: MockYFinanceTicker(
            ticker, info_data=info_fixture, history_data=history_fixture_empty
        ),
    ):
        price, change, pct_change = get_current_price_and_change(
            "MOCK_PRICE_INFO_NO_HIST"
        )
        assert price == 160.0  # Lấy giá từ info
        assert change is None  # Không có PCoS từ info, history rỗng
        assert pct_change is None


@mock.patch("yfinance.Ticker", MockYFinanceTicker)
def test_get_current_price_failure_all_sources_empty():
    with mock.patch(
        "yfinance.Ticker",
        lambda ticker: MockYFinanceTicker(
            ticker, info_data={}, history_data=pd.DataFrame({"Close": []})
        ),
    ):
        price, change, pct_change = get_current_price_and_change("MOCK_ALL_EMPTY")
        assert price is None
        assert change is None
        assert pct_change is None


@mock.patch("yfinance.Ticker", MockYFinanceTicker)
def test_get_current_price_api_error_returns_none():
    price, change, pct_change = get_current_price_and_change("RAISE_ERROR")
    assert price is None
    assert change is None
    assert pct_change is None


@mock.patch("yfinance.Ticker", MockYFinanceTicker)
def test_calculate_hv_annualized_success():
    # Tạo history data mẫu cho MockYFinanceTicker
    dates = pd.date_range(
        end=pd.Timestamp.today() - pd.Timedelta(days=1), periods=60, freq="B"
    )
    prices = np.linspace(100, 150, 60) + np.random.normal(0, 1, 60)
    history_sample = pd.DataFrame({"Close": prices}, index=dates)
    with mock.patch(
        "yfinance.Ticker",
        lambda ticker: MockYFinanceTicker(ticker, history_data=history_sample),
    ):
        hv = calculate_historical_volatility_annualized("MOCK_HV_OK", hv_window_days=20)
        assert hv is not None
        assert isinstance(hv, float)
        assert hv > 0


@mock.patch("yfinance.Ticker", MockYFinanceTicker)
def test_calculate_hv_history_empty():
    with mock.patch(
        "yfinance.Ticker",
        lambda ticker: MockYFinanceTicker(
            ticker, history_data=pd.DataFrame({"Close": []})
        ),
    ):
        hv = calculate_historical_volatility_annualized(
            "MOCK_HV_EMPTY", hv_window_days=20
        )
        assert hv is None


@mock.patch("yfinance.Ticker", MockYFinanceTicker)
def test_calculate_hv_history_too_short_for_calc():
    # hv_window_days + 2
    short_hist = pd.DataFrame(
        {"Close": [100] * 21},
        index=pd.date_range(end=pd.Timestamp.today(), periods=21, freq="B"),
    )
    with mock.patch(
        "yfinance.Ticker",
        lambda ticker: MockYFinanceTicker(ticker, history_data=short_hist),
    ):
        hv = calculate_historical_volatility_annualized(
            "MOCK_HV_SHORT_CALC", hv_window_days=20
        )
        assert hv is None


@mock.patch("yfinance.Ticker", MockYFinanceTicker)
def test_calculate_hv_history_just_enough_for_one_std_value():
    # hv_window_days + 1 cho shift, thêm 1 để rolling không rỗng sau dropna
    just_enough_hist = pd.DataFrame(
        {"Close": np.linspace(100, 105, 22)},
        index=pd.date_range(end=pd.Timestamp.today(), periods=22, freq="B"),
    )
    with mock.patch(
        "yfinance.Ticker",
        lambda ticker: MockYFinanceTicker(ticker, history_data=just_enough_hist),
    ):
        hv = calculate_historical_volatility_annualized(
            "MOCK_HV_JUST_ENOUGH", hv_window_days=20
        )
        assert hv is not None
        assert hv > 0  # Sẽ tính được 1 giá trị HV


@mock.patch("yfinance.Ticker", MockYFinanceTicker)
def test_calculate_hv_api_error():
    hv = calculate_historical_volatility_annualized("RAISE_ERROR", hv_window_days=20)
    assert hv is None


@mock.patch("yfinance.Ticker", MockYFinanceTicker)
def test_calculate_hv_constant_price_returns_zero():
    prices_const = [100.0] * 60
    const_hist_df = pd.DataFrame(
        {"Close": prices_const},
        index=pd.date_range(end=pd.Timestamp.today(), periods=60, freq="B"),
    )
    with mock.patch(
        "yfinance.Ticker",
        lambda ticker: MockYFinanceTicker(ticker, history_data=const_hist_df),
    ):
        hv = calculate_historical_volatility_annualized(
            "MOCK_CONST_PRICE_HV", hv_window_days=20
        )
        assert hv == pytest.approx(0.0)
