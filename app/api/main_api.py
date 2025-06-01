# app/api/main_api.py
import sys
import os
import numpy as np  # Cần cho np.isnan
from fastapi import FastAPI, HTTPException, Query, Path

# Thêm thư mục gốc vào sys.path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(current_file_dir))
)  # Đi lên 3 cấp từ app/api -> project_root
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from app.api.models_api import (
    OptionPriceRequest,
    OptionPriceResponse,
    OptionGreeksRequest,
    OptionGreeksResponse,
    MarketDataResponse,
)
from core.black_scholes import european_call_price, european_put_price, get_all_greeks
from data_fetcher.live_data import (
    get_current_price_and_change,
    calculate_historical_volatility_annualized,
    DEFAULT_DAYS_WINDOW_FOR_HV,  # Sử dụng hằng số mặc định
)

app = FastAPI(
    title="Black-Scholes Option Lab API",
    version="1.1.0",  # Tăng version
    description="API để tính toán giá quyền chọn, Greeks, và lấy dữ liệu thị trường.",
)


@app.post(
    "/options/price", response_model=OptionPriceResponse, tags=["Options Calculations"]
)
async def api_calculate_option_price(params: OptionPriceRequest):
    try:
        price = 0.0
        if params.option_type == "call":
            price = european_call_price(
                params.S, params.K, params.T, params.r, params.sigma
            )
        else:  # put
            price = european_put_price(
                params.S, params.K, params.T, params.r, params.sigma
            )

        if np.isnan(
            price
        ):  # Trường hợp hàm BS trả về NaN do đầu vào không hợp lệ (hiếm khi với Pydantic)
            raise ValueError("Không thể tính giá với các tham số đầu vào.")

        return OptionPriceResponse(
            input_parameters=params,
            calculated_price=price,
            option_type_calculated=params.option_type,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi server nội bộ: {str(e)}")


@app.post(
    "/options/greeks",
    response_model=OptionGreeksResponse,
    tags=["Options Calculations"],
)
async def api_calculate_option_greeks(params: OptionGreeksRequest):
    try:
        greeks_data = get_all_greeks(
            S=params.S,
            K=params.K,
            T=params.T,
            r=params.r,
            sigma=params.sigma,
            option_type=params.option_type,
            days_in_year=params.days_in_year_for_theta,
        )

        if any(np.isnan(value) for value in greeks_data.values()):
            raise ValueError("Không thể tính toán Greeks với các tham số đầu vào.")

        return OptionGreeksResponse(
            input_parameters=params,
            **greeks_data,
            option_type_calculated=params.option_type,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi server nội bộ: {str(e)}")


@app.get(
    "/marketdata/{ticker_symbol}",
    response_model=MarketDataResponse,
    tags=["Market Data"],
)
async def api_get_market_data(
    ticker_symbol: str = Path(
        ..., min_length=1, max_length=10, regex="^[A-Z0-9.-^]+$", example="AAPL"
    ),
    hv_window: int = Query(
        DEFAULT_DAYS_WINDOW_FOR_HV,
        gt=10,
        le=252,
        description="Số ngày cho cửa sổ HV (11-252)",
    ),
):
    price, change, pct_change = get_current_price_and_change(ticker_symbol)
    hv = calculate_historical_volatility_annualized(
        ticker_symbol, hv_window_days=hv_window
    )

    if price is None and hv is None:
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy dữ liệu giá hoặc HV cho mã: {ticker_symbol}",
        )

    return MarketDataResponse(
        ticker_symbol=ticker_symbol.upper(),
        current_price=price,
        price_change=change,
        price_percent_change=pct_change,
        historical_volatility_calculated=hv,
        hv_window_days_used=(
            hv_window if hv is not None else None
        ),  # Chỉ trả về window nếu HV được tính
    )
