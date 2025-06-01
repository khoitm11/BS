# app/api/models_api.py
from pydantic import BaseModel, Field
from typing import Literal, Optional  # Optional dùng cho các trường có thể là None


class OptionParamsBase(BaseModel):
    S: float = Field(
        ..., ge=0, example=100.0, description="Giá tài sản cơ sở hiện tại (S)"
    )
    K: float = Field(
        ..., ge=0, example=100.0, description="Giá thực hiện của quyền chọn (K)"
    )
    T: float = Field(
        ..., gt=0, example=1.0, description="Thời gian đến đáo hạn (năm, T > 0)"
    )
    r: float = Field(
        ..., example=0.05, description="Lãi suất phi rủi ro (vd: 0.05 cho 5%, r)"
    )
    sigma: float = Field(
        ..., gt=0, example=0.20, description="Độ biến động (vd: 0.2 cho 20%, σ > 0)"
    )


class OptionPriceRequest(OptionParamsBase):
    option_type: Literal["call", "put"] = Field(..., example="call")


class OptionPriceResponse(BaseModel):
    input_parameters: OptionPriceRequest
    calculated_price: float
    option_type_calculated: str


class OptionGreeksRequest(
    OptionParamsBase
):  # Tách riêng để không bắt buộc option_type nếu muốn tính chung
    option_type: Literal["call", "put"] = Field(..., example="call")
    days_in_year_for_theta: int = Field(365, ge=252, le=366, example=365)


class OptionGreeksResponse(BaseModel):
    input_parameters: OptionGreeksRequest
    delta: float
    gamma: float
    vega: float
    theta: float  # Sẽ là daily theta
    rho: float
    option_type_calculated: str


class MarketDataResponse(BaseModel):
    ticker_symbol: str
    current_price: Optional[float] = None  # Có thể là None nếu không lấy được
    price_change: Optional[float] = None
    price_percent_change: Optional[float] = None
    historical_volatility_calculated: Optional[float] = None  # Sửa tên cho rõ hơn
    hv_window_days_used: Optional[int] = None
