import numpy as np


def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    dt: float,
    num_paths: int,
    random_seed: int = None,
) -> np.ndarray:
    if random_seed is not None:
        np.random.seed(random_seed)

    num_steps = int(T / dt)
    if num_steps <= 0:
        return np.full((num_paths, 1), S0)

    price_paths = np.zeros((num_paths, num_steps + 1))
    price_paths[:, 0] = S0

    # Wiener increments: dW_t = Z * sqrt(dt), Z ~ N(0,1)
    # random_shocks sẽ chứa các giá trị Z cho mỗi path và mỗi bước thời gian.
    #     # Kích thước: (num_paths, num_steps) vì shock xảy ra ở mỗi bước.
    random_shocks = np.random.standard_normal(size=(num_paths, num_steps))

    for t_step in range(1, num_steps + 1):
        # S_previous là giá tại bước thời gian t_step - 1 (cột trước đó)
        S_previous = price_paths[:, t_step - 1]

        # Phần drift của công thức GBM: (mu - 0.5 * sigma^2) * dt
        drift_component = (mu - 0.5 * sigma**2) * dt

        # Phần ngẫu nhiên (diffusion) của công thức GBM: sigma * Z_t * sqrt(dt)
        # Z_t là random_shocks[:, t_step - 1]
        diffusion_component = sigma * np.sqrt(dt) * random_shocks[:, t_step - 1]

        # Áp dụng công thức GBM: S_t = S_{t-dt} * exp(drift + diffusion)
        price_paths[:, t_step] = S_previous * np.exp(
            drift_component + diffusion_component
        )

    return price_paths


def get_terminal_prices(price_paths: np.ndarray) -> np.ndarray:
    # Giá cuối cùng của mỗi đường giá nằm ở cột cuối cùng của mảng price_paths
    # price_paths[:, -1] sẽ trích xuất tất cả các hàng của cột cuối cùng.
    # Kết quả là một mảng 1 chiều chứa các giá cuối cùng.
    if price_paths.shape[1] > 0:  # Đảm bảo mảng không rỗng ở chiều thứ hai
        terminal_prices = price_paths[:, -1]
    else:  # Trường hợp đặc biệt, có thể là mảng rỗng hoặc không có bước nào
        terminal_prices = np.array([])  # Trả về mảng rỗng để tránh lỗi
    return terminal_prices
