import numpy as np
from core.gbm_simulator import simulate_gbm_paths, get_terminal_prices


def test_simulate_gbm_paths_dimensions_and_initial_value():
    S0_test = 100.0
    mu_test = 0.05
    sigma_test = 0.20
    T_test = 1.0  # 1 năm
    dt_test = 1.0 / 252.0  # Giả sử 252 ngày giao dịch trong năm
    num_paths_test = 10

    num_steps_expected = int(T_test / dt_test)

    generated_paths = simulate_gbm_paths(
        S0_test, mu_test, sigma_test, T_test, dt_test, num_paths_test
    )

    assert generated_paths.shape == (num_paths_test, num_steps_expected + 1)
    assert np.all(generated_paths[:, 0] == S0_test)


def test_get_terminal_prices_shape_and_values():
    S0_test = 50.0
    mu_test = 0.01
    sigma_test = 0.15
    T_test = 0.5  # Nửa năm
    dt_test = 1.0 / 12.0  # Mô phỏng theo tháng
    num_paths_test = 5

    generated_paths = simulate_gbm_paths(
        S0_test, mu_test, sigma_test, T_test, dt_test, num_paths_test
    )
    terminal_prices_calculated = get_terminal_prices(generated_paths)

    assert terminal_prices_calculated.shape == (num_paths_test,)
    assert np.array_equal(terminal_prices_calculated, generated_paths[:, -1])


def test_gbm_path_non_negativity():
    S0_test = 10.0
    mu_test = -0.05  # Drift âm vẫn có thể xảy ra
    sigma_test = 0.30
    T_test = 2.0
    dt_test = 1.0 / 50.0  # 50 bước trong một năm
    num_paths_test = 1000  # Mô phỏng nhiều đường để tăng khả năng bắt lỗi (nếu có)

    generated_paths = simulate_gbm_paths(
        S0_test, mu_test, sigma_test, T_test, dt_test, num_paths_test, random_seed=42
    )
    # GBM theo lý thuyết sẽ không tạo ra giá âm do hàm exp()
    assert np.all(generated_paths >= 0)


def test_gbm_reproducibility_with_seed():
    S0_test = 100.0
    mu_test = 0.05
    sigma_test = 0.20
    T_test = 0.1  # Thời gian ngắn
    dt_test = 1.0 / 252.0
    num_paths_test = 2  # Chỉ cần 2 đường là đủ để so sánh

    paths1 = simulate_gbm_paths(
        S0_test, mu_test, sigma_test, T_test, dt_test, num_paths_test, random_seed=123
    )
    paths2 = simulate_gbm_paths(
        S0_test, mu_test, sigma_test, T_test, dt_test, num_paths_test, random_seed=123
    )
    paths3 = simulate_gbm_paths(
        S0_test, mu_test, sigma_test, T_test, dt_test, num_paths_test, random_seed=456
    )

    assert np.array_equal(paths1, paths2)
    # Rất khó để paths1 và paths3 hoàn toàn giống nhau nếu seed khác nhau.
    # Kiểm tra xem chúng có khác nhau không.
    # Tuy nhiên, do tính ngẫu nhiên, có một xác suất cực nhỏ là chúng vẫn có thể giống nhau.
    # Kiểm tra not np.array_equal
    if (
        num_paths_test > 0 and int(T_test / dt_test) > 0
    ):  # Chỉ kiểm tra nếu có bước mô phỏng
        assert not np.array_equal(paths1, paths3)


def test_gbm_no_time_steps_or_zero_time():
    S0_test = 75.0
    mu_test = 0.03
    sigma_test = 0.10
    dt_test = 1.0 / 252.0
    num_paths_test = 3

    # Trường hợp T < dt (ví dụ, T = 0.001 và dt = 0.004)
    paths_T_less_than_dt = simulate_gbm_paths(
        S0_test, mu_test, sigma_test, T=0.001, dt=dt_test, num_paths=num_paths_test
    )
    assert paths_T_less_than_dt.shape == (num_paths_test, 1)
    assert np.all(paths_T_less_than_dt == S0_test)

    # Trường hợp T = 0
    paths_T_zero = simulate_gbm_paths(
        S0_test, mu_test, sigma_test, T=0.0, dt=dt_test, num_paths=num_paths_test
    )
    assert paths_T_zero.shape == (num_paths_test, 1)
    assert np.all(paths_T_zero == S0_test)
