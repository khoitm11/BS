# Black-Scholes Option Lab 🧪📈✨

Welcome to the **Black-Scholes Option Lab**! This project is an interactive and educational environment designed for students, financial market enthusiasts, quantitative analysts, and anyone curious about the world of options. Dive deep into the Black-Scholes option pricing model, simulate dynamic market movements, analyze option sensitivities (the Greeks), and explore concepts of potential trading edges.

Our lab is meticulously crafted with a robust Python backend, features a high-performance FastAPI for service-oriented architecture, an engaging Streamlit dashboard for real-time analysis and visualization, and integrates MLflow for comprehensive experiment tracking.

## 🌟 Project Highlights & Dazzling Features

*   **🔬 Core Black-Scholes Engine:** Precision calculation of European Call and Put option prices and their vital Greeks (Delta, Gamma, Vega, Theta, Rho).
*   **📉 Geometric Brownian Motion (GBM) Simulator:** Generate and visualize realistic, stochastic stock price paths to understand potential market scenarios.
*   **📊 P&L and Trading Edge Analysis:** Delve into Profit/Loss scenarios for option positions and statistically analyze potential trading edges arising from theoretical mispricings.
*   **Interactive Jupyter Notebooks:** A suite of notebooks for hands-on learning, in-depth exploration, and clear presentation of complex financial concepts:
    *   `01_black_scholes_and_gbm.ipynb`: Your starting point for Black-Scholes pricing fundamentals and GBM path simulations.
    *   `02_option_pnl_and_edge_simulation.ipynb`: Explore P&L distributions and advanced trading edge analysis.
    *   `03_mlflow_tracking_for_edge_sims.ipynb`: Master experiment tracking with MLflow for your simulation studies.
    *   `04_greeks_analysis.ipynb`: A visual deep-dive into understanding and interpreting Option Greeks.
    *   `05_trade.ipynb`: A practical mini-lab recreating and exploring concepts from options trading strategies.
*   **🚀 MLflow Integration:** Systematically log, track, compare, and manage different simulation experiments, including parameters, metrics, and visual artifacts (results are stored locally in the `mlruns/` directory).
*   **🌐 FastAPI Backend Service:** A powerful API providing endpoints for:
    *   Option Pricing (Call/Put)
    *   Greeks Calculation
    *   Market Data Fetching (via `yfinance`)
*   **🖥️ Streamlit Interactive Dashboard:** A sophisticated, user-friendly web interface offering:
    *   Near real-time stock price and historical volatility (HV) fetching using `yfinance`.
    *   Dynamic input of custom option parameters (S, K, T, r, σ).
    *   Instantaneous display of calculated Call/Put prices and all associated Greeks (data fetched via API calls to the FastAPI backend).
    *   Stunning, interactive Plotly charts visualizing how Greeks respond to changes in the underlying asset price.
    *   An auto-refresh mechanism for market data, giving a "live trading desk" feel.
*   **🐳 Dockerized Applications:** Production-ready Dockerfiles for effortless containerization and deployment of both the FastAPI API and the Streamlit Dashboard.
*   **🤖 CI/CD with GitHub Actions:** A fully automated pipeline for linting (Ruff), formatting checks, and running unit tests, ensuring code quality and stability.
*   **✨ Clean Code & Comprehensive Unit Tests:** Emphasis on well-structured, readable, and maintainable Python code, backed by thorough unit tests using `pytest`.

---
![image](https://github.com/user-attachments/assets/1e6921d8-f24f-425d-9258-b2eee4558122)
![image](https://github.com/user-attachments/assets/df498c34-e000-48b3-91c4-8e943ac2bb91)
---

## 🚀 Getting Started: Your Journey Begins Here!

### Prerequisites

*   Python 3.10 or higher
*   Pip (Python package installer)
*   Git version control
*   Docker Desktop (Optional, but recommended for running containerized applications and a full experience)
*   An IDE like PyCharm or VS Code is highly recommended for development.

### 1. Clone the Repository

First, get a local copy of the project:
```bash
git clone https://github.com/khoitm11/Black_Scholes_option_lab.git
cd Black_Scholes_option_lab
```

### 2. Set Up a Python Virtual Environment

Isolate your project dependencies using a virtual environment. This is a best practice.
```bash
# Create a virtual environment ('.venv' is a common name)
python -m venv .venv

# Activate the virtual environment:
# On Windows (PowerShell/CMD):
# .\.venv\Scripts\activate
# On macOS/Linux (Bash/Zsh):
source .venv/bin/activate
```
*(You should see `(.venv)` at the beginning of your terminal prompt.)*

### 3. Install All Dependencies

Install the required Python packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run Unit Tests (Ensure Stability!)

Verify that all core components are functioning as expected:
```bash
pytest
```
*(All tests should pass with green lights!)*

---
![image](https://github.com/user-attachments/assets/61a11bfc-f200-4769-956f-d0fa5025c074)
---

### 5. Dive into Jupyter Notebooks for Exploration

JupyterLab provides an excellent interactive environment to explore the project's logic.
```bash
jupyter lab
```
This will open JupyterLab in your default web browser. Navigate to the `notebooks/` directory to begin your exploration.

---
![image](https://github.com/user-attachments/assets/82e4abdc-9f5a-48a3-854e-1fd76b2d4b0c)

and some simulations...

![image](https://github.com/user-attachments/assets/08499a88-2ab9-4e44-a50c-e90bf2880a88)
---

### 6. Launch the FastAPI API Service (Local)

The API backend serves calculations and data.
- Run from the project root directory:

```
uvicorn app.api.main_api:app --reload --host 0.0.0.0 --port 8000
```
*   The API will be live at `http://localhost:8000`.
*   Interactive API Documentation (Swagger UI): `http://localhost:8000/docs`
*   Alternative API Documentation (ReDoc): `http://localhost:8000/redoc`

---
![image](https://github.com/user-attachments/assets/f1f3bb85-7d1a-4904-8f85-b8cff665248a)
---

### 7. Launch the Streamlit Dashboard (Local)

Experience the interactive UI. **Ensure the FastAPI service (Step 6) is running** as the dashboard will call it for data. (Verify `API_BASE_URL` in `app/dashboard/main_dashboard.py`).
Run from the project root directory:
```bash
streamlit run app/dashboard/main_dashboard.py
```
The dashboard will typically open automatically in your browser at `http://localhost:8501`.

### 8. Track Experiments with MLflow UI (Local)

After executing notebooks that utilize MLflow (e.g., `03_mlflow_tracking_for_edge_sims.ipynb`):
Run from the project root directory (where the 'mlruns/' directory is created):
```bash
mlflow ui
```
Access the MLflow UI in your browser, usually at `http://localhost:5000`, to review your experiment runs.

---
![image](https://github.com/user-attachments/assets/7837eb8e-b6d7-49e7-a1d5-bfccc8301e33)

detail...

![image](https://github.com/user-attachments/assets/bf2f870d-4f64-4c40-a34f-460844255f4f)
---

### 9. Running with Docker (Optional - Full Containerized Experience)

Ensure Docker Desktop is running on your system.

**a. Build the API Docker Image:**
```bash
docker build -t blackscholes-lab-api:latest -f Dockerfile_api .
```

**b. Run the API Docker Container:**
```
docker run --rm -d -p 8001:8000 --name bs_api_container blackscholes-lab-api:latest
```
*(The API will now be accessible at `http://localhost:8001` on your host machine).*

**c. Build the Dashboard Docker Image:**

```
docker build -t blackscholes-lab-dashboard:latest -f Dockerfile_dashboard .
```

**d. Run the Dashboard Docker Container:**
Example for Docker Desktop (Windows/Mac) where API container is mapped to host port 8001:
```
docker run --rm -d -p 8502:8501 --name bs_dashboard_container -e API_BASE_URL="http://host.docker.internal:8001" blackscholes-lab-dashboard:latest
```
*(The Dashboard will be accessible at `http://localhost:8502` on your host machine).*
*Note on `API_BASE_URL` for Docker: `host.docker.internal` allows a container to reach services running on the host (including other containers mapped to host ports). If using custom Docker networks, you could use the API container's service name.*

## 📁 Project Structure Overview

-   `Black_Scholes_option_lab/` *(Root project directory)*
    -   `.github/` *(GitHub specific files)*
        -   `workflows/`
            -   `python_ci.yml` *(GitHub Actions CI workflow)*
    -   `.venv/` *(Python virtual environment - should be in .gitignore)*
    -   `app/` *(Application source code)*
        -   `__init__.py`
        -   `api/` *(FastAPI application module)*
            -   `__init__.py`
            -   `main_api.py` *(Main FastAPI application and endpoints)*
            -   `models_api.py` *(Pydantic models for API requests/responses)*
        -   `dashboard/` *(Streamlit dashboard application module)*
            -   `__init__.py`
            -   `main_dashboard.py` *(Main Streamlit application script)*
    -   `core/` *(Core calculation logic and algorithms)*
        -   `__init__.py`
        -   `black_scholes.py` *(Black-Scholes pricing model and Greeks calculations)*
        -   `gbm_simulator.py` *(Geometric Brownian Motion simulation logic)*
        -   `option_analyzer.py` *(P&L and trading edge analysis functions)*
    -   `data_fetcher/` *(Module for fetching external market data)*
        -   `__init__.py`
        -   `live_data.py` *(Integration with yfinance for stock prices/HV)*
    -   `mlruns/` *(MLflow experiment tracking data - should be in .gitignore)*
    -   `notebooks/` *(Jupyter notebooks for interactive analysis and experimentation)*
        -   `01_black_scholes_and_gbm.ipynb`
        -   `02_option_pnl_and_edge_simulation.ipynb`
        -   `03_mlflow_tracking_for_edge_sims.ipynb`
        -   `04_greeks_analysis.ipynb`
        -   `05_trade.ipynb`
    -   `tests/` *(Unit and integration tests for the project)*
        -   `__init__.py`
        -   `test_black_scholes.py`
        -   `test_gbm_simulator.py`
        -   `test_live_data.py`
        -   `test_option_analyzer.py`
    -   `.dockerignore` *(Specifies files to ignore during Docker image build)*
    -   `.gitignore` *(Specifies intentionally untracked files for Git)*
    -   `Dockerfile_api` *(Instructions to build the Docker image for the FastAPI service)*
    -   `Dockerfile_dashboard` *(Instructions to build the Docker image for the Streamlit dashboard)*
    -   `generate_test_benchmarks.py` *(Utility script, if used for generating test data)*
    -   `LICENSE` *(Recommended: Add an open-source license, e.g., MIT License)*
    -   `pyproject.toml` *(Configuration file for tools like Ruff, pytest, etc.)*
    -   `README.md` *(This file! Your guide to the project)*
    -   `requirements.txt` *(List of Python package dependencies)*

## 🛠️ Technology Stack & Libraries

*   **Primary Language:** Python 3.10+
*   **Core Calculations & Simulations:** NumPy, SciPy
*   **API Development:** FastAPI, Pydantic, Uvicorn (ASGI Server)
*   **Interactive Dashboard & Visualization:** Streamlit, Pandas, Plotly (Express & Graph Objects)
*   **Market Data Fetching:** `yfinance`
*   **Testing Framework:** `pytest`, `unittest.mock`
*   **Code Linting & Formatting:** Ruff (An extremely fast Python linter and formatter, integrating capabilities of Flake8, isort, Black, etc.)
*   **Experiment Tracking & Management:** MLflow
*   **Containerization:** Docker
*   **Continuous Integration & Delivery (CI/CD):** GitHub Actions
 
##   👨‍💻 About Me

Name: Khoi Tran

LinkedIn: www.linkedin.com/in/khoitm11
