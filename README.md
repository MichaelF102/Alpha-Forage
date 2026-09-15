# ⚡ AlphaForge: Institutional Multi-Factor Equity Research & Portfolio Platform

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.63%2B-FF4B4B?logo=streamlit)
![Riskfolio-Lib](https://img.shields.io/badge/Optimization-Riskfolio--Lib%20%26%20SLSQP-00E676)
![Markets](https://img.shields.io/badge/Markets-India%20(NSE%2FBSE)%20%7C%20US%20(NASDAQ%2FNYSE)-38BDF8)
![License](https://img.shields.io/badge/License-MIT-green)

AlphaForge is an institutional-grade quantitative equity research and portfolio construction platform. It provides an end-to-end quantitative workflow for multi-factor security ranking, constrained portfolio optimization, historical strategy backtesting, Fama-French factor return attribution, and comprehensive tail-risk analytics across **Indian** and **United States** equity universes.

---

## 🌟 Key Capabilities

### 1. 🌐 Dynamic Multi-Region Stock Selection
- **Massive Universe Coverage**: Ingests thousands of equities directly from local CSV datasets:
  - **India Universe (`data/India_Stocks_Data.csv`)**: 7,610 securities across NSE and BSE.
  - **US Universe (`data/US_Stocks_Data.csv`)**: 10,937 securities across NASDAQ, NYSE, and OTC.
- **Dynamic Filters**: Real-time filtering by **Exchange** (NSE, BSE, NASDAQ, NYSE, All) and **Market Capitalization** (All, Large Cap, Mid Cap, Small Cap, Micro Cap).
- **Flexible Universe Modes**:
  - **Automatic Mode**: Automatically selects and ranks the top $N$ liquid securities by market cap.
  - **Custom Basket Builder**: Hand-pick any custom equity basket (e.g., *AAPL, MSFT, NVDA, GOOGL, AMZN* or *RELIANCE, TCS, INFY, HDFCBANK*) for targeted portfolio optimization.

---

### 2. 🔬 Real Quantitative Multi-Factor Model
AlphaForge constructs genuine fundamental and market-based factor signals from financial datasets:
- **Momentum Factor**: $12-1\text{ Month Return} = \frac{P_{t-21}}{P_{t-252}} - 1$ (avoids short-term 1-month reversal noise).
- **Low Volatility Factor**: Reverse annualized standard deviation rank ($\sigma_{\mathrm{ann}} = \mathrm{std}(r_{\mathrm{daily}}) \times \sqrt{252}$).
- **Fundamental Value Factor**: Combination of earnings yield and trailing dividend yield:
  $$\text{Value Score} = 0.8 \cdot \frac{1}{\mathrm{P/E}} + 0.2 \cdot \mathrm{Dividend\ Yield}$$
- **Quality / Growth Factor**: Genuine Trailing-Twelve-Month (TTM) Diluted EPS Year-over-Year Growth rate from financial reports.
- **Tactical Factor Tilting**: Interactive sliders with institutional presets:
  - *Balanced (25/25/25/25)*
  - *Aggressive Momentum* (50% Mom, 20% Quality, 15% Vol, 15% Value)
  - *Deep Value & Yield* (50% Value, 25% Quality, 15% Vol, 10% Mom)
  - *Quality Growth* (50% Quality, 25% Mom, 15% Value, 10% Vol)
  - *Low Vol Defense* (50% Vol, 25% Quality, 15% Value, 10% Mom)
  - *Custom Sliders*

---

### 3. ⚖️ Multi-Model Portfolio Optimization
Combines factor-implied expected returns ($\mu_i = r_f + \beta_i \cdot \mathrm{MRP} + (\text{Composite}_i - 0.5) \cdot 8\%$) with historical asset covariance:
- **Maximum Sharpe Ratio**: Solves constrained Mean-Variance Optimization using **Riskfolio-Lib** (with SciPy **SLSQP** fallback).
- **Minimum Volatility**: Allocates to the global minimum variance portfolio for defensive capital preservation.
- **Hierarchical Risk Parity (HRP)**: Machine-learning tree clustering that groups correlated assets and recursively bisects weights, eliminating matrix inversion instabilities.
- **Institutional Constraints**:
  - Fully invested ($\sum w_i = 100\%$).
  - Single security weight limits ($1\% \le w_i \le 15\%$).
  - Adaptive sector diversification limits preventing industry concentration.
- **Efficient Frontier Space**: Interactive 2D scatter visualization plotting 250 simulated portfolios against Min Vol and Max Sharpe points.

---

### 4. 📈 Strategy Backtesting & Monte Carlo Projection
- **Benchmark Comparison**: Evaluates strategy against an Equal Weight portfolio and regional market benchmarks:
  - **India**: Nifty 50 (`^NSEI`) or BSE Sensex (`^BSESN`).
  - **US**: S&P 500 Index (`^GSPC`).
- **Dynamic Analysis**: Interactive date-window slicing, customizable starting capital, cumulative wealth trajectories, underwater drawdowns, and 30-day rolling Sharpe & Volatility.
- **🔮 Monte Carlo Simulation (1,000 Paths)**: Geometric Brownian Motion projection across 1, 2, or 3-year forward horizons displaying:
  - 95th Percentile (Bull Market scenario)
  - 75th Percentile
  - 50th Percentile (Median expected trajectory)
  - 25th Percentile
  - 5th Percentile (Severe Bear Market / Tail Risk Floor)

---

### 5. 🛡️ Institutional Risk Analytics & Stress Testing
- **Tail-Risk Statistics**: Historical VaR (95%), Parametric VaR (95%), and Conditional Value at Risk (CVaR / Expected Shortfall).
- **Advanced Ratios**: Portfolio Beta, Tracking Error, Information Ratio, Treynor Ratio, Omega Ratio, Calmar Ratio, and Downside Deviation.
- **Return Distribution**: Histogram with 95% Historical VaR threshold cutoff line.
- **Macroeconomic Stress Testing**: Quantifies portfolio capitalization loss under severe macro shocks:
  - Global Market Crash (-30%)
  - Central Bank Interest Rate Shock (-10%)
  - Macro Recessionary Shock (-20%)

---

### 6. 🧩 Fama-French Factor Return Attribution
- Decomposes active returns against long-short factor-mimicking portfolios (Top minus Bottom decile returns for Momentum, Volatility, Value, and Quality).
- Calculates daily and annualized active factor contributions.
- Plots cumulative factor alpha generation curves over time.

---

### 7. 🔍 Single Security Intelligence
Deep-dive inspection for any stock selected in the sidebar:
- Price chart with 50-day and 200-day Simple Moving Averages (SMA).
- Institutional **Financial Health Score** (0–100) and multi-axis Radar chart.
- Real-time **Valuation Gauges** for Trailing P/E and Price-to-Book.
- **Wall Street / Dalal Street Analyst Consensus**: 12-month target price forecast gauge (High, Mean, Low vs. current price) and broker recommendation distribution (Buy, Hold, Sell).
- **Institutional Ownership Breakdown**: Institutional, insider, and public float pie chart.
- **Technical Summary**: 14-Day RSI, 20/50/200 SMAs, and momentum signals.

---

## 📁 Repository Structure

```tree
Alpha-Forage/
├── app.py                      # Main Streamlit application landing page
├── main.py                     # CLI entrypoint and backend calculation engine
├── engine.py                   # QuantEngine: multi-factor scoring, optimization, analytics
├── sidebar.py                  # Unified interactive sidebar controls & session runner
├── helper.py                   # Data ingestion, yfinance helpers, visual charts, custom theme
├── requirements.txt            # Python dependencies
├── risk_performance_report.txt # Generated textual institutional report
├── .gitignore                  # Git exclusion rules (virtual environments, pycache, logs)
│
├── data/                       # Stock universes with fundamental metadata
│   ├── India_Stocks_Data.csv   # 7,610 Indian stocks (NSE & BSE)
│   └── US_Stocks_Data.csv      # 10,937 US stocks (NASDAQ, NYSE, OTC)
│
├── pages/                      # Multi-page Streamlit modules
│   ├── 1_Research.py           # Factor ranking table & single stock deep-dive
│   ├── 2_Optimization.py       # Portfolio optimization, allocations, Efficient Frontier
│   ├── 3_Backtesting.py        # Wealth curves, drawdowns, rolling stats, Monte Carlo
│   ├── 4_Risk.py               # Side-by-side benchmark risk, VaR/CVaR, stress tests
│   └── 5_Attribution.py        # Fama-French long-short factor alpha decomposition
│
├── analytics/                  # Performance, attribution, and reporting math
│   ├── attribution.py
│   ├── performance.py
│   └── report.py
│
├── risk/                       # Risk models
│   ├── beta.py
│   ├── correlation.py
│   ├── cvar.py
│   ├── drawdown.py
│   ├── rolling.py
│   ├── stress.py
│   └── var.py
│
├── reports/                    # Exported CSV result matrices
│   ├── factors.csv
│   ├── weights.csv
│   ├── portfolio.csv
│   ├── performance.csv
│   ├── risk.csv
│   ├── expected_returns.csv
│   ├── frontier.csv
│   ├── rolling.csv
│   └── attribution.csv
│
└── tests/                      # Automated unit tests
    └── test_risk_analytics.py
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/MichaelF102/Alpha-Forage.git
cd Alpha-Forage
```

### 2. Set up virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### A. Run the Interactive Web Dashboard (Streamlit)
```bash
streamlit run app.py
```
*Or alternatively:*
```bash
streamlit run main.py
```
Then open `http://localhost:8501` in your browser.

#### In the Web UI:
1. **Toggle Market Region**: Switch between **India 🇮🇳** and **US 🇺🇸** in the sidebar.
2. **Choose Universe Mode**: Select **Automatic (Top Market Cap)** or **Custom Basket** to hand-pick specific equities.
3. **Select Optimization Model**: Choose **Max Sharpe Ratio**, **Minimum Volatility**, or **Hierarchical Risk Parity (HRP)**.
4. **Customize Factor Tilts**: Expand *Tactical Factor Tilts* to select presets (*Aggressive Momentum*, *Deep Value*, etc.) or adjust custom sliders.
5. **Inspect Individual Equities**: Search and select any stock to review financial health radars, analyst consensus, and valuation meters.

---

### B. Run via Command Line (CLI)
You can run the full quantitative pipeline and export institutional reports directly from your terminal:

```bash
# Run for Indian Equities (NSE)
python main.py --region India --exchange NSE --universe-size 30 --top-k 12 --period 2y

# Run for US Equities (NASDAQ / S&P 500 benchmark)
python main.py --region US --exchange NASDAQ --universe-size 30 --top-k 12 --period 2y
```

This will output a terminal summary report and export all CSV matrices to `reports/` and `risk_performance_report.txt`.

---

## 🧪 Running Unit Tests

Run the automated test suite:
```bash
python -m unittest discover -s tests
```

---

## 📜 License
This project is licensed under the MIT License.
