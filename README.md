# FX Fair Value Model

A quantitative Fair Value model for foreign exchange trading based on macroeconomic fundamentals, featuring an interactive dashboard for analysis and visualization.

## Overview

This project implements a Fair Value model for currency pairs using institutional-grade approaches. It leverages macroeconomic data to estimate the "fair value" of currencies and identify potential trading opportunities through mean-reversion strategies.

The model uses rolling OLS regressions to capture time-varying relationships between exchange rates and fundamental factors, with optional LASSO-based feature selection for automatic variable screening.

## Key Features

- **Multiple Currency Support**: EUR, CHF, CAD, CZK (extensible to other currencies)
- **Rolling Window Analysis**: Configurable windows (6M, 9M, 12M, 18M, 24M)
- **Feature Selection**: Manual selection or automatic LASSO-based screening
- **Interactive Dashboard**: Real-time visualization with Plotly and Dash
- **Statistical Significance Testing**: Automated t-tests and significance ratios
- **Backtesting Framework**: Cumulative Fair Value error tracking

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/pablovicente28130/FX-Fair-Value-Model-.git
cd FX-Fair-Value-Model-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

The project requires the following Python packages:
- `numpy` (>= 1.20.0) - Numerical computations
- `pandas` (>= 1.3.0) - Data manipulation
- `scipy` (>= 1.7.0) - Statistical functions
- `scikit-learn` (>= 1.0.0) - Machine learning tools (OLS, LASSO)
- `dash` (>= 2.0.0) - Web dashboard framework
- `dash-bootstrap-components` (>= 1.0.0) - Dashboard styling
- `plotly` (>= 5.0.0) - Interactive visualizations

## Usage

### Running the Dashboard

The quickest way to explore the model is through the interactive dashboard:

```bash
python dashboard.py data.xlsx
```

The dashboard will launch on `http://localhost:8050`. Open this URL in your web browser to access the interface.

**Dashboard Features:**
- Configure rolling window size (6M to 24M)
- Select feature selection method (Manual or LASSO)
- View OLS regression summaries with statistical significance
- Explore rolling betas over time (by currency or by factor)
- Analyze Fair Value errors and trading signals
- Compare currencies side-by-side

### Using the Model Programmatically

For custom analysis or integration into trading systems:

```python
from model import FXFairValueAnalyzer, ModelConfig

# Configure the model
config = ModelConfig(
    currencies=['EUR', 'CHF', 'CAD', 'CZK'],
    years_back=10,
    selected_features=[
        'RYLDIRS02Y_NSA',        # 2Y Real Rates
        'CPIH_SA_P1M1ML12',      # Inflation YoY
        'INTRGDP_NSA_P1M1ML12_3MMA'  # GDP Growth
    ]
)

# Initialize analyzer
analyzer = FXFairValueAnalyzer('data.xlsx', config)
analyzer.load_data()

# Run analysis with 12-month rolling window
analyzer.analyze_all(window_months=12, selection_method='manual')

# Retrieve results
ols_summary = analyzer.get_ols_summary()
significance_ratios = analyzer.get_significance_ratios()
fair_value_errors = analyzer.get_fair_value_errors()

print(ols_summary)
print(significance_ratios)
```

### Command-Line Arguments

The dashboard script accepts the data file path as an argument:

```bash
python dashboard.py [DATA_FILE_PATH]
```

If no path is provided, it defaults to `data.xlsx` in the current directory.

## Project Structure

```
FX-Fair-Value-Model-/
├── model.py              # Core model implementation
│   ├── DataLoader        # Data loading and preprocessing
│   ├── ModelConfig       # Configuration dataclass
│   ├── FXFairValueAnalyzer  # Main analysis engine
│   └── Backtester        # Backtesting utilities
├── dashboard.py          # Interactive Dash application
├── data.xlsx             # Macroeconomic data (JPMaQS format)
├── requirements.txt      # Python dependencies
├── .gitignore           # Git exclusions
└── README.md            # This file
```

## Methodology

### Macroeconomic Factors

The model uses the following fundamental drivers:

| Variable Code | Description | Category |
|--------------|-------------|----------|
| `RYLDIRS02Y_NSA` | 2-Year Real Interest Rates | Interest Rates |
| `RYLDIRS05Y_NSA` | 5-Year Real Interest Rates | Interest Rates |
| `CPIH_SA_P1M1ML12` | Headline Inflation (YoY) | Inflation |
| `CPIC_SA_P1M1ML12` | Core Inflation (YoY) | Inflation |
| `INTRGDP_NSA_P1M1ML12_3MMA` | GDP Growth (YoY, 3M MA) | Growth |
| `DU02YXR_NSA` | 2-Year Duration | Risk |
| `DU05YXR_NSA` | 5-Year Duration | Risk |
| `PCREDITBN_SJA_P1M1ML12` | Private Credit Growth | Credit |

### Modeling Approach

1. **Data Preprocessing**:
   - Weekly resampling of macroeconomic indicators
   - Forward-fill for missing values
   - Standardization of features for LASSO selection

2. **Rolling OLS Regression**:
   - Estimates time-varying sensitivities (betas) using rolling windows
   - Captures evolving relationships between FX rates and fundamentals
   - Accounts for structural breaks and regime changes

3. **Feature Selection**:
   - **Manual**: Use predefined set of features based on economic theory
   - **LASSO**: Automatic feature selection with cross-validation (TimeSeriesSplit)

4. **Fair Value Calculation**:
   - Predicted FX rate based on current fundamental values
   - Uses most recent beta estimates from rolling regression

5. **Trading Signal**:
   - **Positive error** (Spot > Fair Value): Currency is overvalued → Sell signal
   - **Negative error** (Spot < Fair Value): Currency is undervalued → Buy signal

### Statistical Significance

The model computes t-statistics for each beta coefficient and tracks:
- Significance at 5% level (*): |t| > 1.96
- Significance at 1% level (**): |t| > 2.58
- Significance ratio: Percentage of time each factor is significant

## Data Format

The model expects data in JPMaQS format (long format with specific columns):

| Column | Type | Description |
|--------|------|-------------|
| `real_date` | datetime | Observation date |
| `cid` | string | Currency identifier (e.g., 'EUR', 'USD') |
| `xcat` | string | Variable category code |
| `value` | float | Observed value |

**Example:**
```
real_date    | cid | xcat                      | value
-------------|-----|---------------------------|------
2020-01-01   | EUR | RYLDIRS02Y_NSA            | 0.52
2020-01-01   | EUR | CPIH_SA_P1M1ML12          | 1.31
2020-01-01   | EUR | INTRGDP_NSA_P1M1ML12_3MMA | 1.15
```

### Adding New Currencies

Ensure your data includes:
- FX rate series: `FXXR_NSA` (e.g., `EURNOK_FXXR_NSA` for EUR vs NOK)
- All required fundamental indicators for the new currency

Update the configuration:
```python
config = ModelConfig(
    currencies=['EUR', 'CHF', 'CAD', 'CZK', 'GBP'],  # Add GBP
)
```

### Adding New Factors

1. Add the variable code to `all_feature_xcats` in `ModelConfig`
2. Add a readable name to `VARIABLE_NAMES` dictionary in `model.py`
3. Include the feature in `selected_features` (for manual mode)

Example:
```python
config = ModelConfig(
    all_feature_xcats=[
        'RYLDIRS02Y_NSA',
        'CPIH_SA_P1M1ML12',
        'INTRGDP_NSA_P1M1ML12_3MMA',
        'EQXR_NSA',  # Add equity returns as new factor
    ],
    selected_features=[
        'RYLDIRS02Y_NSA',
        'CPIH_SA_P1M1ML12',
        'INTRGDP_NSA_P1M1ML12_3MMA',
        'EQXR_NSA',
    ]
)
```

## Model Limitations

1. **Low R-squared**: Typical for short-term FX models due to high market noise and non-fundamental factors
2. **Linear Assumptions**: Does not capture non-linear relationships or interaction effects
3. **No Transaction Costs**: Signals do not account for bid-ask spreads, carry costs, or slippage
4. **Mean-Reversion Assumption**: Assumes currencies eventually revert to fair value, which may not hold during persistent trends
5. **Look-Ahead Bias**: Ensure data timestamps reflect information availability in live trading
6. **Regime Dependence**: Relationships may break down during crisis periods or structural shifts

## Technical Implementation Notes

- **Data Loading**: Custom XML parser for strict Excel format compatibility (uses `zipfile` instead of `openpyxl`)
- **Missing Data**: Forward-fill strategy assumes persistence of macro indicators between releases
- **Statistical Tests**: Two-tailed t-tests for coefficient significance
- **Dashboard**: Asynchronous callbacks prevent blocking during recalculation

## References

- Inspired by [Nicolas Hurbin's FX Fair Value Model](https://github.com/NicolasHurbin/FX-Faire-Value-Model)
- Data source: JPMaQS Quantamental Indicators
- Methodology: Standard institutional FX Fair Value approaches

## Author

**Pablo Vicente**

## License

MIT License

---

**Disclaimer**: This model is for educational and research purposes only. It should not be used as the sole basis for trading decisions. Past performance does not guarantee future results. Always conduct thorough due diligence and risk assessment before trading.
