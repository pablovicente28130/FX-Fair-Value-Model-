# FX Fair Value Model

A Fair Value model for FX trading based on macroeconomic fundamentals, using LASSO and OLS regressions. 

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

## Getting Started

### Prerequisites

Before running the model, ensure you have:
- **Python 3.8 or higher** installed on your computer
- **pip** (Python package installer, usually comes with Python)

### Installation & Running the Model

Follow these steps to get the model running on your computer:

**Step 1: Download the project**
```bash
git clone https://github.com/pablovicente28130/FX-Fair-Value-Model-.git
cd FX-Fair-Value-Model-
```

**Step 2: Install required packages**
```bash
pip install -r requirements.txt
```

This will install: `numpy`, `pandas`, `scipy`, `scikit-learn`, `dash`, `plotly`, and other dependencies.

**Step 3: Launch the dashboard**
```bash
python dashboard.py
```

**Step 4: Open your browser**

Go to `http://localhost:8050` in your web browser to access the interactive dashboard.

That's it! The model is now running on your computer.

### What You Can Do in the Dashboard

Once the dashboard is open, you can:
- **Adjust rolling window size**: Choose between 6, 9, 12, 18, or 24 months
- **Select feature selection method**: Manual (predefined factors) or LASSO (automatic)
- **View regression results**: See which economic factors are statistically significant
- **Explore beta evolution**: Track how factor sensitivities change over time
- **Analyze Fair Value signals**: Identify overvalued/undervalued currencies
- **Compare currencies**: Side-by-side analysis of EUR, CHF, CAD, and CZK

### Troubleshooting

If the command `python dashboard.py` doesn't work, try:
```bash
python3 dashboard.py
```

If you don't have the `data.xlsx` file, you can specify a different path:
```bash
python dashboard.py /path/to/your/data.xlsx
```

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

- Data source: JPMaQS Quantamental Indicators
- Methodology: Standard institutional FX Fair Value approaches

## Author

**Pablo Vicente**

