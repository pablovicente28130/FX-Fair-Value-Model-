"""
FX Fair Value Model - Core Model (Version 3)
=============================================

Version 3:
- Sélection: Manuel ou LASSO uniquement
- Noms de variables clairs et lisibles
- Modèle en niveaux (pas de dérive)

Auteur: [Ton nom]
Date: Janvier 2025
"""

import zipfile
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


# =============================================================================
# NOMS DES VARIABLES (LISIBLES)
# =============================================================================

VARIABLE_NAMES = {
    'RYLDIRS02Y_NSA': '2Y Real Rates',
    'RYLDIRS05Y_NSA': '5Y Real Rates',
    'CPIH_SA_P1M1ML12': 'Inflation YoY',
    'CPIC_SA_P1M1ML12': 'Core Inflation',
    'CPIH_SJA_P3M3ML3AR': 'Inflation 3M',
    'INTRGDP_NSA_P1M1ML12_3MMA': 'GDP Growth',
    'INTRGDPv5Y_NSA_P1M1ML12_3MMA': 'GDP vs Trend',
    'PCREDITBN_SJA_P1M1ML12': 'Private Credit',
    'DU02YXR_NSA': '2Y Duration',
    'DU05YXR_NSA': '5Y Duration',
    'EQXR_NSA': 'Equities',
    'const': 'Constant',
}

VARIABLE_CODES = {v: k for k, v in VARIABLE_NAMES.items()}


def get_display_name(code: str) -> str:
    return VARIABLE_NAMES.get(code, code)


def get_code_name(display: str) -> str:
    return VARIABLE_CODES.get(display, display)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    currencies: List[str] = field(default_factory=lambda: ['EUR', 'CHF', 'CAD', 'CZK'])
    years_back: int = 10
    
    all_feature_xcats: List[str] = field(default_factory=lambda: [
        'RYLDIRS02Y_NSA',
        'RYLDIRS05Y_NSA',
        'CPIH_SA_P1M1ML12',
        'CPIC_SA_P1M1ML12',
        'CPIH_SJA_P3M3ML3AR',
        'INTRGDP_NSA_P1M1ML12_3MMA',
        'INTRGDPv5Y_NSA_P1M1ML12_3MMA',
        'PCREDITBN_SJA_P1M1ML12',
        'DU02YXR_NSA',
        'DU05YXR_NSA',
    ])
    
    selected_features: List[str] = field(default_factory=lambda: [
        'RYLDIRS02Y_NSA',
        'CPIH_SA_P1M1ML12',
        'INTRGDP_NSA_P1M1ML12_3MMA',
    ])
    
    windows_dict: Dict[int, str] = field(default_factory=lambda: {
        1: "6M", 2: "9M", 3: "12M", 4: "18M", 5: "24M"
    })
    
    resample_freq: str = 'W-MON'


# =============================================================================
# DATA LOADING
# =============================================================================

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_data: Optional[pd.DataFrame] = None
        self.wide_data: Optional[pd.DataFrame] = None
    
    def _parse_xlsx_strict(self) -> pd.DataFrame:
        with zipfile.ZipFile(self.filepath, 'r') as z:
            shared_strings = []
            with z.open('xl/sharedStrings.xml') as f:
                content = f.read().decode('utf-8')
                strings = re.findall(r'<t[^>]*>([^<]*)</t>', content)
                shared_strings = strings
            
            with z.open('xl/worksheets/sheet1.xml') as f:
                content = f.read().decode('utf-8')
            
            rows = re.findall(r'<row[^>]*>(.*?)</row>', content, re.DOTALL)
            
            def parse_cell(cell_xml):
                cell_type = re.search(r't="([^"]*)"', cell_xml)
                cell_type = cell_type.group(1) if cell_type else 'n'
                value_match = re.search(r'<v>([^<]*)</v>', cell_xml)
                if not value_match:
                    return None
                value = value_match.group(1)
                if cell_type == 's':
                    idx = int(value)
                    return shared_strings[idx] if idx < len(shared_strings) else value
                return value
            
            def parse_row(row_xml):
                cells = re.findall(r'<c[^>]*>.*?</c>', row_xml, re.DOTALL)
                return [parse_cell(cell) for cell in cells]
            
            data_rows = []
            for row in rows[1:]:
                values = parse_row(row)
                if len(values) >= 4:
                    try:
                        data_rows.append({
                            'real_date': values[0],
                            'cid': values[1],
                            'xcat': values[2],
                            'value': float(values[3]) if values[3] else np.nan
                        })
                    except (ValueError, TypeError):
                        pass
        
        df = pd.DataFrame(data_rows)
        df['real_date'] = pd.to_datetime(df['real_date'])
        return df
    
    def load_data(self) -> pd.DataFrame:
        if self.raw_data is not None:
            return self.raw_data
        self.raw_data = self._parse_xlsx_strict()
        return self.raw_data
    
    def get_available_currencies(self) -> List[str]:
        if self.raw_data is None:
            self.load_data()
        fx_data = self.raw_data[self.raw_data['xcat'] == 'FXXR_NSA']
        return sorted(fx_data['cid'].unique().tolist())
    
    def to_wide_format(self, currencies: Optional[List[str]] = None,
                       start_date: Optional[str] = None) -> pd.DataFrame:
        if self.raw_data is None:
            self.load_data()
        
        df = self.raw_data.copy()
        if currencies:
            df = df[df['cid'].isin(currencies)]
        if start_date:
            df = df[df['real_date'] >= start_date]
        
        df['varname'] = df['cid'] + '_' + df['xcat']
        df_wide = df.pivot(index='real_date', columns='varname', values='value')
        df_wide = df_wide.sort_index()
        
        self.wide_data = df_wide
        return df_wide


# =============================================================================
# LASSO SELECTION
# =============================================================================

class LassoSelector:
    @staticmethod
    def select(y: pd.Series, X: pd.DataFrame, cv_folds: int = 5) -> Tuple[List[str], Dict]:
        data = pd.concat([y, X], axis=1).dropna()
        if len(data) < 50:
            return X.columns.tolist(), {'error': 'insufficient_data'}
        
        y_clean = data.iloc[:, 0].values
        X_clean = data.iloc[:, 1:].values
        feature_names = data.columns[1:].tolist()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        lasso = LassoCV(cv=tscv, max_iter=10000, random_state=42)
        lasso.fit(X_scaled, y_clean)
        
        selected_idx = np.where(np.abs(lasso.coef_) > 1e-6)[0]
        selected_vars = [feature_names[i] for i in selected_idx]
        
        coef_importance = sorted(
            zip(feature_names, lasso.coef_),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        info = {
            'alpha': lasso.alpha_,
            'n_selected': len(selected_vars),
            'coefficients': dict(coef_importance),
            'r2_score': lasso.score(X_scaled, y_clean),
            'ranking': [get_display_name(c[0]) for c in coef_importance if abs(c[1]) > 1e-6]
        }
        
        return selected_vars if selected_vars else feature_names[:3], info


# =============================================================================
# REGRESSION ENGINE
# =============================================================================

class RegressionEngine:
    @staticmethod
    def run_ols(y: pd.Series, X: pd.DataFrame, add_constant: bool = True) -> Optional[Dict]:
        data = pd.concat([y, X], axis=1).dropna()
        if len(data) < 10:
            return None
        
        y_clean = data.iloc[:, 0]
        X_clean = data.iloc[:, 1:]
        
        if add_constant:
            X_clean = X_clean.copy()
            X_clean.insert(0, 'const', 1.0)
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X_clean, y_clean)
        
        y_pred = model.predict(X_clean)
        residuals = y_clean - y_pred
        
        n = len(y_clean)
        k = X_clean.shape[1]
        
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2
        
        try:
            mse = ss_res / (n - k) if n > k else ss_res
            X_matrix = X_clean.values
            var_coef = mse * np.linalg.inv(X_matrix.T @ X_matrix).diagonal()
            se = np.sqrt(np.abs(var_coef))
            t_stats = model.coef_ / (se + 1e-10)
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), max(n - k, 1)))
        except:
            se = np.full(k, np.nan)
            p_values = np.full(k, np.nan)
        
        return {
            'betas': pd.Series(model.coef_, index=X_clean.columns),
            'pvalues': pd.Series(p_values, index=X_clean.columns),
            'std_errors': pd.Series(se, index=X_clean.columns),
            'residuals': pd.Series(residuals.values, index=y_clean.index),
            'fitted': pd.Series(y_pred, index=y_clean.index),
            'r2': r2,
            'r2_adj': r2_adj,
            'n_obs': n
        }
    
    @staticmethod
    def run_rolling_ols(y: pd.Series, X: pd.DataFrame,
                        window: int, add_constant: bool = True) -> Dict:
        data = pd.concat([y, X], axis=1).dropna()
        y_clean = data.iloc[:, 0]
        X_clean = data.iloc[:, 1:]
        
        if add_constant:
            X_clean = X_clean.copy()
            X_clean.insert(0, 'const', 1.0)
        
        n_obs = len(y_clean)
        n_features = X_clean.shape[1]
        
        betas_arr = np.full((n_obs, n_features), np.nan)
        pvals_arr = np.full((n_obs, n_features), np.nan)
        r2_arr = np.full(n_obs, np.nan)
        
        for i in range(window - 1, n_obs):
            start_idx = i - window + 1
            y_window = y_clean.iloc[start_idx:i+1]
            X_window = X_clean.iloc[start_idx:i+1]
            
            X_no_const = X_window.drop(columns=['const']) if add_constant else X_window
            result = RegressionEngine.run_ols(y_window, X_no_const, add_constant=add_constant)
            
            if result is not None:
                betas_arr[i] = result['betas'].values
                pvals_arr[i] = result['pvalues'].values
                r2_arr[i] = result['r2_adj']
        
        return {
            'betas': pd.DataFrame(betas_arr, index=y_clean.index, columns=X_clean.columns).dropna(how='all'),
            'pvalues': pd.DataFrame(pvals_arr, index=y_clean.index, columns=X_clean.columns).dropna(how='all'),
            'r2': pd.Series(r2_arr, index=y_clean.index).dropna()
        }
    
    @staticmethod
    def compute_significance_ratio(pvalues: pd.DataFrame, threshold: float = 0.05) -> pd.Series:
        mask = pvalues < threshold
        return (mask.sum() / len(mask) * 100).round(2)


# =============================================================================
# FAIR VALUE MODEL
# =============================================================================

class FairValueModel:
    def __init__(self, df_wide: pd.DataFrame, currency: str, config: ModelConfig):
        self.df = df_wide.copy()
        self.currency = currency
        self.config = config
        self.fx_column = f"{currency}_FXXR_NSA"
        
        self.features: Optional[pd.DataFrame] = None
        self.spot_series: Optional[pd.Series] = None
        self.results: Dict = {}
        self.lasso_info: Dict = {}
    
    def prepare_features(self, feature_xcats: List[str]) -> pd.DataFrame:
        if self.fx_column not in self.df.columns:
            raise ValueError(f"FX column '{self.fx_column}' not found")
        
        available_cols = []
        for xcat in feature_xcats:
            col = f"{self.currency}_{xcat}"
            if col in self.df.columns:
                available_cols.append(col)
        
        if not available_cols:
            raise ValueError(f"No features for {self.currency}")
        
        features = self.df[[self.fx_column] + available_cols].copy()
        self.spot_series = features[self.fx_column].cumsum()
        
        features.columns = ['fx_return'] + [c.replace(f"{self.currency}_", '') for c in available_cols]
        
        features = features.resample(self.config.resample_freq).last()
        self.spot_series = self.spot_series.resample(self.config.resample_freq).last()
        
        features['y'] = self.spot_series
        features = features.drop(columns=['fx_return'])
        features = features.dropna()
        
        self.features = features
        return features
    
    def select_variables(self, method: str = 'manual',
                         manual_vars: Optional[List[str]] = None) -> List[str]:
        if self.features is None:
            self.prepare_features(self.config.all_feature_xcats)
        
        y = self.features['y']
        X = self.features.drop(columns=['y'])
        
        if method == 'manual':
            if manual_vars:
                available = [v for v in manual_vars if v in X.columns]
                selected = available if available else X.columns[:3].tolist()
            else:
                selected = [v for v in self.config.selected_features if v in X.columns]
            self.lasso_info = {'method': 'manual'}
            
        elif method == 'lasso':
            selected, self.lasso_info = LassoSelector.select(y, X)
            self.lasso_info['method'] = 'lasso'
        else:
            selected = X.columns[:3].tolist()
        
        return selected
    
    def run_analysis(self, window_months: int = 12,
                     selection_method: str = 'manual',
                     manual_vars: Optional[List[str]] = None) -> Dict:
        
        self.prepare_features(self.config.all_feature_xcats)
        selected_vars = self.select_variables(selection_method, manual_vars)
        
        y = self.features['y']
        X = self.features[selected_vars]
        
        window_size = window_months * 4 if 'W' in self.config.resample_freq else window_months
        
        ols_result = RegressionEngine.run_ols(y, X, add_constant=True)
        rolling_result = RegressionEngine.run_rolling_ols(y, X, window=window_size, add_constant=True)
        
        sig_ratios = RegressionEngine.compute_significance_ratio(rolling_result['pvalues'])
        
        # Importance des variables
        var_importance = {}
        if ols_result:
            for var in selected_vars:
                if var in ols_result['betas'].index:
                    beta = abs(ols_result['betas'][var])
                    pval = ols_result['pvalues'][var]
                    score = beta * (1.0 if pval < 0.05 else 0.5)
                    var_importance[var] = {
                        'beta': ols_result['betas'][var],
                        'pvalue': pval,
                        'significant': pval < 0.05,
                        'score': score,
                        'display_name': get_display_name(var)
                    }
        
        var_importance = dict(sorted(var_importance.items(), key=lambda x: x[1]['score'], reverse=True))
        
        self.results = {
            'ols': ols_result,
            'rolling': rolling_result,
            'significance_ratios': sig_ratios,
            'selected_variables': selected_vars,
            'variable_importance': var_importance,
            'lasso_info': self.lasso_info,
            'window_size': window_size
        }
        
        return self.results
    
    def compute_fair_value(self) -> pd.DataFrame:
        if 'rolling' not in self.results:
            raise ValueError("Run analysis first")
        
        rolling_betas = self.results['rolling']['betas']
        selected_vars = self.results['selected_variables']
        
        common_idx = self.features.index.intersection(rolling_betas.index)
        features_aligned = self.features.loc[common_idx]
        betas_aligned = rolling_betas.loc[common_idx]
        
        spot = features_aligned['y']
        
        X = features_aligned[selected_vars].copy()
        X.insert(0, 'const', 1.0)
        
        betas_lagged = betas_aligned.shift(1)
        fair_value = (X * betas_lagged).sum(axis=1).dropna()
        
        common_idx = spot.index.intersection(fair_value.index)
        spot = spot.loc[common_idx]
        fair_value = fair_value.loc[common_idx]
        
        misalignment = (spot - fair_value) / fair_value.abs().replace(0, np.nan) * 100
        misalignment = misalignment.fillna(0).clip(-50, 50)
        
        signal = pd.Series(index=common_idx, dtype=str)
        signal[misalignment > 0] = 'OVERVALUED'
        signal[misalignment <= 0] = 'UNDERVALUED'
        
        rolling_mean = misalignment.rolling(window=52, min_periods=10).mean()
        rolling_std = misalignment.rolling(window=52, min_periods=10).std()
        z_score = (misalignment - rolling_mean) / rolling_std.replace(0, np.nan)
        z_score = z_score.fillna(0).clip(-3, 3)
        
        result_df = pd.DataFrame({
            'spot': spot,
            'fair_value': fair_value,
            'misalignment_pct': misalignment,
            'z_score': z_score,
            'signal': signal
        })
        
        self.fair_value_df = result_df
        return result_df
    
    def get_current_signal(self) -> Dict:
        if not hasattr(self, 'fair_value_df'):
            self.compute_fair_value()
        
        fv = self.fair_value_df
        last = fv.iloc[-1]
        
        return {
            'currency': self.currency,
            'date': fv.index[-1],
            'spot': last['spot'],
            'fair_value': last['fair_value'],
            'misalignment_pct': last['misalignment_pct'],
            'z_score': last['z_score'],
            'signal': last['signal']
        }


# =============================================================================
# MAIN ANALYZER
# =============================================================================

class FXFairValueAnalyzer:
    def __init__(self, filepath: str, config: Optional[ModelConfig] = None):
        self.filepath = filepath
        self.config = config or ModelConfig()
        self.loader = DataLoader(filepath)
        self.df_wide: Optional[pd.DataFrame] = None
        self.models: Dict[str, FairValueModel] = {}
        self.results: Dict[str, Dict] = {}
    
    def load_data(self) -> pd.DataFrame:
        self.loader.load_data()
        
        end_date = self.loader.raw_data['real_date'].max()
        start_date = end_date - pd.DateOffset(years=self.config.years_back)
        
        self.df_wide = self.loader.to_wide_format(
            currencies=self.config.currencies,
            start_date=start_date.strftime('%Y-%m-%d')
        )
        
        return self.df_wide
    
    def get_available_features(self, currency: str) -> List[str]:
        if self.df_wide is None:
            self.load_data()
        
        available = []
        for xcat in self.config.all_feature_xcats:
            col = f"{currency}_{xcat}"
            if col in self.df_wide.columns:
                available.append(xcat)
        return available
    
    def analyze_currency(self, currency: str,
                         window_months: int = 12,
                         selection_method: str = 'manual',
                         manual_vars: Optional[List[str]] = None) -> Optional[Dict]:
        
        if self.df_wide is None:
            self.load_data()
        
        fx_col = f"{currency}_FXXR_NSA"
        if fx_col not in self.df_wide.columns:
            return None
        
        model = FairValueModel(self.df_wide, currency, self.config)
        
        try:
            results = model.run_analysis(window_months, selection_method, manual_vars)
            results['fair_value'] = model.compute_fair_value()
            results['current_signal'] = model.get_current_signal()
            results['spot_series'] = model.spot_series
            results['features'] = model.features
            
            self.models[currency] = model
            self.results[currency] = results
            return results
        except Exception as e:
            print(f"Error analyzing {currency}: {e}")
            return None
    
    def analyze_all(self, window_months: int = 12,
                    selection_method: str = 'manual',
                    manual_vars: Optional[List[str]] = None) -> Dict:
        
        if self.df_wide is None:
            self.load_data()
        
        for ccy in self.config.currencies:
            self.analyze_currency(ccy, window_months, selection_method, manual_vars)
        
        return self.results
    
    def get_ols_summary(self) -> pd.DataFrame:
        rows = []
        for ccy, result in self.results.items():
            if result is None or result.get('ols') is None:
                continue
            
            ols = result['ols']
            row = {'Currency': ccy}
            
            for var in ols['betas'].index:
                if var == 'const':
                    continue
                display_name = get_display_name(var)
                beta = ols['betas'][var]
                pval = ols['pvalues'][var]
                
                formatted = f"{beta:.4f}"
                if pval < 0.01:
                    formatted += "**"
                elif pval < 0.05:
                    formatted += "*"
                
                row[display_name] = formatted
            
            row['R²'] = f"{ols['r2_adj']*100:.1f}%"
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_significance_ratios(self) -> pd.DataFrame:
        data = {}
        for ccy, result in self.results.items():
            if result is not None:
                ratios = result['significance_ratios']
                renamed = {get_display_name(k): v for k, v in ratios.items() if k != 'const'}
                data[ccy] = renamed
        return pd.DataFrame(data)
    
    def get_rolling_betas(self, currency: str) -> pd.DataFrame:
        if currency not in self.results:
            return pd.DataFrame()
        betas = self.results[currency]['rolling']['betas'].copy()
        betas.columns = [get_display_name(c) for c in betas.columns]
        return betas
    
    def get_rolling_pvalues(self, currency: str) -> pd.DataFrame:
        if currency not in self.results:
            return pd.DataFrame()
        pvals = self.results[currency]['rolling']['pvalues'].copy()
        pvals.columns = [get_display_name(c) for c in pvals.columns]
        return pvals
    
    def get_all_signals(self) -> pd.DataFrame:
        signals = []
        for ccy, result in self.results.items():
            if result is not None and 'current_signal' in result:
                signals.append(result['current_signal'])
        return pd.DataFrame(signals)
    
    def get_variable_importance(self, currency: str) -> pd.DataFrame:
        if currency not in self.results:
            return pd.DataFrame()
        
        var_imp = self.results[currency].get('variable_importance', {})
        rows = []
        for var, info in var_imp.items():
            rows.append({
                'Variable': info['display_name'],
                'Beta': info['beta'],
                'P-value': info['pvalue'],
                'Significatif': '✓' if info['significant'] else '',
            })
        return pd.DataFrame(rows)
    
    def get_all_variable_importance(self) -> pd.DataFrame:
        all_data = []
        for ccy in self.config.currencies:
            if ccy in self.results and self.results[ccy]:
                var_imp = self.results[ccy].get('variable_importance', {})
                for var, info in var_imp.items():
                    all_data.append({
                        'Devise': ccy,
                        'Variable': info['display_name'],
                        'Beta': info['beta'],
                        'Significatif': info['significant']
                    })
        return pd.DataFrame(all_data)


# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """
    Backtest the Fair Value strategy.
    
    Strategy rules:
    - SELL (short) when misalignment > +threshold (overvalued)
    - BUY (long) when misalignment < -threshold (undervalued)  
    - FLAT when between thresholds
    """
    
    def __init__(self, analyzer: FXFairValueAnalyzer):
        self.analyzer = analyzer
        self.results: Dict[str, Dict] = {}
    
    def run_backtest(self, 
                     currency: str,
                     entry_threshold: float = 1.0,
                     exit_threshold: float = 0.5,
                     use_zscore: bool = True,
                     transaction_cost_bps: float = 2.0) -> Dict:
        """
        Run backtest for a single currency.
        
        Parameters
        ----------
        currency : str
            Currency to backtest (e.g., 'EUR')
        entry_threshold : float
            Number of standard deviations to enter a position
        exit_threshold : float
            Number of standard deviations to exit (return to flat)
        use_zscore : bool
            If True, use z-score; if False, use raw misalignment %
        transaction_cost_bps : float
            Transaction cost in basis points (one-way)
        
        Returns
        -------
        dict with backtest results
        """
        if currency not in self.analyzer.results:
            return {}
        
        fv_df = self.analyzer.results[currency]['fair_value'].copy()
        
        # Signal column
        if use_zscore:
            signal_col = fv_df['z_score']
        else:
            # Normalize misalignment to use same threshold logic
            signal_col = fv_df['misalignment_pct'] / fv_df['misalignment_pct'].std()
        
        # Calculate FX returns (weekly)
        spot = fv_df['spot']
        fx_returns = spot.diff()
        
        # Generate positions: +1 = long, -1 = short, 0 = flat
        positions = pd.Series(index=fv_df.index, data=0.0)
        current_pos = 0.0
        
        for i in range(1, len(signal_col)):
            signal = signal_col.iloc[i-1]  # Use lagged signal (tradeable)
            
            if current_pos == 0:
                # Not in position - check for entry
                if signal > entry_threshold:
                    current_pos = -1.0  # Short (overvalued)
                elif signal < -entry_threshold:
                    current_pos = 1.0   # Long (undervalued)
            elif current_pos == 1:
                # Long position - check for exit
                if signal > -exit_threshold:
                    current_pos = 0.0
            elif current_pos == -1:
                # Short position - check for exit
                if signal < exit_threshold:
                    current_pos = 0.0
            
            positions.iloc[i] = current_pos
        
        # Calculate strategy returns
        strategy_returns = positions.shift(1) * fx_returns  # Shift to avoid look-ahead
        strategy_returns = strategy_returns.fillna(0)
        
        # Transaction costs
        position_changes = positions.diff().abs()
        costs = position_changes * (transaction_cost_bps / 10000) * spot.abs()
        strategy_returns_net = strategy_returns - costs.fillna(0)
        
        # Cumulative PnL
        cumulative_pnl = strategy_returns_net.cumsum()
        
        # Performance metrics
        total_return = cumulative_pnl.iloc[-1] if len(cumulative_pnl) > 0 else 0
        
        # Annualized return (assuming weekly data)
        n_years = len(strategy_returns_net) / 52
        ann_return = total_return / n_years if n_years > 0 else 0
        
        # Sharpe ratio (annualized)
        if strategy_returns_net.std() > 0:
            sharpe = (strategy_returns_net.mean() / strategy_returns_net.std()) * np.sqrt(52)
        else:
            sharpe = 0
        
        # Max drawdown
        rolling_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - rolling_max
        max_drawdown = drawdown.min()
        
        # Hit rate
        winning_periods = (strategy_returns_net > 0).sum()
        total_periods = (strategy_returns_net != 0).sum()
        hit_rate = winning_periods / total_periods * 100 if total_periods > 0 else 0
        
        # Number of trades
        n_trades = (position_changes > 0).sum()
        
        # Long/Short breakdown
        long_returns = strategy_returns_net[positions.shift(1) > 0]
        short_returns = strategy_returns_net[positions.shift(1) < 0]
        
        long_pnl = long_returns.sum() if len(long_returns) > 0 else 0
        short_pnl = short_returns.sum() if len(short_returns) > 0 else 0
        
        # Time in market
        time_long = (positions > 0).sum() / len(positions) * 100
        time_short = (positions < 0).sum() / len(positions) * 100
        time_flat = (positions == 0).sum() / len(positions) * 100
        
        result = {
            'currency': currency,
            'cumulative_pnl': cumulative_pnl,
            'strategy_returns': strategy_returns_net,
            'positions': positions,
            'total_return': total_return,
            'annualized_return': ann_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate,
            'n_trades': n_trades,
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'time_long_pct': time_long,
            'time_short_pct': time_short,
            'time_flat_pct': time_flat,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'transaction_cost_bps': transaction_cost_bps,
        }
        
        self.results[currency] = result
        return result
    
    def run_backtest_all(self, **kwargs) -> Dict:
        """Run backtest for all currencies."""
        for ccy in self.analyzer.config.currencies:
            if ccy in self.analyzer.results:
                self.run_backtest(ccy, **kwargs)
        return self.results
    
    def get_summary_table(self) -> pd.DataFrame:
        """Get summary table of all backtests."""
        rows = []
        for ccy, res in self.results.items():
            rows.append({
                'Currency': f"{ccy}/USD",
                'Total Return': f"{res['total_return']:.2f}",
                'Ann. Return': f"{res['annualized_return']:.2f}",
                'Sharpe': f"{res['sharpe_ratio']:.2f}",
                'Max DD': f"{res['max_drawdown']:.2f}",
                'Hit Rate': f"{res['hit_rate']:.1f}%",
                'Trades': int(res['n_trades']),
                'Long PnL': f"{res['long_pnl']:.2f}",
                'Short PnL': f"{res['short_pnl']:.2f}",
            })
        return pd.DataFrame(rows)
    
    def get_combined_pnl(self) -> pd.DataFrame:
        """Get combined PnL across all currencies."""
        pnl_dict = {}
        for ccy, res in self.results.items():
            pnl_dict[ccy] = res['cumulative_pnl']
        
        df = pd.DataFrame(pnl_dict)
        df['Portfolio'] = df.sum(axis=1)
        return df


if __name__ == "__main__":
    filepath = "data.xlsx"
    config = ModelConfig()
    
    analyzer = FXFairValueAnalyzer(filepath, config)
    analyzer.load_data()
    analyzer.analyze_all(window_months=12, selection_method='manual')
    print(analyzer.get_ols_summary())
