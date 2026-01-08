"""
FX Fair Value Model - Dashboard (Version 4)
===========================================

Professional dashboard with:
- Sidebar configuration
- English labels
- No emojis
- Clean professional design

Author: [Your name]
Date: January 2025
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from model import (
    FXFairValueAnalyzer, ModelConfig, Backtester,
    VARIABLE_NAMES, get_display_name
)


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_CONFIG = ModelConfig(
    currencies=['EUR', 'CHF', 'CAD', 'CZK'],
    years_back=10,
)

DATA_PATH = "data.xlsx"

COLORS = {
    'overvalued': '#c0392b',
    'undervalued': '#27ae60',
    'spot': '#2c3e50',
    'fair_value': '#e67e22',
    'significant': '#27ae60',
    'not_significant': '#bdc3c7',
    'sidebar_bg': '#2c3e50',
}

# Sidebar style
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "280px",
    "padding": "20px",
    "backgroundColor": COLORS['sidebar_bg'],
    "color": "white",
    "overflowY": "auto",
}

# Content style
CONTENT_STYLE = {
    "marginLeft": "300px",
    "padding": "20px",
}


# =============================================================================
# DASHBOARD
# =============================================================================

class FXDashboard:
    def __init__(self, data_path: str, config: ModelConfig):
        self.data_path = data_path
        self.config = config
        self.analyzer = None
        
        self.app = Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.FLATLY,
                "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css"
            ],
            suppress_callback_exceptions=True
        )
        
        self._load_and_analyze()
        self.app.layout = self._create_layout()
        self._register_callbacks()
    
    def _load_and_analyze(self):
        print("Loading data...")
        self.analyzer = FXFairValueAnalyzer(self.data_path, self.config)
        self.analyzer.load_data()
        self.analyzer.analyze_all(window_months=12, selection_method='manual')
        print("Dashboard ready!")
    
    def _get_available_features(self):
        if self.config.currencies:
            return self.analyzer.get_available_features(self.config.currencies[0])
        return []
    
    def _create_sidebar(self):
        """Create the sidebar with configuration options."""
        
        available_features = self._get_available_features()
        feature_options = [
            {'label': f" {get_display_name(f)}", 'value': f}
            for f in available_features
        ]
        
        return html.Div([
            # Title
            html.H4("FX Fair Value Model", className="text-white mb-4"),
            html.Hr(style={'borderColor': 'rgba(255,255,255,0.2)'}),
            
            # Regression Window
            html.Label("Regression Window", className="text-white fw-bold mb-2"),
            dcc.Slider(
                id='window-slider',
                min=1, max=5, step=1, value=3,
                marks={k: {'label': v, 'style': {'color': 'white'}} 
                       for k, v in self.config.windows_dict.items()},
                className="mb-4"
            ),
            
            html.Hr(style={'borderColor': 'rgba(255,255,255,0.2)'}),
            
            # Selection Method
            html.Label("Variable Selection", className="text-white fw-bold mb-2"),
            dcc.RadioItems(
                id='selection-mode',
                options=[
                    {'label': ' Manual', 'value': 'manual'},
                    {'label': ' LASSO (auto)', 'value': 'lasso'},
                ],
                value='manual',
                className="text-white mb-3",
                inputStyle={"marginRight": "8px"},
                labelStyle={"display": "block", "marginBottom": "8px"}
            ),
            
            # Manual Variable Selection
            html.Div([
                html.Label("Select Variables", className="text-white fw-bold mb-2"),
                dcc.Checklist(
                    id='manual-vars',
                    options=feature_options,
                    value=['RYLDIRS02Y_NSA', 'CPIH_SA_P1M1ML12', 'INTRGDP_NSA_P1M1ML12_3MMA'],
                    className="text-white",
                    inputStyle={"marginRight": "8px"},
                    labelStyle={"display": "block", "marginBottom": "6px", "fontSize": "0.9rem"}
                )
            ], id='manual-vars-container'),
            
            html.Hr(style={'borderColor': 'rgba(255,255,255,0.2)'}),
            
            # Update Button
            dbc.Button(
                "Update Analysis", 
                id='update-btn',
                color="light",
                className="w-100 mt-2"
            ),
            
            html.Hr(style={'borderColor': 'rgba(255,255,255,0.2)', 'marginTop': '20px'}),
            
            # Backtest Settings
            html.Label("Backtest Settings", className="text-white fw-bold mb-2"),
            
            html.Label("Entry Threshold (σ)", className="text-white small mt-2"),
            dcc.Slider(
                id='entry-threshold',
                min=0.5, max=2.5, step=0.25, value=0.5,
                marks={0.5: '0.5', 1.0: '1.0', 1.5: '1.5', 2.0: '2.0', 2.5: '2.5'},
                className="mb-3"
            ),
            
            html.Label("Transaction Cost (bps)", className="text-white small"),
            dcc.Input(
                id='transaction-cost',
                type='number',
                value=2.0,
                min=0, max=20, step=0.5,
                className="form-control form-control-sm mb-3",
                style={'backgroundColor': 'rgba(255,255,255,0.1)', 'color': 'white', 'border': 'none'}
            ),
            
            dbc.Button(
                "Run Backtest", 
                id='backtest-btn',
                color="warning",
                className="w-100"
            ),
            
            # Footer
            html.Hr(style={'borderColor': 'rgba(255,255,255,0.2)', 'marginTop': '30px'}),
            html.Small("Data: JPMaQS", className="text-muted d-block"),
            html.Small("Period: 2013-2023", className="text-muted d-block"),
            
        ], style=SIDEBAR_STYLE)
    
    def _create_content(self):
        """Create the main content area."""
        
        return html.Div([
            # Header
            html.H2("FX Fair Value Analysis", className="mb-1"),
            html.P("Currency valuation model based on macroeconomic fundamentals", 
                   className="text-muted mb-4"),
            
            # =================================================
            # ROW 0: Methodology Section (collapsible)
            # =================================================
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.H5("Methodology", className="mb-0 d-inline"),
                        dbc.Button(
                            "Show / Hide",
                            id="methodology-toggle",
                            color="link",
                            size="sm",
                            className="float-end"
                        )
                    ])
                ]),
                dbc.Collapse([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("Fair Value Model", className="fw-bold"),
                                html.P([
                                    "The model estimates a currency's Fair Value based on macroeconomic fundamentals. ",
                                    "It regresses cumulative FX returns on macro variables (real rates, inflation, GDP growth, etc.) ",
                                    "using a rolling window approach."
                                ], className="small"),
                                html.P([
                                    html.Strong("Interpretation: "),
                                    "When the Spot price is above the Fair Value, the currency is considered ",
                                    html.Span("overvalued", style={'color': COLORS['overvalued'], 'fontWeight': 'bold'}),
                                    " (expected to depreciate). When below, it is ",
                                    html.Span("undervalued", style={'color': COLORS['undervalued'], 'fontWeight': 'bold'}),
                                    " (expected to appreciate)."
                                ], className="small"),
                            ], width=4),
                            dbc.Col([
                                html.H6("Variable Selection", className="fw-bold"),
                                html.P([
                                    html.Strong("Manual: "), 
                                    "User selects which macro variables to include in the regression."
                                ], className="small mb-2"),
                                html.P([
                                    html.Strong("LASSO: "), 
                                    "Automatic selection using L1 regularization. The algorithm penalizes the absolute size of coefficients, ",
                                    "shrinking less important variables to zero. This prevents overfitting and identifies the most relevant drivers."
                                ], className="small mb-2"),
                                html.P([
                                    html.Strong("Rolling Regression: "),
                                    "Coefficients are re-estimated at each period using only past data (out-of-sample), ",
                                    "capturing time-varying relationships between FX and macro fundamentals."
                                ], className="small"),
                            ], width=4),
                            dbc.Col([
                                html.H6("Point-in-Time Data", className="fw-bold"),
                                html.P([
                                    "All macro data used in this model is ",
                                    html.Strong("point-in-time"),
                                    ", meaning each observation reflects only the information that was actually available at that date."
                                ], className="small mb-2"),
                                html.P([
                                    html.Strong("Why it matters: "),
                                    "Official statistics (GDP, inflation, etc.) are often revised months or years after initial release. ",
                                    "Using revised data would introduce ",
                                    html.Em("look-ahead bias"),
                                    ", making backtests unrealistically optimistic."
                                ], className="small mb-2"),
                                html.P([
                                    html.Strong("Data Source: "),
                                    "JPMaQS (J.P. Morgan Macrosynergy Quantamental System), a joint venture between J.P. Morgan and Macrosynergy. ",
                                    "This dataset provides high-quality, point-in-time macroeconomic indicators specifically designed for quantitative research and trading strategies."
                                ], className="small"),
                            ], width=4),
                        ]),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col([
                                html.H6("Key Concepts", className="fw-bold"),
                                html.Ul([
                                    html.Li([
                                        html.Strong("Misalignment: "),
                                        "Percentage deviation between market price (Spot) and model estimate (Fair Value)."
                                    ], className="small"),
                                    html.Li([
                                        html.Strong("R²: "),
                                        "Proportion of FX variance explained by the model. Higher is better."
                                    ], className="small"),
                                    html.Li([
                                        html.Strong("Statistical Significance: "),
                                        "* means p<0.05 (95% confidence), ** means p<0.01 (99% confidence)."
                                    ], className="small"),
                                ], className="mb-0")
                            ], width=12),
                        ])
                    ])
                ], id="methodology-collapse", is_open=False)
            ], className="mb-4"),
            
            # =================================================
            # ROW 1: Key Variables Section
            # =================================================
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        "Model Variables ",
                        html.I(
                            id="variables-info-icon",
                            className="bi bi-info-circle",
                            style={'cursor': 'pointer', 'fontSize': '0.9rem'}
                        ),
                    ], className="mb-0"),
                    dbc.Tooltip(
                        "Green badge = statistically significant (p<5%). "
                        "Gray badge = not significant. "
                        "The sign (+/-) indicates the direction of the relationship with the currency.",
                        target="variables-info-icon",
                        placement="right"
                    )
                ]),
                dbc.CardBody([
                    html.P("Selected variables and their impact on each currency pair", 
                           className="text-muted small mb-3"),
                    html.Div(id='variables-importance-section')
                ])
            ], className="mb-4"),
            
            # =================================================
            # ROW 2: Main Chart
            # =================================================
            dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col(html.H5("Spot vs Fair Value", className="mb-0"), width=9),
                        dbc.Col([
                            dcc.Dropdown(
                                id='main-ccy-dropdown',
                                options=[{'label': f"{c}/USD", 'value': c} for c in self.config.currencies],
                                value=self.config.currencies[0],
                                clearable=False,
                                style={'minWidth': '120px'}
                            )
                        ], width=3)
                    ], align="center")
                ]),
                dbc.CardBody([
                    dcc.Graph(id='main-fair-value-graph', config={'displayModeBar': False})
                ])
            ], className="mb-4"),
            
            # =================================================
            # ROW 3: Regression Results (full width)
            # =================================================
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        "Regression Coefficients ",
                        html.I(
                            id="coefficients-info-icon",
                            className="bi bi-info-circle",
                            style={'cursor': 'pointer', 'fontSize': '0.9rem'}
                        ),
                    ], className="mb-0"),
                    dbc.Tooltip(
                        "Beta coefficients from OLS regression over the full sample period. "
                        "* indicates significance at 5% level, ** at 1% level. "
                        "R² shows the percentage of FX variance explained by the model.",
                        target="coefficients-info-icon",
                        placement="right"
                    )
                ]),
                dbc.CardBody([
                    html.Div(id='ols-table-container', style={'overflowX': 'auto'})
                ])
            ], className="mb-4"),
            
            # =================================================
            # ROW 4: Rolling Betas (full width)
            # =================================================
            dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col([
                            html.H5([
                                "Rolling Betas Over Time ",
                                html.I(
                                    id="rolling-info-icon",
                                    className="bi bi-info-circle",
                                    style={'cursor': 'pointer', 'fontSize': '0.9rem'}
                                ),
                            ], className="mb-0"),
                            dbc.Tooltip(
                                "Evolution of regression coefficients over time using a rolling window. "
                                "Colored segments indicate statistical significance (p<5%), "
                                "gray segments indicate non-significance. "
                                "This shows how the relationship between FX and macro variables changes over time.",
                                target="rolling-info-icon",
                                placement="right"
                            )
                        ], width=9),
                        dbc.Col([
                            dcc.Dropdown(
                                id='betas-ccy-dropdown',
                                options=[{'label': f"{c}/USD", 'value': c} for c in self.config.currencies],
                                value=self.config.currencies[0],
                                clearable=False
                            )
                        ], width=3)
                    ], align="center")
                ]),
                dbc.CardBody([
                    dcc.Graph(id='rolling-betas-graph', config={'displayModeBar': False})
                ])
            ], className="mb-4"),
            
            # =================================================
            # ROW 5: Misalignment Comparison
            # =================================================
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        "Misalignment Comparison ",
                        html.I(
                            id="misalignment-info-icon",
                            className="bi bi-info-circle",
                            style={'cursor': 'pointer', 'fontSize': '0.9rem'}
                        ),
                    ], className="mb-0"),
                    dbc.Tooltip(
                        "Misalignment = (Spot - Fair Value) / Fair Value in %. "
                        "Positive values indicate the currency is overvalued (Spot > Fair Value), "
                        "negative values indicate undervaluation. "
                        "Extreme readings (beyond +/- 2 standard deviations) may signal trading opportunities.",
                        target="misalignment-info-icon",
                        placement="right"
                    )
                ]),
                dbc.CardBody([
                    dcc.Graph(id='misalignment-comparison-graph', config={'displayModeBar': False})
                ])
            ], className="mb-4"),
            
            # =================================================
            # ROW 6: Backtest Results
            # =================================================
            dbc.Card([
                dbc.CardHeader([
                    html.H5([
                        "Backtest Results ",
                        html.I(
                            id="backtest-info-icon",
                            className="bi bi-info-circle",
                            style={'cursor': 'pointer', 'fontSize': '0.9rem'}
                        ),
                    ], className="mb-0"),
                    dbc.Tooltip(
                        "Strategy: Go LONG when currency is undervalued (misalignment < -threshold), "
                        "go SHORT when overvalued (misalignment > +threshold). "
                        "Exit when misalignment returns toward zero. "
                        "PnL is calculated using weekly returns with transaction costs deducted.",
                        target="backtest-info-icon",
                        placement="right"
                    )
                ]),
                dbc.CardBody([
                    html.Div(id='backtest-results-container', children=[
                        html.P("Click 'Run Backtest' to see results", className="text-muted text-center my-4")
                    ])
                ])
            ], className="mb-4"),
            
        ], style=CONTENT_STYLE)
    
    def _create_layout(self):
        """Create the full layout with sidebar and content."""
        return html.Div([
            self._create_sidebar(),
            self._create_content()
        ])
    
    def _register_callbacks(self):
        
        @self.app.callback(
            Output("methodology-collapse", "is_open"),
            Input("methodology-toggle", "n_clicks"),
            State("methodology-collapse", "is_open"),
        )
        def toggle_methodology(n_clicks, is_open):
            if n_clicks:
                return not is_open
            return is_open
        
        @self.app.callback(
            Output('manual-vars-container', 'style'),
            Input('selection-mode', 'value')
        )
        def toggle_manual_vars(mode):
            if mode == 'manual':
                return {'display': 'block'}
            return {'display': 'none'}
        
        @self.app.callback(
            [
                Output('ols-table-container', 'children'),
                Output('variables-importance-section', 'children'),
            ],
            Input('update-btn', 'n_clicks'),
            [
                State('window-slider', 'value'),
                State('selection-mode', 'value'),
                State('manual-vars', 'value')
            ]
        )
        def update_analysis(n_clicks, window_idx, selection_mode, manual_vars):
            window_str = self.config.windows_dict[window_idx]
            window_months = int(window_str.replace('M', ''))
            
            self.analyzer.analyze_all(
                window_months=window_months,
                selection_method=selection_mode,
                manual_vars=manual_vars if selection_mode == 'manual' else None
            )
            
            # === OLS TABLE ===
            ols_summary = self.analyzer.get_ols_summary()
            if not ols_summary.empty:
                ols_table = dbc.Table.from_dataframe(
                    ols_summary,
                    striped=True, bordered=True, hover=True,
                    className="small text-center mb-0"
                )
            else:
                ols_table = html.P("No data available")
            
            # === VARIABLES IMPORTANCE SECTION ===
            var_cards = []
            
            for ccy in self.config.currencies:
                if ccy not in self.analyzer.results or self.analyzer.results[ccy] is None:
                    continue
                
                result = self.analyzer.results[ccy]
                var_imp = result.get('variable_importance', {})
                lasso_info = result.get('lasso_info', {})
                
                # Variable badges
                var_badges = []
                for var, info in var_imp.items():
                    badge_color = "success" if info['significant'] else "secondary"
                    sign = "+" if info['beta'] > 0 else ""
                    var_badges.append(
                        dbc.Badge(
                            f"{info['display_name']} ({sign}{info['beta']:.3f})",
                            color=badge_color,
                            className="me-1 mb-1"
                        )
                    )
                
                # Method info
                if lasso_info.get('method') == 'lasso':
                    method_text = f"LASSO selection (alpha={lasso_info.get('alpha', 0):.4f})"
                else:
                    method_text = "Manual selection"
                
                r2_value = result['ols']['r2_adj'] * 100 if result.get('ols') else 0
                
                card = dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.Strong(f"{ccy}/USD"),
                            className="py-2"
                        ),
                        dbc.CardBody([
                            html.Div(var_badges, className="mb-2"),
                            html.Hr(className="my-2"),
                            html.Small([
                                html.Span(method_text, className="text-muted"),
                                html.Br(),
                                html.Span(f"R² = {r2_value:.1f}%", className="text-muted")
                            ])
                        ], className="py-2")
                    ], className="h-100")
                ], width=3, className="mb-3")
                
                var_cards.append(card)
            
            variables_section = dbc.Row(var_cards)
            
            return ols_table, variables_section
        
        @self.app.callback(
            Output('main-fair-value-graph', 'figure'),
            [
                Input('update-btn', 'n_clicks'),
                Input('main-ccy-dropdown', 'value')
            ]
        )
        def update_main_graph(n_clicks, currency):
            if currency not in self.analyzer.results or self.analyzer.results[currency] is None:
                return go.Figure()
            
            fv_df = self.analyzer.results[currency]['fair_value']
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    f'{currency}/USD - Spot vs Fair Value',
                    'Misalignment (%)'
                ),
                vertical_spacing=0.12,
                row_heights=[0.6, 0.4]
            )
            
            # Plot 1: Spot vs Fair Value
            fig.add_trace(
                go.Scatter(
                    x=fv_df.index, y=fv_df['spot'],
                    name='Spot', 
                    line=dict(color=COLORS['spot'], width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=fv_df.index, y=fv_df['fair_value'],
                    name='Fair Value', 
                    line=dict(color=COLORS['fair_value'], width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            # Confidence band
            std_misalign = fv_df['misalignment_pct'].std() / 100
            upper = fv_df['fair_value'] * (1 + std_misalign)
            lower = fv_df['fair_value'] * (1 - std_misalign)
            
            fig.add_trace(
                go.Scatter(x=fv_df.index, y=upper, mode='lines', line=dict(width=0), showlegend=False),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=fv_df.index, y=lower, mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(230,126,34,0.1)',
                    name='Fair Value Band'
                ),
                row=1, col=1
            )
            
            # Plot 2: Misalignment
            colors = [COLORS['overvalued'] if x > 0 else COLORS['undervalued'] 
                     for x in fv_df['misalignment_pct']]
            
            fig.add_trace(
                go.Bar(
                    x=fv_df.index, y=fv_df['misalignment_pct'],
                    name='Misalignment',
                    marker_color=colors,
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.add_hline(y=0, line_dash="solid", line_color="#bdc3c7", line_width=1, row=2, col=1)
            
            # Thresholds
            std = fv_df['misalignment_pct'].std()
            fig.add_hline(y=2*std, line_dash="dot", line_color=COLORS['overvalued'], 
                         annotation_text="+2σ", row=2, col=1)
            fig.add_hline(y=-2*std, line_dash="dot", line_color=COLORS['undervalued'],
                         annotation_text="-2σ", row=2, col=1)
            
            fig.update_layout(
                height=600,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
                margin=dict(t=60, b=40)
            )
            
            fig.update_yaxes(title_text="Cumulative Level", row=1, col=1)
            fig.update_yaxes(title_text="Misalignment (%)", row=2, col=1)
            
            return fig
        
        @self.app.callback(
            Output('rolling-betas-graph', 'figure'),
            [
                Input('update-btn', 'n_clicks'),
                Input('betas-ccy-dropdown', 'value')
            ]
        )
        def update_betas_graph(n_clicks, currency):
            betas = self.analyzer.get_rolling_betas(currency)
            pvals = self.analyzer.get_rolling_pvalues(currency)
            
            if betas.empty:
                return go.Figure()
            
            fig = go.Figure()
            
            colors_list = ['#3498db', '#e74c3c', '#27ae60', '#9b59b6', '#f39c12']
            
            col_idx = 0
            for col in betas.columns:
                if col == 'Constant':
                    continue
                
                col_betas = betas[col]
                col_pvals = pvals[col] if col in pvals.columns else pd.Series(1, index=betas.index)
                
                sig_mask = col_pvals < 0.05
                sig_betas = col_betas.where(sig_mask, np.nan)
                nonsig_betas = col_betas.where(~sig_mask, np.nan)
                
                color = colors_list[col_idx % len(colors_list)]
                
                fig.add_trace(go.Scatter(
                    x=sig_betas.index, y=sig_betas.values,
                    name=col,
                    mode='lines', line=dict(width=2, color=color)
                ))
                
                fig.add_trace(go.Scatter(
                    x=nonsig_betas.index, y=nonsig_betas.values,
                    showlegend=False,
                    mode='lines', line=dict(color='#ecf0f1', width=1)
                ))
                
                col_idx += 1
            
            fig.add_hline(y=0, line_dash="solid", line_color="#bdc3c7", line_width=1)
            
            fig.update_layout(
                height=280,
                margin=dict(l=50, r=20, t=20, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                yaxis_title="Beta"
            )
            
            return fig
        
        @self.app.callback(
            Output('misalignment-comparison-graph', 'figure'),
            Input('update-btn', 'n_clicks')
        )
        def update_misalignment_comparison(n_clicks):
            fig = go.Figure()
            
            colors_list = ['#2c3e50', '#e74c3c', '#27ae60', '#8e44ad']
            
            for i, ccy in enumerate(self.config.currencies):
                if ccy not in self.analyzer.results or self.analyzer.results[ccy] is None:
                    continue
                
                fv_df = self.analyzer.results[ccy]['fair_value']
                
                fig.add_trace(go.Scatter(
                    x=fv_df.index, y=fv_df['misalignment_pct'],
                    name=f"{ccy}/USD", 
                    mode='lines', 
                    line=dict(width=1.5, color=colors_list[i % len(colors_list)])
                ))
            
            fig.add_hline(y=0, line_dash="solid", line_color="#bdc3c7", line_width=1)
            
            fig.update_layout(
                height=280,
                margin=dict(l=50, r=20, t=20, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                yaxis_title="Misalignment (%)",
                xaxis_title=""
            )
            
            return fig
        
        @self.app.callback(
            Output('backtest-results-container', 'children'),
            Input('backtest-btn', 'n_clicks'),
            [
                State('entry-threshold', 'value'),
                State('transaction-cost', 'value')
            ]
        )
        def run_backtest(n_clicks, entry_threshold, transaction_cost):
            if not n_clicks:
                return html.P("Click 'Run Backtest' to see results", className="text-muted text-center my-4")
            
            # Run backtest
            backtester = Backtester(self.analyzer)
            backtester.run_backtest_all(
                entry_threshold=entry_threshold,
                exit_threshold=entry_threshold * 0.5,
                transaction_cost_bps=transaction_cost
            )
            
            # Summary table
            summary_df = backtester.get_summary_table()
            
            # Combined PnL chart
            combined_pnl = backtester.get_combined_pnl()
            
            # Create PnL chart
            fig_pnl = go.Figure()
            
            colors_list = ['#2c3e50', '#e74c3c', '#27ae60', '#8e44ad']
            
            for i, ccy in enumerate(self.config.currencies):
                if ccy in combined_pnl.columns:
                    fig_pnl.add_trace(go.Scatter(
                        x=combined_pnl.index,
                        y=combined_pnl[ccy],
                        name=f"{ccy}/USD",
                        mode='lines',
                        line=dict(width=1.5, color=colors_list[i % len(colors_list)])
                    ))
            
            # Add portfolio line
            if 'Portfolio' in combined_pnl.columns:
                fig_pnl.add_trace(go.Scatter(
                    x=combined_pnl.index,
                    y=combined_pnl['Portfolio'],
                    name='Portfolio (sum)',
                    mode='lines',
                    line=dict(width=3, color='#f39c12', dash='dash')
                ))
            
            fig_pnl.add_hline(y=0, line_dash="solid", line_color="#bdc3c7", line_width=1)
            
            fig_pnl.update_layout(
                height=350,
                margin=dict(l=50, r=20, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                yaxis_title="Cumulative PnL",
                xaxis_title=""
            )
            
            # Calculate portfolio metrics
            portfolio_returns = combined_pnl['Portfolio'].diff().fillna(0) if 'Portfolio' in combined_pnl.columns else pd.Series([0])
            portfolio_total = combined_pnl['Portfolio'].iloc[-1] if 'Portfolio' in combined_pnl.columns else 0
            n_years = len(portfolio_returns) / 52
            portfolio_ann_return = portfolio_total / n_years if n_years > 0 else 0
            portfolio_sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(52) if portfolio_returns.std() > 0 else 0
            portfolio_max_dd = (combined_pnl['Portfolio'] - combined_pnl['Portfolio'].cummax()).min() if 'Portfolio' in combined_pnl.columns else 0
            
            # Build results layout
            results_layout = html.Div([
                # Metrics cards
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Portfolio Total Return", className="text-muted mb-1"),
                                html.H4(f"{portfolio_total:.2f}", 
                                       style={'color': COLORS['undervalued'] if portfolio_total > 0 else COLORS['overvalued']})
                            ])
                        ])
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Annualized Return", className="text-muted mb-1"),
                                html.H4(f"{portfolio_ann_return:.2f}")
                            ])
                        ])
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Sharpe Ratio", className="text-muted mb-1"),
                                html.H4(f"{portfolio_sharpe:.2f}")
                            ])
                        ])
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Max Drawdown", className="text-muted mb-1"),
                                html.H4(f"{portfolio_max_dd:.2f}", style={'color': COLORS['overvalued']})
                            ])
                        ])
                    ], width=3),
                ], className="mb-4"),
                
                # PnL Chart
                html.H6("Cumulative PnL by Currency", className="fw-bold mb-2"),
                dcc.Graph(figure=fig_pnl, config={'displayModeBar': False}),
                
                # Summary table
                html.H6("Performance by Currency", className="fw-bold mt-4 mb-2"),
                dbc.Table.from_dataframe(
                    summary_df,
                    striped=True, bordered=True, hover=True,
                    className="small text-center"
                ),
                
                # Strategy description
                html.Hr(),
                html.Small([
                    html.Strong("Strategy: "),
                    f"Entry at ±{entry_threshold}σ, Exit at ±{entry_threshold*0.5}σ, Transaction cost: {transaction_cost} bps"
                ], className="text-muted")
            ])
            
            return results_layout
    
    def run(self, debug=False, port=8050):
        print(f"\n{'='*50}")
        print("FX FAIR VALUE MODEL - DASHBOARD")
        print(f"{'='*50}")
        print(f"\nOpen http://localhost:{port}")
        print("Press Ctrl+C to stop\n")
        
        self.app.run(debug=debug, port=port)


if __name__ == '__main__':
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data.xlsx"
    dashboard = FXDashboard(data_path, MODEL_CONFIG)
    dashboard.run(debug=False, port=8050)
