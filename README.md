# HMM Regime Detection Trading System

A comprehensive quantitative trading system that uses Hidden Markov Models (HMM) with GARCH volatility integration for market regime detection and algorithmic trading.

## 🎯 Overview

This system implements a sophisticated 3-state HMM to detect Bull/Bear/Sideways market regimes using SPY data, integrated with GARCH(1,1) volatility modeling for enhanced signal quality. The system provides comprehensive backtesting, performance analytics, and forward-looking predictions with actionable trading signals.

## 🚀 Key Features

### Core Functionality
- **3-State HMM Regime Detection**: Bull, Sideways, Bear market identification
- **GARCH(1,1) Integration**: Volatility forecasting and standardized residuals as HMM observations
- **Advanced Backtesting**: Walk-forward analysis with transaction costs and risk management
- **Real-time Predictions**: 20-day forward regime forecasts with confidence intervals
- **Comprehensive Analytics**: 50+ performance metrics and statistical tests

### Technical Specifications
- **Data Source**: Yahoo Finance (SPY 2014-2024, 10 years of data)
- **Models**: GARCH(1,1) + 3-state Gaussian HMM
- **Strategy**: Regime-based position management with volatility scaling
- **Risk Management**: Position limits, transaction costs (0.1%), drawdown controls
- **Validation**: Statistical significance tests and model diagnostics

## 📁 Project Structure

```
hmm_regime_trading/
├── data/
│   ├── data_loader.py          # SPY data acquisition & preprocessing
│   └── data_utils.py           # Data validation utilities
├── models/
│   ├── garch_model.py          # GARCH(1,1) implementation
│   ├── hmm_model.py            # 3-state HMM with Gaussian emissions
│   └── model_utils.py          # Model validation & diagnostics
├── trading/
│   ├── strategy.py             # Regime-based trading strategy
│   ├── backtester.py           # Comprehensive backtesting framework
│   └── risk_manager.py         # Position sizing & risk controls
├── analysis/
│   ├── performance.py          # Performance metrics calculation
│   ├── visualization.py        # Plotting & visualization suite
│   └── prediction.py           # Forward regime prediction & signals
├── config/
│   └── settings.py             # Configuration parameters
├── requirements.txt            # Python dependencies
├── main.py                     # Main execution script
└── README.md                   # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start

1. **Clone the repository**:
```bash
git clone <repository-url>
cd hmm_regime_trading
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the system**:
```bash
python main.py
```

### Dependencies
```
numpy==1.24.3
pandas==2.0.3
yfinance==0.2.18
scipy==1.11.1
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
arch==6.2.0
hmmlearn==0.3.0
statsmodels==0.14.0
plotly==5.15.0
tqdm==4.65.0
```

## 💻 Usage

### Command Line Interface

```bash
# Full analysis (default)
python main.py

# Backtesting only
python main.py --mode backtest

# Predictions only
python main.py --mode predict

# Custom parameters
python main_with_predictions.py --symbol SPY --start-date 2020-01-01 --end-date 2025-09-23

# Verbose output
python main.py --verbose

# Custom stock
python main_with_predictions.py --symbol NVDA
```

### Programmatic Usage

```python
from main import HMMTradingSystem

# Initialize system
system = HMMTradingSystem()

# Run full analysis
results = system.run_full_analysis()

# Access components
data = results['data_summary']
performance = results['performance_analysis']
forecasts = results['forecast_results']
```

## 📊 Output & Results

### Generated Files
- **Performance Plots**: Regime detection, equity curves, drawdown analysis
- **Risk Analysis**: VaR metrics, correlation analysis, rolling statistics
- **Prediction Charts**: Forward regime probabilities and trading signals
- **Log Files**: Detailed execution logs and diagnostics
- **Performance Report**: Comprehensive text-based analysis

### Key Metrics Provided
- **Returns**: Total, annualized, excess returns vs benchmark
- **Risk-Adjusted**: Sharpe, Sortino, Calmar, Information ratios
- **Risk Metrics**: Maximum drawdown, VaR, beta, correlation
- **Trading Stats**: Win rate, profit factor, average trade duration
- **Regime Analysis**: Performance by market regime
- **Statistical Tests**: Normality, stationarity, significance tests

## 🔧 Configuration

### Model Parameters
```python
# GARCH Configuration
GARCH_CONFIG = {
    'p': 1,                 # GARCH lag order
    'q': 1,                 # ARCH lag order
    'mean': 'Zero',         # Mean model
    'dist': 'Normal'        # Error distribution
}

# HMM Configuration
HMM_CONFIG = {
    'n_components': 3,      # Bull, Sideways, Bear
    'covariance_type': 'full',
    'n_iter': 1000,
    'random_state': 42
}

# Trading Configuration
TRADING_CONFIG = {
    'transaction_cost': 0.001,  # 0.1% per trade
    'max_position': 1.0,        # 100% max position
    'risk_free_rate': 0.02      # 2% annual
}
```

### Regime Strategy
- **Bull Regime (0)**: 100% long position
- **Sideways Regime (1)**: 0% (flat/cash)
- **Bear Regime (2)**: 100% short position

Position sizes are adjusted based on:
- Volatility forecasts (inverse volatility scaling)
- Regime confidence levels
- Risk management rules

## 📈 Performance Features

### Backtesting Framework
- **Walk-Forward Analysis**: Rolling window validation
- **Transaction Costs**: Realistic 0.1% per trade
- **Risk Management**: Position limits and stop-losses
- **Benchmark Comparison**: Against buy-and-hold SPY

### Analytics Suite
- **50+ Performance Metrics**: Comprehensive risk-return analysis
- **Regime-Specific Analysis**: Performance breakdown by market regime
- **Rolling Statistics**: Time-varying performance metrics
- **Statistical Validation**: Significance tests and model diagnostics

### Visualization
- **Interactive Dashboards**: Plotly-based interactive charts
- **Static Plots**: High-quality matplotlib/seaborn visualizations
- **Regime Overlay**: Price charts with regime background colors
- **Performance Attribution**: Detailed performance breakdowns

## 🔮 Prediction & Signals

### Forward Prediction
- **20-Day Horizon**: Regime probability forecasts
- **Confidence Intervals**: 95% confidence bands
- **Signal Strength**: Categorized signal quality (Strong/Moderate/Weak)
- **Risk Assessment**: Risk level for each prediction

### Trading Signals
```python
# Example signal output
{
    'regime': 0,                    # Bull regime
    'regime_probability': 0.85,     # 85% confidence
    'action': 'BUY',               # Recommended action
    'position_size': '75%',        # Suggested position
    'risk_level': 'Medium',        # Risk assessment
    'entry_price': 'Market',       # Entry recommendation
    'stop_loss': '5% below entry', # Risk management
    'take_profit': '15% above entry',
    'hold_period': '5-20 days'
}
```

## 🧪 Model Validation

### GARCH Diagnostics
- Ljung-Box tests for residual autocorrelation
- Jarque-Bera normality tests
- AIC/BIC model selection criteria
- Volatility clustering analysis

### HMM Validation
- Cross-validation accuracy
- Regime stability analysis
- Transition probability validation
- State interpretability assessment

### Strategy Testing
- Out-of-sample testing
- Monte Carlo simulations
- Sensitivity analysis
- Robustness checks

## ⚠️ Risk Disclaimers

### Important Warnings
- **Past Performance**: Does not guarantee future results
- **Model Risk**: Models may fail during market stress
- **Transaction Costs**: Real costs may vary from assumptions
- **Market Risk**: All trading involves substantial risk of loss

### Risk Management Features
- Position size limits
- Volatility-adjusted sizing
- Confidence-based scaling
- Stop-loss mechanisms
- Drawdown monitoring

## 🔬 Academic Background

### Methodology
This system implements state-of-the-art quantitative finance techniques:

1. **GARCH Modeling**: Autoregressive Conditional Heteroskedasticity for volatility
2. **Hidden Markov Models**: Unobserved regime state estimation
3. **Viterbi Algorithm**: Most likely state sequence decoding
4. **Forward-Backward Algorithm**: State probability computation
5. **Walk-Forward Analysis**: Realistic backtesting methodology

### References
- Baum, L.E. (1972). "An Inequality and Associated Maximization Technique"
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"
- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series"
- Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models"

## 📞 Support & Contributing

### Issues
Report bugs and request features through the issue tracker.

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Submit a pull request

### Development Setup
```bash
# Development installation
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black hmm_regime_trading/
flake8 hmm_regime_trading/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤖 Generated with Claude Code

This comprehensive trading system was developed using Claude Code, Anthropic's official CLI for Claude. The system demonstrates advanced quantitative finance techniques implemented with production-quality code architecture, comprehensive testing, and detailed documentation.

**For educational and research purposes only. Not financial advice.**