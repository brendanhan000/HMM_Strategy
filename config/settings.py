"""
Configuration settings for HMM Regime Detection Trading System
"""

import numpy as np
from datetime import datetime, timedelta

# Data Configuration
DATA_CONFIG = {
    'symbol': 'SPY',
    'start_date': '2014-01-01',
    'end_date': '2025-09-23',
    'train_ratio': 0.7,  # 70% for training, 30% for testing
}

# GARCH Model Configuration
GARCH_CONFIG = {
    'p': 1,  # GARCH lag order
    'q': 1,  # ARCH lag order
    'mean': 'Zero',  # Mean model
    'vol': 'GARCH',  # Volatility model
    'dist': 'Normal',  # Error distribution
}

# HMM Model Configuration
HMM_CONFIG = {
    'n_components': 3,  # Bull, Sideways, Bear
    'covariance_type': 'full',  # Full covariance matrix
    'n_iter': 1000,  # Maximum iterations
    'tol': 1e-6,  # Convergence tolerance
    'random_state': 42,
    'algorithm': 'viterbi',
}

# Regime Labels
REGIME_LABELS = {
    0: 'Bull',
    1: 'Sideways',
    2: 'Bear'
}

REGIME_COLORS = {
    0: 'green',
    1: 'yellow',
    2: 'red'
}

# Trading Strategy Configuration
TRADING_CONFIG = {
    'transaction_cost': 0.001,  # 0.1% per trade
    'max_position': 1.0,  # Maximum position size (100%)
    'min_position': -1.0,  # Minimum position size (-100%)
    'rebalance_frequency': 'daily',
    'risk_free_rate': 0.02,  # 2% annual risk-free rate
    'leverage': 1.0,  # No leverage
}

# Position sizing based on regime
REGIME_POSITIONS = {
    0: 1.0,   # Bull: 100% long
    1: 0.0,   # Sideways: 0% (flat)
    2: -1.0,  # Bear: 100% short
}

# Performance Analysis Configuration
PERFORMANCE_CONFIG = {
    'benchmark': 'SPY',
    'periods_per_year': 252,  # Trading days per year
    'confidence_level': 0.95,
    'var_alpha': 0.05,  # Value at Risk alpha
}

# Visualization Configuration
VIZ_CONFIG = {
    'figsize': (15, 10),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'husl',
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'forecast_horizon': 20,  # 20 trading days
    'confidence_intervals': [0.05, 0.95],  # 95% CI
    'min_prob_threshold': 0.6,  # Minimum probability for regime signal
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': 'hmm_trading.log',
}